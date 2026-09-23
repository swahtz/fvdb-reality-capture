# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from ..enums import CameraModel, ProjectionMethod
from ..functional import Crop, apply_crop, resolve_projection_method

from .gaussian_splat_dataset import SfmDataset
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats

if TYPE_CHECKING:
    from .gaussian_splat_reconstruction import GaussianSplatReconstructionConfig


def _needs_depth_render(config: "GaussianSplatReconstructionConfig") -> bool:
    return config.sparse_depth_reg > 0.0 or config.dense_depth_reg > 0.0


@dataclass
class RenderOutputs:
    """
    Primary tensors returned by a render backend.

    This groups the main image-like outputs produced by a backend into a named structure so
    call sites can access them by meaning rather than by tuple position. The ``image`` tensor
    contains the primary rendered channels, ``alpha`` contains the accumulated opacity image,
    and ``depth`` is included only when the selected render path also produces a depth channel.
    """

    image: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor | None = None


RenderBackendName = Literal["image_space", "world_space"]


class TrainingView(Protocol):
    """
    The per-image part of a training forward pass, ready to render any number of crops.

    A backend does the work that does not depend on the crop once per view (projection for the
    image-space backend, the full render for the world-space one) and returns one of these, so the
    training loop can render each crop without repeating it.
    """

    def render_crop(self, crop: Crop) -> RenderOutputs:
        """
        Render one crop of this view.

        Args:
            crop (tuple[int, int, int, int]): Crop rectangle as ``(origin_w, origin_h, width, height)``.

        Returns:
            RenderOutputs: The rendered crop, alpha image, and optional depth image.
        """
        ...


def _split_render_outputs(rendered: torch.Tensor, alpha: torch.Tensor, num_channels: int) -> RenderOutputs:
    image = rendered[..., :num_channels]
    depth = rendered[..., -1:] if rendered.shape[-1] == num_channels + 1 else None
    return RenderOutputs(image=image, alpha=alpha, depth=depth)


@dataclass
class _ProjectedTrainingView:
    """Image-space view: the Gaussians are projected once, each crop rasterizes from that projection."""

    model: GaussianSplat3d
    projected_gaussians: ProjectedGaussianSplats
    tile_size: int

    def render_crop(self, crop: Crop) -> RenderOutputs:
        crop_origin_w, crop_origin_h, crop_w, crop_h = crop
        rendered, alphas = self.model.render_from_projected_gaussians(
            self.projected_gaussians,
            crop_width=crop_w,
            crop_height=crop_h,
            crop_origin_w=crop_origin_w,
            crop_origin_h=crop_origin_h,
            tile_size=self.tile_size,
        )
        return _split_render_outputs(rendered, alphas, self.model.num_channels)


@dataclass
class _RenderedTrainingView:
    """World-space view: the full image is rendered once, each crop is a slice of it."""

    rendered: torch.Tensor
    alpha: torch.Tensor
    num_channels: int

    def render_crop(self, crop: Crop) -> RenderOutputs:
        rendered, alpha = apply_crop(self.rendered, self.alpha, crop)
        return _split_render_outputs(rendered, alpha, self.num_channels)


def projection_method_from_config(value: str) -> ProjectionMethod:
    mapping = {
        "auto": ProjectionMethod.AUTO,
        "analytic": ProjectionMethod.ANALYTIC,
        "unscented": ProjectionMethod.UNSCENTED,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported projection_method {value}")
    return mapping[value]


def _forward_only_camera_models(dataset: SfmDataset, config: "GaussianSplatReconstructionConfig") -> list[CameraModel]:
    """Camera models in ``dataset`` that resolve to the unscented projection under ``config``.

    That projection has no backward pass, so rasterizing from it in image space gives the Gaussian
    geometry no gradient.
    """
    projection_method = projection_method_from_config(config.projection_method)
    return [
        CameraModel(int(camera_model))
        for camera_model in torch.unique(torch.from_numpy(dataset.camera_models)).tolist()
        if resolve_projection_method(CameraModel(int(camera_model)), projection_method) == ProjectionMethod.UNSCENTED
    ]


def resolve_training_backend(
    backend: "RenderBackend",
    dataset: SfmDataset,
    config: "GaussianSplatReconstructionConfig",
    logger: logging.Logger,
) -> "RenderBackend":
    """Return a backend that can train the cameras in ``dataset``, swapping image space for world space if needed.

    The image-space backend cannot train cameras that need the unscented projection, since that
    projection is forward-only. Rather than fail, such scenes are rendered through the world-space
    backend, which differentiates through the cameras directly. The switch is logged once. The
    2D-gradient statistics that drive Gaussian densification are only accumulated by the analytic
    projection, so refinement skips insertion on these scenes; the optimizer reports that when it
    first happens.
    """
    if not isinstance(backend, ImageSpaceRenderBackend):
        return backend
    forward_only = _forward_only_camera_models(dataset, config)
    if not forward_only:
        return backend
    names = ", ".join(camera_model.name for camera_model in forward_only)
    logger.warning(
        f"Camera models {names} use the unscented projection under projection_method={config.projection_method!r}, "
        "which has no backward pass, so the image-space render backend cannot train them. Rendering through the "
        "world-space backend instead. Gaussian densification statistics are not accumulated on this path, so "
        "refinement will not insert new Gaussians. Undistort the images to train with the image-space backend."
    )
    return WorldSpaceRenderBackend()


def _camera_model_from_batch(camera_models: torch.Tensor) -> CameraModel:
    unique_camera_models = torch.unique(camera_models.to(dtype=torch.int32))
    if unique_camera_models.numel() != 1:
        raise NotImplementedError("Rendering a minibatch with multiple camera models is not supported")
    return CameraModel(int(unique_camera_models.item()))


def _distortion_coeffs_for_batch(
    camera_model: CameraModel, distortion_coeffs: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    if camera_model in (CameraModel.PINHOLE, CameraModel.ORTHOGRAPHIC):
        return None
    return distortion_coeffs.to(device)


class RenderBackend(Protocol):
    """
    Interface implemented by Gaussian splat rendering backends.

    A render backend encapsulates one concrete strategy for rendering Gaussian splats, such as
    image-space projection or world-space rendering. Backends are responsible for validating that
    the scene cameras are compatible with the chosen rendering path and for producing the tensors
    consumed during training and evaluation.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Validate that the backend can render the camera models present in a scene.

        Implementations should raise an exception early if the dataset contains camera models,
        distortion settings, or batching patterns that the backend cannot support.

        Args:
            model (GaussianSplat3d): Gaussian splat model that will be rendered by the backend.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        ...

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> TrainingView:
        """
        Do the per-image work of a training forward pass and return a view that renders its crops.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            TrainingView: Renders any crop of this view via :meth:`TrainingView.render_crop`.
        """
        ...

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image and return the primary image outputs.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image, alpha image, and optional depth image.
        """
        ...


class ImageSpaceRenderBackend:
    """
    Backend that projects Gaussians in image space before rasterization.

    This backend first projects Gaussians into each camera view using the image-space FVDB APIs
    and then rasterizes those projected Gaussians. It is the natural fit for the classic 3DGS
    rendering path and for renderers that need projected-gaussian intermediates during training.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Probe the scene cameras to ensure image-space rendering supports them.

        Cameras that resolve to the unscented projection are rejected. That projection is forward-only,
        so rasterizing from it would train features and opacities while the Gaussian geometry received
        no gradient at all. :func:`resolve_training_backend` routes such scenes to the world-space
        backend before this check is reached.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        if config.batch_size > 1 and torch.unique(torch.from_numpy(dataset.camera_models)).numel() > 1:
            raise NotImplementedError("batch_size > 1 is not supported for scenes with multiple camera models")
        forward_only = _forward_only_camera_models(dataset, config)
        if forward_only:
            names = ", ".join(camera_model.name for camera_model in forward_only)
            raise ValueError(
                f"The image-space render backend cannot train {names} cameras with the "
                f"{config.projection_method!r} projection method: the unscented projection is forward-only, so "
                'the Gaussian geometry would receive no gradient. Use render_backend="world_space", which '
                "differentiates through these cameras directly."
            )
        self._probe(model, dataset, config, device, render_depth=_needs_depth_render(config))

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> TrainingView:
        """
        Project the Gaussians into the target cameras once; the returned view rasterizes each crop.

        If depth regularization is enabled, the projection also carries the depth channel. Tile
        intersections and opacities are cached on the projection, so rendering several crops repeats
        only the rasterization.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            TrainingView: Renders any crop of this view from the shared projection.
        """
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device)
        projection_method = projection_method_from_config(config.projection_method)
        projection_function = (
            model.project_gaussians_for_images_and_depths
            if _needs_depth_render(config)
            else model.project_gaussians_for_images
        )
        projected_gaussians = projection_function(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method,
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return _ProjectedTrainingView(model=model, projected_gaussians=projected_gaussians, tile_size=config.tile_size)

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image using image-space projection and rasterization.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image and alpha image, plus depth when provided by the
            selected render path.
        """
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device)
        image, alpha = model.render_images(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return RenderOutputs(image=image, alpha=alpha)

    @staticmethod
    def _probe(
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
        render_depth: bool,
    ) -> None:
        """
        Run a lightweight backend probe over the scene camera models.

        This helper renders a minimal example for each distinct camera model present in the
        dataset so unsupported camera/distortion combinations fail early, before optimization
        starts.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose distinct camera models should be probed.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
            render_depth (bool): Whether the probe should use the depth-producing render API.
        """
        if len(dataset) == 0:
            return
        seen: set[int] = set()
        projection_method = projection_method_from_config(config.projection_method)
        with torch.no_grad():
            for dataset_idx, scene_idx in enumerate(dataset.indices):
                camera_model = int(dataset.sfm_scene.images[scene_idx].camera_metadata.camera_model)
                if camera_model in seen:
                    continue
                seen.add(camera_model)
                datum = dataset[dataset_idx]
                world_to_camera = datum["world_to_camera"].unsqueeze(0).to(device)
                projection = datum["projection"].unsqueeze(0).to(device)
                distortion_coeffs = datum["distortion_coeffs"].unsqueeze(0).to(device)
                world_to_camera = world_to_camera.contiguous()
                projection = projection.contiguous()
                distortion_coeffs = distortion_coeffs.contiguous()
                height, width = datum["image"].shape[:2]
                camera_model_enum = CameraModel(camera_model)
                distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model_enum, distortion_coeffs, model.device)
                if render_depth:
                    model.project_gaussians_for_images_and_depths(
                        world_to_camera_matrices=world_to_camera,
                        projection_matrices=projection,
                        image_width=width,
                        image_height=height,
                        near=config.near_plane,
                        far=config.far_plane,
                        camera_model=camera_model_enum,
                        projection_method=projection_method,
                        distortion_coeffs=distortion_coeffs_arg,
                        sh_degree_to_use=0,
                        min_radius_2d=config.min_radius_2d,
                        eps_2d=config.eps_2d,
                        antialias=config.antialias,
                    )
                else:
                    model.project_gaussians_for_images(
                        world_to_camera_matrices=world_to_camera,
                        projection_matrices=projection,
                        image_width=width,
                        image_height=height,
                        near=config.near_plane,
                        far=config.far_plane,
                        camera_model=camera_model_enum,
                        projection_method=projection_method,
                        distortion_coeffs=distortion_coeffs_arg,
                        sh_degree_to_use=0,
                        min_radius_2d=config.min_radius_2d,
                        eps_2d=config.eps_2d,
                        antialias=config.antialias,
                    )


class WorldSpaceRenderBackend:
    """
    Backend that renders directly from world-space Gaussians.

    This backend uses the world-space FVDB rendering APIs directly. It avoids the explicit
    projected-gaussian intermediate used by the image-space backend while preserving the same
    high-level interface expected by :class:`GaussianSplatReconstruction`.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Probe the scene cameras to ensure world-space rendering supports them.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        if config.batch_size > 1 and torch.unique(torch.from_numpy(dataset.camera_models)).numel() > 1:
            raise NotImplementedError("batch_size > 1 is not supported for scenes with multiple camera models")
        if len(dataset) == 0:
            return
        seen: set[int] = set()
        projection_method = projection_method_from_config(config.projection_method)
        with torch.no_grad():
            for dataset_idx, scene_idx in enumerate(dataset.indices):
                camera_model = int(dataset.sfm_scene.images[scene_idx].camera_metadata.camera_model)
                if camera_model in seen:
                    continue
                seen.add(camera_model)
                datum = dataset[dataset_idx]
                world_to_camera = datum["world_to_camera"].unsqueeze(0).to(device)
                projection = datum["projection"].unsqueeze(0).to(device)
                distortion_coeffs = datum["distortion_coeffs"].unsqueeze(0).to(device)
                world_to_camera = world_to_camera.contiguous()
                projection = projection.contiguous()
                distortion_coeffs = distortion_coeffs.contiguous()
                height, width = datum["image"].shape[:2]
                camera_model_enum = CameraModel(camera_model)
                distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model_enum, distortion_coeffs, model.device)
                render_function = (
                    model.render_images_and_depths_from_world
                    if _needs_depth_render(config)
                    else model.render_images_from_world
                )
                render_function(
                    world_to_camera_matrices=world_to_camera,
                    projection_matrices=projection,
                    image_width=width,
                    image_height=height,
                    near=config.near_plane,
                    far=config.far_plane,
                    camera_model=camera_model_enum,
                    projection_method=projection_method,
                    distortion_coeffs=distortion_coeffs_arg,
                    sh_degree_to_use=0,
                    tile_size=config.tile_size,
                    min_radius_2d=config.min_radius_2d,
                    eps_2d=config.eps_2d,
                    antialias=config.antialias,
                )

    def forward_train(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> TrainingView:
        """
        Render the full training view directly from world-space Gaussians; the returned view slices crops.

        The world-space renderer has no projection stage to share between crops, so the full image
        is rendered once and each crop is a slice of it, which keeps the crop-based training loop
        uniform across backends.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Full image width in pixels before cropping.
            image_height (int): Full image height in pixels before cropping.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            TrainingView: Slices any crop out of the rendered view.
        """
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device)
        render_function = (
            model.render_images_and_depths_from_world if _needs_depth_render(config) else model.render_images_from_world
        )
        rendered, alpha = render_function(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return _RenderedTrainingView(rendered=rendered, alpha=alpha, num_channels=model.num_channels)

    def forward_eval(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        camera_models: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """
        Render a full evaluation image directly from world-space Gaussians.

        Args:
            model (GaussianSplat3d): Gaussian splat model to render.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            world_to_camera_matrices (torch.Tensor): Batch of world-to-camera matrices.
            projection_matrices (torch.Tensor): Batch of camera intrinsics matrices.
            camera_models (torch.Tensor): Batch of encoded :class:`fvdb_reality_capture.CameraModel` values.
            distortion_coeffs (torch.Tensor): Batch of packed distortion coefficients.
            image_width (int): Output image width in pixels.
            image_height (int): Output image height in pixels.
            sh_degree_to_use (int): Maximum spherical harmonics degree to render.

        Returns:
            RenderOutputs: The rendered image and alpha image, plus depth when provided by the
            selected render path.
        """
        camera_model = _camera_model_from_batch(camera_models)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device)
        image, alpha = model.render_images_from_world(
            world_to_camera_matrices=world_to_camera_matrices,
            projection_matrices=projection_matrices,
            image_width=image_width,
            image_height=image_height,
            near=config.near_plane,
            far=config.far_plane,
            camera_model=camera_model,
            projection_method=projection_method_from_config(config.projection_method),
            distortion_coeffs=distortion_coeffs_arg,
            sh_degree_to_use=sh_degree_to_use,
            tile_size=config.tile_size,
            min_radius_2d=config.min_radius_2d,
            eps_2d=config.eps_2d,
            antialias=config.antialias,
        )
        return RenderOutputs(image=image, alpha=alpha)


def make_render_backend(name: RenderBackendName) -> RenderBackend:
    if name == "image_space":
        return ImageSpaceRenderBackend()
    if name == "world_space":
        return WorldSpaceRenderBackend()
    raise ValueError(f"Unsupported render_backend {name}")
