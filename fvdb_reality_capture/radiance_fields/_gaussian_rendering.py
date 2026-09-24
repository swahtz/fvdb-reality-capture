# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, Literal, Protocol

import numpy as np
import torch

from ..enums import CameraModel, ProjectionMethod
from ..functional import (
    Crop,
    apply_crop,
    rasterize_screen_space_gaussians,
    requires_distortion_coeffs,
    resolve_projection_method,
)

from .gaussian_splat_dataset import SfmDataset
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats

if TYPE_CHECKING:
    from .gaussian_splat_reconstruction import GaussianSplatReconstructionConfig

_logger = logging.getLogger(__name__)


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

    A backend does the work that does not depend on the crop once per view (projection and feature
    evaluation for the image-space backend, the full render for the world-space one) and returns one
    of these. Crops render from detached copies of that shared work, so the training loop calls
    ``loss.backward()`` after each crop, which frees that crop's rasterization and loss graphs and
    accumulates its gradient into the copies. :meth:`finish_backward` then runs the shared work's
    backward once with the accumulated gradients. The shared backward costs the same whatever the
    crop count, and the densification statistics it records see the whole image's gradient once, as
    with a single crop (up to the crop-seam effect on SSIM noted at ``crops_per_image``).
    """

    def render_crop(self, crop: Crop) -> RenderOutputs:
        """
        Render one crop of this view from the detached per-view tensors.

        Args:
            crop (tuple[int, int, int, int]): Crop rectangle as ``(origin_w, origin_h, width, height)``.

        Returns:
            RenderOutputs: The rendered crop, alpha image, and optional depth image.
        """
        ...

    def finish_backward(self) -> None:
        """Run the shared per-view backward once, with the gradients the crops accumulated."""
        ...


def _split_render_outputs(rendered: torch.Tensor, alpha: torch.Tensor, num_channels: int) -> RenderOutputs:
    image = rendered[..., :num_channels]
    depth = rendered[..., -1:] if rendered.shape[-1] == num_channels + 1 else None
    return RenderOutputs(image=image, alpha=alpha, depth=depth)


class _SharedWork:
    """Detached copies of a view's shared tensors, and the one backward that connects them to the model."""

    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self._tensors = tensors
        self.leaves = [tensor.detach().requires_grad_(tensor.requires_grad) for tensor in tensors]

    def backward(self) -> None:
        connected = [(tensor, leaf.grad) for tensor, leaf in zip(self._tensors, self.leaves) if leaf.grad is not None]
        if connected:
            torch.autograd.backward([tensor for tensor, _ in connected], [grad for _, grad in connected])
        # The gradients have been handed on, so they are dropped here rather than living as long as the leaves.
        for leaf in self.leaves:
            leaf.grad = None


class _ProjectedTrainingView:
    """Image-space view: projection and features are computed once, each crop rasterizes from copies of them."""

    def __init__(self, projected_gaussians: ProjectedGaussianSplats, tile_size: int, num_channels: int) -> None:
        self.projected_gaussians = projected_gaussians
        projected = projected_gaussians.projected_gaussians
        self._shared = _SharedWork(
            [projected.means2d, projected.conics, projected_gaussians.render_quantities, projected_gaussians.opacities]
        )
        means2d, conics, self._features, self._opacities = self._shared.leaves
        self._projected = replace(projected, means2d=means2d, conics=conics)
        self._tiles = projected_gaussians.tile_intersection(tile_size)
        self._num_channels = num_channels

    def render_crop(self, crop: Crop) -> RenderOutputs:
        full_image = tuple(crop) == (0, 0, self._tiles.image_width, self._tiles.image_height)
        rendered, alphas = rasterize_screen_space_gaussians(
            self._projected, self._features, self._opacities, self._tiles, crop=None if full_image else crop
        )
        return _split_render_outputs(rendered, alphas, self._num_channels)

    def finish_backward(self) -> None:
        self._shared.backward()


class _RenderedTrainingView:
    """World-space view: the full image is rendered once, each crop is a slice of a copy of it."""

    def __init__(self, rendered: torch.Tensor, alpha: torch.Tensor, num_channels: int) -> None:
        self._shared = _SharedWork([rendered, alpha])
        self._rendered, self._alpha = self._shared.leaves
        self._num_channels = num_channels

    def render_crop(self, crop: Crop) -> RenderOutputs:
        rendered, alpha = apply_crop(self._rendered, self._alpha, crop)
        return _split_render_outputs(rendered, alpha, self._num_channels)

    def finish_backward(self) -> None:
        self._shared.backward()


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


def _forward_only_explanation(forward_only: list[CameraModel], config: "GaussianSplatReconstructionConfig") -> str:
    names = ", ".join(camera_model.name for camera_model in forward_only)
    return (
        f"Camera models {names} resolve to the unscented projection under "
        f"projection_method={config.projection_method!r}, which has no backward pass, so the image-space render "
        "backend cannot train them."
    )


def _distinct_camera_batches(
    dataset: SfmDataset, device: torch.device
) -> Iterator[tuple[CameraModel, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]]:
    """One single-image batch per distinct camera model in ``dataset``, for probing a render path.

    Yields ``(camera_model, world_to_camera, projection, distortion_coeffs, width, height)``, with the
    distortion coefficients already ``None`` for camera models that take none.
    """
    seen: set[int] = set()
    for scene_idx in dataset.indices:
        image_meta = dataset.sfm_scene.images[scene_idx]
        camera_meta = image_meta.camera_metadata
        camera_model = int(camera_meta.camera_model)
        if camera_model in seen:
            continue
        seen.add(camera_model)
        # The camera metadata carries everything the probe needs, so no image is decoded.
        world_to_camera = torch.from_numpy(image_meta.world_to_camera_matrix).float().unsqueeze(0).to(device)
        projection = torch.from_numpy(camera_meta.projection_matrix).float().unsqueeze(0).to(device)
        coeffs = (
            camera_meta.distortion_coeffs if camera_meta.distortion_coeffs.size != 0 else np.zeros((12,), np.float32)
        )
        distortion_coeffs = torch.from_numpy(coeffs).float().unsqueeze(0).to(device)
        camera_model_enum = CameraModel(camera_model)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model_enum, distortion_coeffs, device)
        yield (
            camera_model_enum,
            world_to_camera.contiguous(),
            projection.contiguous(),
            distortion_coeffs_arg,
            camera_meta.width,
            camera_meta.height,
        )


def _probe_projection(
    model: GaussianSplat3d,
    config: "GaussianSplatReconstructionConfig",
    camera_model: CameraModel,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    distortion_coeffs: torch.Tensor | None,
    width: int,
    height: int,
) -> None:
    projection_function = (
        model.project_gaussians_for_images_and_depths
        if _needs_depth_render(config)
        else model.project_gaussians_for_images
    )
    projection_function(
        world_to_camera_matrices=world_to_camera,
        projection_matrices=projection,
        image_width=width,
        image_height=height,
        near=config.near_plane,
        far=config.far_plane,
        camera_model=camera_model,
        projection_method=projection_method_from_config(config.projection_method),
        distortion_coeffs=distortion_coeffs,
        sh_degree_to_use=0,
        min_radius_2d=config.min_radius_2d,
        eps_2d=config.eps_2d,
        antialias=config.antialias,
    )


def _probe_world_space_render(
    model: GaussianSplat3d,
    config: "GaussianSplatReconstructionConfig",
    camera_model: CameraModel,
    world_to_camera: torch.Tensor,
    projection: torch.Tensor,
    distortion_coeffs: torch.Tensor | None,
    width: int,
    height: int,
) -> None:
    render_function = (
        model.render_images_and_depths_from_world if _needs_depth_render(config) else model.render_images_from_world
    )
    render_function(
        world_to_camera_matrices=world_to_camera,
        projection_matrices=projection,
        image_width=width,
        image_height=height,
        near=config.near_plane,
        far=config.far_plane,
        camera_model=camera_model,
        projection_method=projection_method_from_config(config.projection_method),
        distortion_coeffs=distortion_coeffs,
        sh_degree_to_use=0,
        tile_size=config.tile_size,
        min_radius_2d=config.min_radius_2d,
        eps_2d=config.eps_2d,
        antialias=config.antialias,
    )


def _check_camera_batching(dataset: SfmDataset, config: "GaussianSplatReconstructionConfig") -> None:
    """Minibatches hold one camera model, so a multi-model scene can only be trained one image at a time."""
    if config.batch_size > 1 and len(set(dataset.camera_models.tolist())) > 1:
        raise NotImplementedError("batch_size > 1 is not supported for scenes with multiple camera models")


def _camera_model_from_batch(camera_models: torch.Tensor) -> CameraModel:
    unique_camera_models = torch.unique(camera_models.to(dtype=torch.int32))
    if unique_camera_models.numel() != 1:
        raise NotImplementedError("Rendering a minibatch with multiple camera models is not supported")
    return CameraModel(int(unique_camera_models.item()))


def _distortion_coeffs_for_batch(
    camera_model: CameraModel, distortion_coeffs: torch.Tensor, device: torch.device
) -> torch.Tensor | None:
    if not requires_distortion_coeffs(camera_model):
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

    It cannot train cameras that need the unscented projection, which has no backward pass;
    :class:`RoutedRenderBackend` sends those batches to :class:`WorldSpaceRenderBackend` instead.
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

        Cameras that resolve to the unscented projection are rejected: rasterizing from that projection
        would train features and opacities while the Gaussian geometry received no gradient at all.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        _check_camera_batching(dataset, config)
        forward_only = _forward_only_camera_models(dataset, config)
        if forward_only:
            raise ValueError(
                _forward_only_explanation(forward_only, config)
                + ' Use render_backend="world_space", or the routed backend that "image_space" selects, which '
                "differentiates through the 3D parameters directly for those cameras."
            )
        with torch.no_grad():
            for camera_model, world_to_camera, projection, distortion_coeffs, width, height in _distinct_camera_batches(
                dataset, device
            ):
                _probe_projection(
                    model, config, camera_model, world_to_camera, projection, distortion_coeffs, width, height
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
        Project the Gaussians into the target cameras once; the returned view rasterizes each crop.

        If depth regularization is enabled, the projection also carries the depth channel. The view
        keeps the tile intersection and the opacities of that projection, so rendering several crops
        repeats only the rasterization.

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
        return self._train_view(
            model,
            config,
            _camera_model_from_batch(camera_models),
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )

    def _train_view(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> TrainingView:
        """The body of :meth:`forward_train` for an already resolved camera model."""
        projection_method = projection_method_from_config(config.projection_method)
        distortion_coeffs_arg = _distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device)
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
        return _ProjectedTrainingView(projected_gaussians, config.tile_size, model.num_channels)

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
        return self._eval_render(
            model,
            config,
            _camera_model_from_batch(camera_models),
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )

    def _eval_render(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """The body of :meth:`forward_eval` for an already resolved camera model."""
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
        _check_camera_batching(dataset, config)
        with torch.no_grad():
            for camera_model, world_to_camera, projection, distortion_coeffs, width, height in _distinct_camera_batches(
                dataset, device
            ):
                _probe_world_space_render(
                    model, config, camera_model, world_to_camera, projection, distortion_coeffs, width, height
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
        return self._train_view(
            model,
            config,
            _camera_model_from_batch(camera_models),
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )

    def _train_view(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> TrainingView:
        """The body of :meth:`forward_train` for an already resolved camera model."""
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
        return _RenderedTrainingView(rendered, alpha, model.num_channels)

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
        return self._eval_render(
            model,
            config,
            _camera_model_from_batch(camera_models),
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )

    def _eval_render(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        sh_degree_to_use: int,
    ) -> RenderOutputs:
        """The body of :meth:`forward_eval` for an already resolved camera model."""
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


class RoutedRenderBackend:
    """
    Image space where it can train, world space where it must, chosen per camera batch.

    Batches whose camera has an analytic projection go to :class:`ImageSpaceRenderBackend`. Batches
    whose camera resolves to the unscented projection, which has no backward pass, go to
    :class:`WorldSpaceRenderBackend`, which differentiates through the 3D parameters directly. The
    same choice is made for training and evaluation, so the renderer being measured is the one being
    optimized. Because it is per batch, a scene that mixes pinhole and distortion cameras keeps its
    pinhole views in image space, and they keep feeding the densification statistics, which only the
    analytic projection's backward produces.
    """

    def __init__(self) -> None:
        self._image_space = ImageSpaceRenderBackend()
        self._world_space = WorldSpaceRenderBackend()

    def _backend_for(
        self, camera_model: CameraModel, config: "GaussianSplatReconstructionConfig"
    ) -> ImageSpaceRenderBackend | WorldSpaceRenderBackend:
        projection_method = projection_method_from_config(config.projection_method)
        if resolve_projection_method(camera_model, projection_method) == ProjectionMethod.UNSCENTED:
            return self._world_space
        return self._image_space

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Probe every camera model in the scene through the path its batches will take.

        Camera models that will render in world space are logged once, since views from them add
        nothing to the densification statistics.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        _check_camera_batching(dataset, config)
        forward_only = _forward_only_camera_models(dataset, config)
        if forward_only:
            _logger.warning(
                _forward_only_explanation(forward_only, config)
                + " Views from those cameras are rendered in world space, which differentiates through the 3D "
                "parameters directly; views from other cameras stay in image space. Gaussian densification "
                "statistics come only from image-space views. Undistort the images to train every view in image space."
            )
        with torch.no_grad():
            for camera_model, world_to_camera, projection, distortion_coeffs, width, height in _distinct_camera_batches(
                dataset, device
            ):
                probe = (
                    _probe_projection
                    if self._backend_for(camera_model, config) is self._image_space
                    else _probe_world_space_render
                )
                probe(model, config, camera_model, world_to_camera, projection, distortion_coeffs, width, height)

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
        """Per-image training work with the backend chosen for this batch's camera; see :class:`RenderBackend`."""
        camera_model = _camera_model_from_batch(camera_models)
        return self._backend_for(camera_model, config)._train_view(
            model,
            config,
            camera_model,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )

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
        """Render an evaluation image with the backend chosen for this batch's camera; see :class:`RenderBackend`."""
        camera_model = _camera_model_from_batch(camera_models)
        return self._backend_for(camera_model, config)._eval_render(
            model,
            config,
            camera_model,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            image_width,
            image_height,
            sh_degree_to_use,
        )


def make_render_backend(name: RenderBackendName) -> RenderBackend:
    """Return the backend a :class:`GaussianSplatReconstructionConfig` ``render_backend`` value names.

    ``"image_space"`` is the routed backend: image space for cameras it can train, world space for the rest.
    """
    if name == "image_space":
        return RoutedRenderBackend()
    if name == "world_space":
        return WorldSpaceRenderBackend()
    raise ValueError(f"Unsupported render_backend {name}")
