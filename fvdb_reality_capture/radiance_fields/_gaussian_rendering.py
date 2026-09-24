# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Protocol

import torch

from ..enums import CameraModel, ProjectionMethod
from ..functional import (
    Crop,
    rasterize_screen_space_gaussians,
    rasterize_world_space_gaussians,
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


RenderBackendName = Literal["auto", "image_space", "world_space"]


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


class _WorldSpaceTrainingView:
    """World-space view: projection, features, opacities and tiles are computed once, each crop rasterizes.

    The world-space rasterizer differentiates through the 3D parameters directly, so the crops reach
    ``means``, ``quats`` and ``log_scales`` on their own. Features and opacities are the shared inputs
    (the features carry the spherical-harmonics graph, and through it the pose-adjustment graph), so the
    crops rasterize from detached copies of them and :meth:`finish_backward` propagates their accumulated
    gradients once. The rasterizer returns no gradient for the camera matrices, so they are passed
    detached; passing them with their graph would make each crop's backward walk, and free, the
    pose-adjustment graph the features still need.
    """

    def __init__(
        self,
        model: GaussianSplat3d,
        projected_gaussians: ProjectedGaussianSplats,
        world_to_camera_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor | None,
        tile_size: int,
        num_channels: int,
    ) -> None:
        self._model = model
        self._projected = projected_gaussians.projected_gaussians
        self._shared = _SharedWork([projected_gaussians.render_quantities, projected_gaussians.opacities])
        self._features, self._opacities = self._shared.leaves
        self._world_to_camera = world_to_camera_matrices.detach()
        self._projection = projection_matrices.detach()
        self._distortion_coeffs = distortion_coeffs
        self._tiles = projected_gaussians.tile_intersection(tile_size)
        self._num_channels = num_channels

    def render_crop(self, crop: Crop) -> RenderOutputs:
        full_image = tuple(crop) == (0, 0, self._tiles.image_width, self._tiles.image_height)
        rendered, alphas = rasterize_world_space_gaussians(
            self._model.means,
            self._model.quats,
            self._model.log_scales,
            self._projected,
            self._features,
            self._opacities,
            self._world_to_camera,
            self._projection,
            self._tiles,
            distortion_coeffs=self._distortion_coeffs,
            crop=None if full_image else crop,
        )
        return _split_render_outputs(rendered, alphas, self._num_channels)

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


def _pose_optimization_note(camera_models: list[CameraModel], config: "GaussianSplatReconstructionConfig") -> str:
    """Explain what pose optimization loses for cameras that render in world space, or return ``""``."""
    if not config.optimize_camera_poses or not camera_models:
        return ""
    names = ", ".join(camera_model.name for camera_model in camera_models)
    return (
        f" Camera pose optimization is enabled, but the world-space rasterizer returns no gradient for the camera "
        f"matrices, so the poses of {names} views are trained only through the view direction of the spherical "
        "harmonics and the depth term, not through the rendered geometry."
    )


def _distinct_camera_batches(
    dataset: SfmDataset, device: torch.device
) -> Iterator[tuple[CameraModel, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]]:
    """One single-image batch per distinct camera model in ``dataset``, for probing a render path.

    Yields ``(camera_model, world_to_camera, projection, distortion_coeffs, width, height)`` with the packed
    ``[1, 12]`` distortion coefficients and the image size the dataset delivers.
    """
    seen: set[int] = set()
    camera_models = dataset.camera_models
    projections = dataset.projection_matrices
    distortion = dataset.distortion_coeffs
    sizes = dataset.image_sizes
    for dataset_idx, scene_idx in enumerate(dataset.indices):
        camera_model = int(camera_models[dataset_idx])
        if camera_model in seen:
            continue
        seen.add(camera_model)
        # The dataset's own per-image arrays carry everything the probe needs, so no image is decoded.
        world_to_camera = torch.from_numpy(dataset.sfm_scene.images[scene_idx].world_to_camera_matrix).float()
        height, width = (int(v) for v in sizes[dataset_idx])
        width, height = _delivered_image_size(dataset, width, height)
        yield (
            CameraModel(camera_model),
            world_to_camera.unsqueeze(0).to(device).contiguous(),
            torch.from_numpy(projections[dataset_idx]).float().unsqueeze(0).to(device).contiguous(),
            torch.from_numpy(distortion[dataset_idx]).float().unsqueeze(0).to(device),
            width,
            height,
        )


def _delivered_image_size(dataset: SfmDataset, width: int, height: int) -> tuple[int, int]:
    """The size of the images ``dataset`` actually delivers for a camera of ``width`` by ``height``."""
    patch_size = getattr(dataset, "patch_size", None)
    if patch_size is None:
        return width, height
    return min(width, patch_size), min(height, patch_size)


def _render_arguments(
    model: GaussianSplat3d,
    config: "GaussianSplatReconstructionConfig",
    camera_model: CameraModel,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    image_width: int,
    image_height: int,
    sh_degree_to_use: int,
) -> dict[str, Any]:
    """The keyword arguments every :class:`GaussianSplat3d` projection and render call takes for one camera batch."""
    return dict(
        world_to_camera_matrices=world_to_camera_matrices,
        projection_matrices=projection_matrices,
        image_width=image_width,
        image_height=image_height,
        near=config.near_plane,
        far=config.far_plane,
        camera_model=camera_model,
        projection_method=projection_method_from_config(config.projection_method),
        distortion_coeffs=_distortion_coeffs_for_batch(camera_model, distortion_coeffs, model.device),
        sh_degree_to_use=sh_degree_to_use,
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


def _project_for_training(
    model: GaussianSplat3d,
    config: "GaussianSplatReconstructionConfig",
    arguments: dict[str, Any],
    accumulate_statistics: bool = True,
) -> ProjectedGaussianSplats:
    """Project one camera batch for a training step, with the depth channel when a depth term is on.

    Radii are recorded in the forward for every training projection, since the kernel records them only
    in a backward that reaches the projected means. ``accumulate_statistics`` wires the gradient
    accumulators in; the world-space path turns it off so its views do not count as samples (with
    antialiasing on, the opacity gradient would otherwise reach the analytic backward and bump the step counts).
    """
    projection_function = (
        model.project_gaussians_for_images_and_depths
        if _needs_depth_render(config)
        else model.project_gaussians_for_images
    )
    return projection_function(**arguments, accumulate_statistics=accumulate_statistics, record_radii=True)


class _ModelRenderBackend:
    """
    Shared forward path of the backends that render a :class:`GaussianSplat3d`.

    Validation checks the camera batching, lets the subclass judge the scene's camera models, then
    probes one image per camera model through :meth:`_probe`. The forward passes resolve the batch's
    camera model once and hand the shared render arguments to :meth:`_train_view` and
    :meth:`_eval_render`, which are the only methods a backend has to provide.
    """

    def validate_scene_cameras(
        self,
        model: GaussianSplat3d,
        dataset: SfmDataset,
        config: "GaussianSplatReconstructionConfig",
        device: torch.device,
    ) -> None:
        """
        Check the scene's cameras and probe one image per camera model through this backend's path.

        Args:
            model (GaussianSplat3d): Gaussian splat model used for the probe render.
            dataset (SfmDataset): Dataset whose cameras should be validated.
            config (GaussianSplatReconstructionConfig): Reconstruction config controlling render behavior.
            device (torch.device): Device on which validation probes should run.
        """
        _check_camera_batching(dataset, config)
        self._validate_camera_models(dataset, config)
        with torch.no_grad():
            for camera_model, world_to_camera, projection, distortion_coeffs, width, height in _distinct_camera_batches(
                dataset, device
            ):
                arguments = _render_arguments(
                    model, config, camera_model, world_to_camera, projection, distortion_coeffs, width, height, 0
                )
                self._probe(model, config, camera_model, arguments)

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
        """Do the per-image training work for this batch; see :class:`RenderBackend`."""
        camera_model = _camera_model_from_batch(camera_models)
        arguments = _render_arguments(
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
        return self._train_view(model, config, camera_model, arguments)

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
        """Render a full evaluation image for this batch; see :class:`RenderBackend`."""
        camera_model = _camera_model_from_batch(camera_models)
        arguments = _render_arguments(
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
        return self._eval_render(model, config, camera_model, arguments)

    def _validate_camera_models(self, dataset: SfmDataset, config: "GaussianSplatReconstructionConfig") -> None:
        """Raise or warn about camera models this backend cannot fully train."""

    def _probe(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        arguments: dict[str, Any],
    ) -> None:
        """Run one image of ``camera_model`` through this backend's path under ``torch.no_grad``."""
        raise NotImplementedError

    def _train_view(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        arguments: dict[str, Any],
    ) -> TrainingView:
        """The training view for one batch, given the resolved camera model and shared render arguments."""
        raise NotImplementedError

    def _eval_render(
        self,
        model: GaussianSplat3d,
        config: "GaussianSplatReconstructionConfig",
        camera_model: CameraModel,
        arguments: dict[str, Any],
    ) -> RenderOutputs:
        """The evaluation render for one batch, given the resolved camera model and shared render arguments."""
        raise NotImplementedError


class ImageSpaceRenderBackend(_ModelRenderBackend):
    """
    Backend that projects Gaussians in image space before rasterization.

    This backend first projects Gaussians into each camera view using the image-space FVDB APIs
    and then rasterizes those projected Gaussians. It is the natural fit for the classic 3DGS
    rendering path and for renderers that need projected-gaussian intermediates during training.

    It cannot train cameras that need the unscented projection, which has no backward pass;
    ``render_backend="auto"`` chooses :class:`WorldSpaceRenderBackend` for a scene with such cameras.
    """

    def _validate_camera_models(self, dataset: SfmDataset, config: "GaussianSplatReconstructionConfig") -> None:
        # Rasterizing from the unscented projection would train features and opacities while the geometry
        # received no gradient at all, so those cameras are rejected rather than half-trained.
        forward_only = _forward_only_camera_models(dataset, config)
        if forward_only:
            raise ValueError(
                _forward_only_explanation(forward_only, config)
                + ' Use render_backend="world_space" (what "auto" chooses for this scene), or undistort the images.'
            )

    def _probe(self, model, config, camera_model, arguments) -> None:
        _project_for_training(model, config, arguments)

    def _train_view(self, model, config, camera_model, arguments) -> TrainingView:
        # Project once; the view keeps the tile intersection and opacities, so each crop only rasterizes.
        return _ProjectedTrainingView(
            _project_for_training(model, config, arguments), config.tile_size, model.num_channels
        )

    def _eval_render(self, model, config, camera_model, arguments) -> RenderOutputs:
        image, alpha = model.render_images(**arguments, tile_size=config.tile_size)
        return RenderOutputs(image=image, alpha=alpha)


class WorldSpaceRenderBackend(_ModelRenderBackend):
    """
    Backend that renders directly from world-space Gaussians.

    This backend uses the world-space FVDB rendering APIs directly. It avoids the explicit
    projected-gaussian intermediate used by the image-space backend while preserving the same
    high-level interface expected by :class:`GaussianSplatReconstruction`. For training, the projection,
    features, opacities and tiles of a view are computed once and every crop rasterizes from them.
    """

    def _validate_camera_models(self, dataset: SfmDataset, config: "GaussianSplatReconstructionConfig") -> None:
        note = _pose_optimization_note([CameraModel(m) for m in sorted(set(dataset.camera_models.tolist()))], config)
        if note:
            _logger.warning(note.strip())

    @staticmethod
    def _render_function(model: GaussianSplat3d, config: "GaussianSplatReconstructionConfig"):
        return (
            model.render_images_and_depths_from_world if _needs_depth_render(config) else model.render_images_from_world
        )

    def _probe(self, model, config, camera_model, arguments) -> None:
        self._render_function(model, config)(**arguments, tile_size=config.tile_size)

    def _train_view(self, model, config, camera_model, arguments) -> TrainingView:
        # The shared stages run once here; each crop only rasterizes, as in image space.
        return _WorldSpaceTrainingView(
            model,
            _project_for_training(model, config, arguments, accumulate_statistics=False),
            arguments["world_to_camera_matrices"],
            arguments["projection_matrices"],
            arguments["distortion_coeffs"],
            config.tile_size,
            model.num_channels,
        )

    def _eval_render(self, model, config, camera_model, arguments) -> RenderOutputs:
        image, alpha = model.render_images_from_world(**arguments, tile_size=config.tile_size)
        return RenderOutputs(image=image, alpha=alpha)


def make_render_backend(name: RenderBackendName) -> RenderBackend:
    """Return the pure backend a ``render_backend`` value names; ``"auto"`` needs the scene, see :func:`resolve_render_backend`."""
    if name == "image_space":
        return ImageSpaceRenderBackend()
    if name == "world_space":
        return WorldSpaceRenderBackend()
    if name == "auto":
        raise ValueError('render_backend="auto" is resolved against the scene by resolve_render_backend')
    raise ValueError(f"Unsupported render_backend {name}")


def resolve_render_backend(config: "GaussianSplatReconstructionConfig", dataset: SfmDataset) -> RenderBackend:
    """Choose the one backend a reconstruction run uses, from its config and the scene's cameras.

    A run renders every batch with the same backend. ``"auto"`` picks image space, which feeds the
    densification statistics, unless some camera in the scene resolves to the unscented projection,
    which has no backward pass; then the whole run renders in world space so that geometry still
    receives a gradient, and the choice and its consequences are logged once. The explicit names are
    returned as they are, and validation decides whether the scene suits them.

    Args:
        config (GaussianSplatReconstructionConfig): Reconstruction config; reads ``render_backend``,
            ``projection_method`` and ``optimize_camera_poses``.
        dataset (SfmDataset): The training dataset whose camera models decide ``"auto"``.

    Returns:
        RenderBackend: The backend for the run.
    """
    if config.render_backend != "auto":
        return make_render_backend(config.render_backend)
    forward_only = _forward_only_camera_models(dataset, config)
    if not forward_only:
        return ImageSpaceRenderBackend()
    _logger.warning(
        _forward_only_explanation(forward_only, config)
        + " The run renders every view in world space, which differentiates through the 3D parameters "
        "directly; the 2D mean-gradient densification statistics, which only image-space views produce, are "
        "unavailable, so gradient-driven duplication and splitting are skipped while radius-based splitting and "
        "deletion proceed."
        + _pose_optimization_note(forward_only, config)
        + " Undistort the images to train in image space."
    )
    return WorldSpaceRenderBackend()
