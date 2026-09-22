# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 4 of the composable pipeline: alpha-blend features into images or pixel sets."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as nnf
from fvdb import JaggedTensor

from ..enums import RollingShutterType
from ._autograd import (
    _RasterizeScreenSpaceGaussiansFn,
    _RasterizeScreenSpaceGaussiansSparseFn,
    _RasterizeWorldSpaceGaussiansFn,
)
from ._opacity import compute_gaussian_opacities
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection

Crop = tuple[int, int, int, int]
"""A crop as ``(origin_w, origin_h, width, height)`` in pixels."""


def _validate_crop(crop: Crop, image_width: int, image_height: int) -> Crop:
    """Clamp a crop to the image and reject crops that are empty or outside it."""
    origin_w, origin_h, width, height = crop
    if origin_w < 0 or origin_h < 0:
        raise ValueError(f"Crop origin must be non-negative, got ({origin_w}, {origin_h})")
    if width <= 0 or height <= 0:
        raise ValueError(f"Crop size must be positive, got ({width}, {height})")
    width = min(width, image_width - origin_w)
    height = min(height, image_height - origin_h)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Crop with origin ({origin_w}, {origin_h}) and size ({crop[2]}, {crop[3]}) lies outside the "
            f"{image_width}x{image_height} image"
        )
    return origin_w, origin_h, width, height


def _crop_window_mask(
    crop: Crop | None, masks: torch.Tensor | None, num_cameras: int, image_width: int, image_height: int, device
) -> tuple[Crop | None, torch.Tensor | None]:
    """Fold a crop into the per-pixel mask so tiles outside the crop are skipped by the rasterizer."""
    if crop is None:
        return None, masks
    origin_w, origin_h, width, height = _validate_crop(crop, image_width, image_height)
    window = torch.zeros(num_cameras, image_height, image_width, dtype=torch.bool, device=device)
    window[:, origin_h : origin_h + height, origin_w : origin_w + width] = True
    return (origin_w, origin_h, width, height), window if masks is None else masks & window


def _apply_crop(images: torch.Tensor, alphas: torch.Tensor, crop: Crop | None) -> tuple[torch.Tensor, torch.Tensor]:
    if crop is None:
        return images, alphas
    origin_w, origin_h, width, height = crop
    return (
        images[:, origin_h : origin_h + height, origin_w : origin_w + width],
        alphas[:, origin_h : origin_h + height, origin_w : origin_w + width],
    )


def pixel_mask_to_tile_mask(pixel_mask: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Mark a tile as rendered if any of its pixels is.

    Args:
        pixel_mask (torch.Tensor): Boolean per-pixel mask, ``[C, H, W]``.
        tile_size (int): Tile side length in pixels.

    Returns:
        tile_mask (torch.Tensor): Boolean per-tile mask, ``[C, ceil(H / tile_size), ceil(W / tile_size)]``.
    """
    pooled = nnf.max_pool2d(pixel_mask.unsqueeze(1).float(), kernel_size=tile_size, stride=tile_size, ceil_mode=True)
    return pooled.bool().squeeze(1)


def _apply_pixel_mask(
    images: torch.Tensor, alphas: torch.Tensor, pixel_mask: torch.Tensor, backgrounds: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill masked-out pixels with the background and zero alpha; the tile mask only skips whole tiles."""
    keep = pixel_mask.unsqueeze(-1).to(images.dtype)
    if backgrounds is not None:
        background = backgrounds[:, None, None, :]
    else:
        background = torch.zeros(1, 1, 1, images.shape[-1], device=images.device, dtype=images.dtype)
    return images * keep + background * (1.0 - keep), alphas * keep


def rasterize_screen_space_gaussians(
    projected: ProjectedGaussians,
    features: torch.Tensor,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    crop: Crop | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-blend projected Gaussians into dense images.

    Differentiable with respect to the projection, ``features`` and ``logit_opacities``. A ``crop``
    selects a window of the images: tiles outside it are skipped and the result is exactly the
    corresponding region of the uncropped render.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        masks (torch.Tensor | None): Boolean per-pixel render mask, ``[C, H, W]``. Masked-out pixels
            receive the background with zero alpha and no gradient.
        crop (tuple[int, int, int, int] | None): ``(origin_w, origin_h, width, height)`` window to keep,
            clamped to the image.

    Returns:
        images (torch.Tensor): Blended features, ``[C, H, W, D]`` (or the crop size).
        alphas (torch.Tensor): Accumulated alpha in ``[0, 1)``, ``[C, H, W, 1]`` (or the crop size).
    """
    opacities = compute_gaussian_opacities(logit_opacities, projected)
    crop, masks = _crop_window_mask(
        crop, masks, projected.num_cameras, tiles.image_width, tiles.image_height, opacities.device
    )
    tile_masks = pixel_mask_to_tile_mask(masks, tiles.tile_size) if masks is not None else None
    images, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeScreenSpaceGaussiansFn.apply(
            projected.means2d,
            projected.conics,
            features,
            opacities,
            tiles.image_width,
            tiles.image_height,
            0,
            0,
            tiles.tile_size,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            False,
            backgrounds,
            tile_masks,
        ),
    )
    if masks is not None:
        images, alphas = _apply_pixel_mask(images, alphas, masks, backgrounds)
    return _apply_crop(images, alphas, crop)


def rasterize_world_space_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    projected: ProjectedGaussians,
    features: torch.Tensor,
    logit_opacities: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projection_matrices: torch.Tensor,
    tiles: GaussianTileIntersection,
    distortion_coeffs: torch.Tensor | None = None,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
    crop: Crop | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-blend 3D Gaussians into dense images by evaluating them along per-pixel rays.

    Differentiable with respect to the 3D parameters, ``features`` and ``logit_opacities``, which makes
    it the training path for the unscented projection. The projection supplies the tile
    intersections and the camera model; the 3D parameters are evaluated directly.

    Args:
        means (torch.Tensor): Gaussian centers in world space, ``[N, 3]``.
        quats (torch.Tensor): Gaussian rotations as quaternions, ``[N, 4]``.
        log_scales (torch.Tensor): Natural-log scale factors, ``[N, 3]``.
        projected (ProjectedGaussians): Output of :func:`project_gaussians` for these Gaussians and cameras.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        world_to_camera_matrices (torch.Tensor): World-to-camera transforms, ``[C, 4, 4]``.
        projection_matrices (torch.Tensor): Camera intrinsics, ``[C, 3, 3]``.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.
        distortion_coeffs (torch.Tensor | None): Packed OpenCV distortion coefficients, ``[C, 12]``.
            Zeros are used if ``None``.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        masks (torch.Tensor | None): Boolean per-pixel render mask, ``[C, H, W]``.
        crop (tuple[int, int, int, int] | None): ``(origin_w, origin_h, width, height)`` window to keep.

    Returns:
        images (torch.Tensor): Blended features, ``[C, H, W, D]`` (or the crop size).
        alphas (torch.Tensor): Accumulated alpha in ``[0, 1)``, ``[C, H, W, 1]`` (or the crop size).
    """
    opacities = compute_gaussian_opacities(logit_opacities, projected)
    if distortion_coeffs is None:
        distortion_coeffs = torch.zeros(
            projected.num_cameras, 12, device=world_to_camera_matrices.device, dtype=world_to_camera_matrices.dtype
        )
    crop, masks = _crop_window_mask(
        crop, masks, projected.num_cameras, tiles.image_width, tiles.image_height, opacities.device
    )
    tile_masks = pixel_mask_to_tile_mask(masks, tiles.tile_size) if masks is not None else None
    images, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeWorldSpaceGaussiansFn.apply(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_camera_matrices,
            world_to_camera_matrices,
            projection_matrices,
            distortion_coeffs,
            int(RollingShutterType.NONE),
            int(projected.camera_model),
            tiles.image_width,
            tiles.image_height,
            0,
            0,
            tiles.tile_size,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            backgrounds,
            tile_masks,
        ),
    )
    if masks is not None:
        images, alphas = _apply_pixel_mask(images, alphas, masks, backgrounds)
    return _apply_crop(images, alphas, crop)


def rasterize_screen_space_gaussians_sparse(
    projected: ProjectedGaussians,
    features: torch.Tensor,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
    backgrounds: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Alpha-blend projected Gaussians at the requested pixels only.

    Differentiable with respect to the projection, ``features`` and ``logit_opacities``. Results are
    returned in the order of ``sparse_tiles.pixels_to_render``, duplicates included.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        features (torch.Tensor): Per-camera, per-Gaussian features, ``[C, N, D]``.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        sparse_tiles (SparseGaussianTileIntersection): Output of :func:`intersect_gaussian_tiles_sparse`.
        backgrounds (torch.Tensor | None): Per-camera background features, ``[C, D]``. Black if ``None``.
        masks (torch.Tensor | None): Boolean per-tile render mask, ``[C, num_tiles_h, num_tiles_w]``.

    Returns:
        features (JaggedTensor): Blended features per requested pixel, one ``[P_c, D]`` list per camera.
        alphas (JaggedTensor): Accumulated alpha per requested pixel, one ``[P_c, 1]`` list per camera.
    """
    opacities = compute_gaussian_opacities(logit_opacities, projected)
    rendered, alphas = cast(
        tuple[torch.Tensor, torch.Tensor],
        _RasterizeScreenSpaceGaussiansSparseFn.apply(
            projected.means2d,
            projected.conics,
            features,
            opacities,
            sparse_tiles.unique_pixels,
            sparse_tiles.image_width,
            sparse_tiles.image_height,
            0,
            0,
            sparse_tiles.tile_size,
            sparse_tiles.tile_offsets,
            sparse_tiles.tile_gaussian_ids,
            sparse_tiles.active_tiles,
            sparse_tiles.tile_pixel_mask,
            sparse_tiles.tile_pixel_cumsum,
            sparse_tiles.pixel_map,
            False,
            backgrounds,
            masks,
        ),
    )
    requested = sparse_tiles.pixels_to_render
    return (
        requested.jagged_like(sparse_tiles.expand_to_requested(rendered)),
        requested.jagged_like(sparse_tiles.expand_to_requested(alphas)),
    )
