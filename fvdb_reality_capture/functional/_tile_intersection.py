# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 3 of the composable pipeline: bin projected Gaussians into image tiles."""

from __future__ import annotations

import math

import torch
from fvdb import JaggedTensor
from fvdb import functional as F

from ._opacity import compute_gaussian_opacities
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection


def _tile_grid(image_width: int, image_height: int, tile_size: int) -> tuple[int, int]:
    return math.ceil(image_height / tile_size), math.ceil(image_width / tile_size)


def _culling_inputs(
    projected: ProjectedGaussians, logit_opacities: torch.Tensor | None, opacities: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Conics and opacities for the tighter iso-contour tile test, or ``None`` for the bounding-box test."""
    if opacities is not None:
        return projected.conics, opacities.detach()
    if logit_opacities is None:
        return None, None
    with torch.no_grad():
        return projected.conics, compute_gaussian_opacities(logit_opacities, projected)


def intersect_gaussian_tiles(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor | None = None,
    tile_size: int = 16,
    *,
    opacities: torch.Tensor | None = None,
) -> GaussianTileIntersection:
    """Bin projected Gaussians into the tiles of the full image, sorted by camera, tile and depth.

    Not differentiable. Passing ``logit_opacities`` enables a tighter per-tile culling test based on
    each Gaussian's iso-contour at the opacity threshold instead of its bounding box, which reduces
    the work of every later stage.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor | None): Logit opacities, ``[N]``, for tighter culling.
        tile_size (int): Tile side length in pixels.
        opacities (torch.Tensor | None): Precomputed ``[C, N]`` opacities from
            :func:`compute_gaussian_opacities`, to avoid recomputing them per stage. Derived from
            ``logit_opacities`` when ``None``.

    Returns:
        tiles (GaussianTileIntersection): The tile intersections.
    """
    num_tiles_h, num_tiles_w = _tile_grid(projected.image_width, projected.image_height, tile_size)
    conics, opacities = _culling_inputs(projected, logit_opacities, opacities)
    tile_offsets, tile_gaussian_ids = F.intersect_gaussian_tiles(
        projected.means2d,
        projected.radii,
        projected.depths,
        projected.num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        conics=conics,
        opacities=opacities,
    )
    return GaussianTileIntersection(
        tile_offsets=tile_offsets,
        tile_gaussian_ids=tile_gaussian_ids,
        tile_size=tile_size,
        image_width=projected.image_width,
        image_height=projected.image_height,
    )


def as_pixel_jagged(pixels_to_render: JaggedTensor | torch.Tensor) -> JaggedTensor:
    """Normalize a pixel selection to a JaggedTensor with one ``[P_c, 2]`` list of ``(row, col)`` per camera.

    Args:
        pixels_to_render (JaggedTensor | torch.Tensor): A JaggedTensor, returned as is, or a ``[C, P, 2]``
            tensor with the same number of pixels for every camera.

    Returns:
        pixels (JaggedTensor): The selection as a JaggedTensor.
    """
    if isinstance(pixels_to_render, JaggedTensor):
        return pixels_to_render
    if isinstance(pixels_to_render, torch.Tensor):
        if pixels_to_render.dim() != 3 or pixels_to_render.shape[-1] != 2:
            raise ValueError(f"pixels_to_render tensor must have shape [C, P, 2], got {tuple(pixels_to_render.shape)}")
        return JaggedTensor(list(pixels_to_render.unbind(0)))
    raise TypeError(
        f"pixels_to_render must be a fvdb.JaggedTensor or torch.Tensor, got {type(pixels_to_render).__name__}"
    )


def deduplicate_pixels(
    pixels_to_render: JaggedTensor, image_width: int, image_height: int
) -> tuple[JaggedTensor, torch.Tensor, bool]:
    """Remove pixels that appear more than once within a camera.

    Args:
        pixels_to_render (JaggedTensor): ``(row, col)`` integer pixels, one list per camera.
        image_width (int): Image width, used to linearize pixel coordinates.
        image_height (int): Image height, used to linearize pixel coordinates.

    Returns:
        unique_pixels (JaggedTensor): The pixels with duplicates removed, first occurrence kept in order.
        inverse_indices (torch.Tensor): For each flat requested pixel, its index into the flat unique pixels.
        has_duplicates (bool): Whether anything was removed. When ``False`` the input is returned as is.
    """
    jdata = pixels_to_render.jdata
    total_pixels = jdata.shape[0]
    device = jdata.device
    if total_pixels == 0:
        return pixels_to_render, torch.empty(0, dtype=torch.long, device=device), False

    jidx = pixels_to_render.jidx
    single_list = jidx.shape[0] == 0
    rows = jdata[:, 0].long()
    cols = jdata[:, 1].long()
    keys = rows * image_width + cols
    if not single_list:
        keys = keys + jidx.long() * (image_height * image_width)

    sorted_keys, sort_perm = keys.sort()
    is_group_start = torch.ones(total_pixels, dtype=torch.bool, device=device)
    if total_pixels > 1:
        is_group_start[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_ids = is_group_start.long().cumsum(0) - 1
    num_unique = int(group_ids[-1].item()) + 1
    if num_unique == total_pixels:
        return pixels_to_render, torch.arange(total_pixels, dtype=torch.long, device=device), False

    inverse_indices = torch.empty(total_pixels, dtype=torch.long, device=device)
    inverse_indices[sort_perm] = group_ids
    unique_orig_indices = sort_perm[is_group_start.nonzero(as_tuple=False).squeeze(1)]
    unique_jdata = jdata[unique_orig_indices]

    num_lists = pixels_to_render.num_tensors
    if single_list:
        unique_batch_idx = torch.zeros(num_unique, dtype=torch.long, device=device)
    else:
        unique_batch_idx = jidx.long()[unique_orig_indices]
    counts_per_list = torch.bincount(unique_batch_idx, minlength=num_lists)
    new_offsets = torch.zeros(num_lists + 1, dtype=torch.long, device=device)
    new_offsets[1:] = counts_per_list.cumsum(0)
    unique_pixels = JaggedTensor.from_data_and_offsets(unique_jdata, new_offsets)
    return unique_pixels, inverse_indices, True


def intersect_gaussian_tiles_sparse(
    pixels_to_render: JaggedTensor | torch.Tensor,
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor | None = None,
    tile_size: int = 16,
    *,
    opacities: torch.Tensor | None = None,
) -> SparseGaussianTileIntersection:
    """Bin projected Gaussians into only the tiles that contain requested pixels.

    Not differentiable. Requested pixels are deduplicated per camera before the layout is built;
    the result records how to expand per-pixel outputs back to the requested order.

    Args:
        pixels_to_render (JaggedTensor | torch.Tensor): ``(row, col)`` integer pixels, one list per
            camera, or a ``[C, P, 2]`` tensor.
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor | None): Logit opacities, ``[N]``, for tighter culling.
        tile_size (int): Tile side length in pixels. The sparse kernels require ``16``.
        opacities (torch.Tensor | None): Precomputed ``[C, N]`` opacities from
            :func:`compute_gaussian_opacities`, to avoid recomputing them per stage. Derived from
            ``logit_opacities`` when ``None``.

    Returns:
        sparse_tiles (SparseGaussianTileIntersection): The sparse tile intersections.
    """
    pixels = as_pixel_jagged(pixels_to_render)
    num_tiles_h, num_tiles_w = _tile_grid(projected.image_width, projected.image_height, tile_size)
    unique_pixels, inverse_indices, has_duplicates = deduplicate_pixels(
        pixels, projected.image_width, projected.image_height
    )
    active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map = F.build_sparse_gaussian_tile_layout(
        tile_size, num_tiles_h, num_tiles_w, unique_pixels
    )
    conics, opacities = _culling_inputs(projected, logit_opacities, opacities)
    tile_offsets, tile_gaussian_ids = F.intersect_gaussian_tiles_sparse(
        projected.means2d,
        projected.radii,
        projected.depths,
        active_tile_mask,
        active_tiles,
        projected.num_cameras,
        tile_size,
        num_tiles_h,
        num_tiles_w,
        conics=conics,
        opacities=opacities,
    )
    return SparseGaussianTileIntersection(
        tile_offsets=tile_offsets,
        tile_gaussian_ids=tile_gaussian_ids,
        pixels_to_render=pixels,
        unique_pixels=unique_pixels,
        inverse_indices=inverse_indices,
        has_duplicates=has_duplicates,
        active_tiles=active_tiles,
        active_tile_mask=active_tile_mask,
        tile_pixel_mask=tile_pixel_mask,
        tile_pixel_cumsum=tile_pixel_cumsum,
        pixel_map=pixel_map,
        tile_size=tile_size,
        image_width=projected.image_width,
        image_height=projected.image_height,
    )
