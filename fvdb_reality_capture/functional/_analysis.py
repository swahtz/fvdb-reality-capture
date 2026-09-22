# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Non-differentiable analysis of which Gaussians contribute to which pixels."""

from __future__ import annotations

import torch
from fvdb import JaggedTensor
from fvdb import functional as F

from ._opacity import compute_gaussian_opacities
from ._types import GaussianTileIntersection, ProjectedGaussians, SparseGaussianTileIntersection


def rasterize_num_contributing_gaussians(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count the Gaussians that contribute non-negligible opacity to each pixel.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.

    Returns:
        num_contributing (torch.Tensor): Contributor count per pixel, ``[C, H, W]``, ``int32``.
        alphas (torch.Tensor): Accumulated alpha per pixel, ``[C, H, W]``.
    """
    with torch.no_grad():
        return _count_dense(projected, compute_gaussian_opacities(logit_opacities, projected), tiles)


def _count_dense(
    projected: ProjectedGaussians, opacities: torch.Tensor, tiles: GaussianTileIntersection
) -> tuple[torch.Tensor, torch.Tensor]:
    return F.rasterize_num_contributing_gaussians(
        projected.means2d,
        projected.conics,
        opacities,
        tiles.tile_offsets,
        tiles.tile_gaussian_ids,
        tiles.image_width,
        tiles.image_height,
        0,
        0,
        tiles.tile_size,
    )


def rasterize_contributing_gaussian_ids(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    tiles: GaussianTileIntersection,
    top_k_contributors: int = 0,
    num_contributing: torch.Tensor | None = None,
) -> tuple[JaggedTensor, JaggedTensor]:
    """List the Gaussians contributing to each pixel, front to back, with their blend weights.

    With ``top_k_contributors > 0`` at most that many of the most visible contributors are kept per
    pixel. Otherwise every contributor is listed, which needs the per-pixel counts; they are computed
    here unless ``num_contributing`` supplies them.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        tiles (GaussianTileIntersection): Output of :func:`intersect_gaussian_tiles` for ``projected``.
        top_k_contributors (int): Contributors to keep per pixel, or ``0`` for all of them.
        num_contributing (torch.Tensor | None): Counts from :func:`rasterize_num_contributing_gaussians`,
            ``[C, H, W]``. Used only when ``top_k_contributors <= 0``.

    Returns:
        gaussian_ids (JaggedTensor): Contributor indices, nested as cameras, then pixels, then contributors.
        weights (JaggedTensor): Blend weight of each listed contributor, same structure.
    """
    with torch.no_grad():
        opacities = compute_gaussian_opacities(logit_opacities, projected)
        if top_k_contributors <= 0 and num_contributing is None:
            num_contributing, _ = _count_dense(projected, opacities, tiles)
        return F.rasterize_contributing_gaussian_ids(
            projected.means2d,
            projected.conics,
            opacities,
            tiles.tile_offsets,
            tiles.tile_gaussian_ids,
            tiles.image_width,
            tiles.image_height,
            0,
            0,
            tiles.tile_size,
            top_k_contributors,
            num_contributing if top_k_contributors <= 0 else None,
        )


def rasterize_num_contributing_gaussians_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Count contributing Gaussians at the requested pixels only.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        sparse_tiles (SparseGaussianTileIntersection): Output of :func:`intersect_gaussian_tiles_sparse`.

    Returns:
        num_contributing (JaggedTensor): Contributor count per requested pixel, ``int32``, one list per camera.
        alphas (JaggedTensor): Accumulated alpha per requested pixel, one list per camera.
    """
    with torch.no_grad():
        counts, alphas = _count_sparse_unique(
            projected, compute_gaussian_opacities(logit_opacities, projected), sparse_tiles
        )
    requested = sparse_tiles.pixels_to_render
    return (
        requested.jagged_like(sparse_tiles.expand_to_requested(counts.jdata)),
        requested.jagged_like(sparse_tiles.expand_to_requested(alphas.jdata)),
    )


def _count_sparse_unique(
    projected: ProjectedGaussians, opacities: torch.Tensor, sparse_tiles: SparseGaussianTileIntersection
) -> tuple[JaggedTensor, JaggedTensor]:
    """Contributor counts and alphas over the unique pixels, as the kernel produces them."""
    return F.rasterize_num_contributing_gaussians_sparse(
        projected.means2d,
        projected.conics,
        opacities,
        sparse_tiles.tile_offsets,
        sparse_tiles.tile_gaussian_ids,
        sparse_tiles.unique_pixels,
        sparse_tiles.active_tiles,
        sparse_tiles.tile_pixel_mask,
        sparse_tiles.tile_pixel_cumsum,
        sparse_tiles.pixel_map,
        sparse_tiles.image_width,
        sparse_tiles.image_height,
        0,
        0,
        sparse_tiles.tile_size,
    )


def _expand_contributions(sparse_tiles: SparseGaussianTileIntersection, per_unique: JaggedTensor) -> JaggedTensor:
    """Expand a per-unique-pixel contributor list to the requested pixels, repeating duplicates.

    The result nests cameras, then requested pixels, then contributors, so it has the same list
    structure the dense kernel returns.
    """
    device = per_unique.jdata.device
    inverse = sparse_tiles.inverse_indices
    offsets_unique = per_unique.joffsets.to(device)
    starts = offsets_unique[inverse]
    counts = offsets_unique[1:][inverse] - starts
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)
    segment = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
    within = torch.arange(int(offsets[-1].item()), device=device) - offsets[segment]
    gather = starts[segment] + within
    requested = sparse_tiles.pixels_to_render
    camera = requested.jidx.to(device).long()
    within_camera = torch.arange(camera.numel(), device=device) - requested.joffsets.to(device)[camera]
    list_ids = torch.stack([camera, within_camera], dim=1).to(torch.int32)
    return JaggedTensor.from_data_offsets_and_list_ids(per_unique.jdata.index_select(0, gather), offsets, list_ids)


def rasterize_contributing_gaussian_ids_sparse(
    projected: ProjectedGaussians,
    logit_opacities: torch.Tensor,
    sparse_tiles: SparseGaussianTileIntersection,
    top_k_contributors: int = 0,
) -> tuple[JaggedTensor, JaggedTensor]:
    """List contributing Gaussians, with blend weights, at the requested pixels only.

    Mode selection follows :func:`rasterize_contributing_gaussian_ids`. Results are returned in the
    order of ``sparse_tiles.pixels_to_render``, duplicates included.

    Args:
        projected (ProjectedGaussians): Output of :func:`project_gaussians`.
        logit_opacities (torch.Tensor): Logit opacities, ``[N]``.
        sparse_tiles (SparseGaussianTileIntersection): Output of :func:`intersect_gaussian_tiles_sparse`.
        top_k_contributors (int): Contributors to keep per pixel, or ``0`` for all of them.

    Returns:
        gaussian_ids (JaggedTensor): Contributor indices, nested as cameras, then pixels, then contributors.
        weights (JaggedTensor): Blend weight of each listed contributor, same structure.
    """
    with torch.no_grad():
        opacities = compute_gaussian_opacities(logit_opacities, projected)
        num_contributing = None
        if top_k_contributors <= 0:
            num_contributing, _ = _count_sparse_unique(projected, opacities, sparse_tiles)
        ids, weights = F.rasterize_contributing_gaussian_ids_sparse(
            projected.means2d,
            projected.conics,
            opacities,
            sparse_tiles.tile_offsets,
            sparse_tiles.tile_gaussian_ids,
            sparse_tiles.unique_pixels,
            sparse_tiles.active_tiles,
            sparse_tiles.tile_pixel_mask,
            sparse_tiles.tile_pixel_cumsum,
            sparse_tiles.pixel_map,
            sparse_tiles.image_width,
            sparse_tiles.image_height,
            0,
            0,
            sparse_tiles.tile_size,
            top_k_contributors,
            num_contributing,
        )
        if sparse_tiles.has_duplicates:
            ids = _expand_contributions(sparse_tiles, ids)
            weights = _expand_contributions(sparse_tiles, weights)
    return ids, weights
