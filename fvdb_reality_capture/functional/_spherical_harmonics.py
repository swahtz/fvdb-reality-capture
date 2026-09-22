# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Stage 2 of the composable pipeline: view-dependent per-Gaussian features."""

from __future__ import annotations

import math
from typing import cast

import torch

from ..enums import GaussianRenderMode
from ._autograd import _EvaluateGaussianSHFn
from ._types import ProjectedGaussians


def sh_degree_from_coefficients(shN: torch.Tensor) -> int:
    """The spherical-harmonics degree implied by the number of higher-order coefficient bands.

    Args:
        shN (torch.Tensor): Higher-order coefficients, ``[N, K - 1, D]`` with ``K = (degree + 1)**2``.

    Returns:
        degree (int): The spherical-harmonics degree.
    """
    return math.isqrt(shN.shape[1] + 1) - 1


def evaluate_gaussian_sh(
    means: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    world_to_camera_matrices: torch.Tensor,
    projected: ProjectedGaussians,
    sh_degree_to_use: int = -1,
    render_mode: GaussianRenderMode = GaussianRenderMode.FEATURES,
) -> torch.Tensor:
    """Evaluate spherical harmonics into the per-camera, per-Gaussian features to rasterize.

    Gaussians culled by the projection (zero radii) receive zero features. Differentiable with
    respect to ``sh0``, ``shN``, ``means`` and ``world_to_camera_matrices``.

    Args:
        means (torch.Tensor): Gaussian centers in world space, ``[N, 3]``.
        sh0 (torch.Tensor): Degree-0 coefficients, ``[N, 1, D]``.
        shN (torch.Tensor): Higher-order coefficients, ``[N, K - 1, D]``. May have zero bands.
        world_to_camera_matrices (torch.Tensor): World-to-camera transforms, ``[C, 4, 4]``.
        projected (ProjectedGaussians): Projection of the same Gaussians into the same cameras.
        sh_degree_to_use (int): Highest degree to evaluate. ``-1`` uses every band in ``shN``.
        render_mode (GaussianRenderMode): Which features to produce; see the enum for shapes.

    Returns:
        features (torch.Tensor): ``[C, N, D]``, ``[C, N, 1]`` or ``[C, N, D + 1]`` depending on ``render_mode``.
    """
    render_mode = GaussianRenderMode(render_mode)
    depths = projected.depths.unsqueeze(-1)
    if render_mode == GaussianRenderMode.DEPTH:
        return depths

    available_degree = sh_degree_from_coefficients(shN)
    degree = available_degree if sh_degree_to_use < 0 else sh_degree_to_use
    if degree > available_degree:
        raise ValueError(f"sh_degree_to_use={degree} exceeds the degree {available_degree} available in shN")
    if degree == 0:
        shN = sh0.new_empty(sh0.shape[0], 0, sh0.shape[2])

    empty_ids = torch.empty(0, dtype=torch.int32, device=means.device)
    features = cast(
        torch.Tensor,
        _EvaluateGaussianSHFn.apply(
            degree,
            projected.num_cameras,
            means,
            world_to_camera_matrices,
            empty_ids,
            empty_ids,
            sh0,
            shN,
            projected.radii,
        ),
    )
    if render_mode == GaussianRenderMode.FEATURES_AND_DEPTH:
        features = torch.cat([features, depths], dim=-1)
    return features
