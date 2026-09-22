# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Per-camera opacity computation shared by the rasterization and analysis stages."""

from __future__ import annotations

import torch

from ._types import ProjectedGaussians


def compute_gaussian_opacities(logit_opacities: torch.Tensor, projected: ProjectedGaussians) -> torch.Tensor:
    """Turn logit opacities into the per-camera opacities the rasterization kernels consume.

    Applies the sigmoid, repeats the result for every camera, and multiplies in the anti-aliasing
    compensation factors when the projection computed them.

    Args:
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        projected (ProjectedGaussians): Projection whose camera count and compensations to use.

    Returns:
        opacities (torch.Tensor): Opacities in ``[0, 1]``, shape ``[C, N]``, contiguous.
    """
    # The kernels require a contiguous [C, N] tensor, so the per-camera copy is materialized.
    opacities = torch.sigmoid(logit_opacities).repeat(projected.num_cameras, 1)
    if projected.compensations is not None:
        opacities = opacities * projected.compensations
    return opacities


def resolve_opacities(
    logit_opacities: torch.Tensor, projected: ProjectedGaussians, opacities: torch.Tensor | None
) -> torch.Tensor:
    """Return ``opacities`` if the caller precomputed them, else derive them from ``logit_opacities``.

    Lets a pipeline that runs several stages compute the ``[C, N]`` opacities once with
    :func:`compute_gaussian_opacities` and pass them through, instead of each stage materializing
    its own copy.

    Args:
        logit_opacities (torch.Tensor): Logit opacities, shape ``[N]``.
        projected (ProjectedGaussians): Projection whose camera count and compensations to use.
        opacities (torch.Tensor | None): Precomputed opacities, shape ``[C, N]``, or ``None``.

    Returns:
        opacities (torch.Tensor): Opacities in ``[0, 1]``, shape ``[C, N]``.
    """
    if opacities is not None:
        if tuple(opacities.shape) != (projected.num_cameras, projected.num_gaussians):
            raise ValueError(
                f"opacities must have shape [{projected.num_cameras}, {projected.num_gaussians}], got {tuple(opacities.shape)}"
            )
        return opacities
    return compute_gaussian_opacities(logit_opacities, projected)
