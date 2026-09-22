# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Enums used by the Gaussian splatting API.

The camera enums are owned by :mod:`fvdb` and re-exported here unchanged, so
:class:`fvdb_reality_capture.CameraModel` is the same object as :class:`fvdb.CameraModel`.
:class:`GaussianRenderMode` belongs to the composable rendering pipeline in
:mod:`fvdb_reality_capture.functional` and is defined here.
"""

from enum import IntEnum

from fvdb import CameraModel, ProjectionMethod, RollingShutterType

__all__ = ["RollingShutterType", "CameraModel", "ProjectionMethod", "GaussianRenderMode"]


class GaussianRenderMode(IntEnum):
    """
    Which per-Gaussian features :func:`fvdb_reality_capture.functional.evaluate_gaussian_sh` produces
    for rasterization.
    """

    FEATURES = 0
    """Spherical-harmonics evaluated features only, ``[C, N, D]``."""

    DEPTH = 1
    """View-space depth only, ``[C, N, 1]``. No spherical harmonics are evaluated."""

    FEATURES_AND_DEPTH = 2
    """Spherical-harmonics features with depth appended as the last channel, ``[C, N, D + 1]``."""
