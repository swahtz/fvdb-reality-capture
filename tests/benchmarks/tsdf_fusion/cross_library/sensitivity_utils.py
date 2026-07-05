# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the nvblox-defaults sensitivity analysis.

Two knobs let us re-run the Table 2 / Table 5 sweeps at nvblox's
upstream default operating point instead of the paper's
workload-matched configuration, on hardware different from the
paper's 48 GB RTX 6000 Ada:

  - `clip_scene_to_max_range`: radially clips every frame's point
    cloud to `max_range_m` of its sensor origin. nvblox enforces its
    integration range internally
    (`lidar_projective_integrator_max_integration_distance_m`,
    upstream default 10 m); fvdb / VDBFusion integrate whatever
    points they are given, so matching nvblox's default workload
    means pre-clipping their input to the same radius.

  - `apply_gpu_mem_cap`: emulates a smaller GPU by pre-allocating
    (and holding) a "squatter" buffer so the visible device total
    shrinks to `cap_total_gb`. The paper's OOM thresholds were
    measured on a 48 GB RTX 6000 Ada (49140 MiB = 51.5e9 bytes
    total); on a 96 GB RTX PRO 6000 Blackwell, pass
    `cap_total_gb=51.5` to reproduce the same memory budget. Both
    torch-side (fvdb) and cudaMalloc-side (nvblox) allocations see
    the reduced free memory because the squat is real device memory.

Memory-accounting caveat: the squat itself is counted by
`torch.cuda.max_memory_allocated()` and by `torch.cuda.mem_get_info`
deltas. Report workload peaks with `squat_nbytes()` subtracted.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np

# Held reference to the squatter tensor. Module-global on purpose:
# the cap lasts for the process lifetime.
_SQUAT_TENSOR = None
_SQUAT_NBYTES = 0


def clip_scene_to_max_range(scene, max_range_m: float):
    """Return a shallow copy of `scene` with every frame's points
    clipped to within `max_range_m` of that frame's sensor origin.

    Works on the duck-typed scene interface shared by MaiCityScene /
    KittiScene (`points_per_frame` list of [N,3] fp32 world-frame
    arrays + `sensor_origins` [F,3]). `n_frames` / `total_points`
    are properties derived from `points_per_frame`, so no fixup is
    needed beyond replacing the list.
    """
    clipped = copy.copy(scene)
    new_pts = []
    kept = 0
    total = 0
    for i, pts in enumerate(scene.points_per_frame):
        origin = scene.sensor_origins[i]
        r2 = np.sum((pts - origin[None, :]) ** 2, axis=1)
        mask = r2 <= (max_range_m * max_range_m)
        total += pts.shape[0]
        kept += int(mask.sum())
        new_pts.append(np.ascontiguousarray(pts[mask]))
    clipped.points_per_frame = new_pts
    print(f"[max-range clip] {max_range_m} m: kept {kept:,} / {total:,} "
          f"points ({100.0 * kept / max(total, 1):.1f}%)", flush=True)
    return clipped


def apply_gpu_mem_cap(cap_total_gb: Optional[float]) -> int:
    """Shrink the visible GPU to `cap_total_gb` (decimal GB of
    emulated TOTAL device memory) by cudaMalloc-ing the difference
    and holding it for the life of the process. Returns the squat
    size in bytes (0 if no cap or the device is already smaller).

    Call once, early, before any large allocations. Idempotent.
    """
    global _SQUAT_TENSOR, _SQUAT_NBYTES
    if cap_total_gb is None or cap_total_gb <= 0:
        return 0
    if _SQUAT_TENSOR is not None:
        return _SQUAT_NBYTES
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    squat = int(total_b - cap_total_gb * 1e9)
    if squat <= 0:
        print(f"[gpu-mem-cap] device total {total_b / 1e9:.1f} GB <= cap "
              f"{cap_total_gb} GB; no squat needed", flush=True)
        return 0
    _SQUAT_TENSOR = torch.empty(squat, dtype=torch.uint8, device="cuda")
    _SQUAT_NBYTES = squat
    free_after, _ = torch.cuda.mem_get_info()
    print(f"[gpu-mem-cap] emulating a {cap_total_gb:.1f} GB device: "
          f"squatted {squat / 1e9:.1f} GB, {free_after / 1e9:.1f} GB free "
          f"remains", flush=True)
    return squat


def squat_nbytes() -> int:
    """Bytes held by the memory-cap squatter (0 if uncapped)."""
    return _SQUAT_NBYTES


# =====================================================================
# LiDAR spherical range-image helpers.
#
# These were originally private to `nvblox_runner.py` (which runs in
# the dedicated nvblox conda env). They live here now so the bench
# DRIVERS (fvdb env) can precompute range images once per sweep and
# share them across every per-voxel-size nvblox subprocess -- the
# reprojection is voxel-size independent, and redoing it (plus
# re-writing a ~1 GB points npz) in every subprocess dominated nvblox
# bench wall time (~47 s of overhead vs ~13 s of integration per
# voxel size on Mai City). numpy-only: importable from both envs.
# =====================================================================


def world_points_to_sensor_frame(
    points_world: np.ndarray,
    sensor_to_world: np.ndarray,
) -> np.ndarray:
    """Invert the sensor pose to bring world-frame points back into
    the sensor-local frame nvblox expects.
    `sensor_to_world` is the 4x4 camera-to-world transform.
    """
    # p_s = R_ws^T (p_w - t_ws)
    R = sensor_to_world[:3, :3]
    t = sensor_to_world[:3, 3]
    return (points_world - t[None, :]) @ R


def points_to_spherical_range_image(
    points_sensor: np.ndarray,
    num_azimuth: int,
    num_elevation: int,
    vertical_fov_rad: float,
) -> np.ndarray:
    """Reproject an [N, 3] sensor-frame point cloud to an
    (H, W) = (num_elevation, num_azimuth) range image matching
    nvblox's `Sensor.from_lidar` parameterisation.

    Reference: nvblox's internal `Lidar::project` uses
        azimuth = atan2(y, x)                              # (-pi, pi]
        elevation = atan2(z, sqrt(x**2 + y**2))            # (-v_fov/2, +v_fov/2)
        u = (azimuth + pi) / (2*pi) * num_azimuth
        v = (elevation + v_fov/2) / v_fov * num_elevation

    Points outside the vertical FoV or mapped to the same pixel as a
    closer point are discarded (kept: min range per pixel).

    Returns an fp32 (num_elevation, num_azimuth) array; pixels with
    no hit are 0.0 (nvblox skips depth==0 automatically).
    """
    if points_sensor.shape[0] == 0:
        return np.zeros((num_elevation, num_azimuth), dtype=np.float32)

    x = points_sensor[:, 0]
    y = points_sensor[:, 1]
    z = points_sensor[:, 2]
    radial_xy = np.sqrt(x * x + y * y)
    ranges = np.sqrt(radial_xy * radial_xy + z * z)

    # Filter zero-range and out-of-FoV points. Keep epsilon for
    # numerical stability at the pole (radial_xy == 0 would NaN
    # atan2(z, 0) to +/-pi/2 which is fine but let's be defensive).
    valid = (ranges > 1e-4) & (radial_xy > 1e-6)
    x, y, z, radial_xy, ranges = x[valid], y[valid], z[valid], radial_xy[valid], ranges[valid]

    azimuth = np.arctan2(y, x)                   # (-pi, pi]
    elevation = np.arctan2(z, radial_xy)         # roughly (-v_fov/2, +v_fov/2) for LiDAR

    # Clamp into the valid range the sensor accepts.
    el_min = -vertical_fov_rad / 2.0
    el_max = +vertical_fov_rad / 2.0
    in_fov = (elevation >= el_min) & (elevation < el_max)
    azimuth, elevation, ranges = azimuth[in_fov], elevation[in_fov], ranges[in_fov]

    # Map to pixel indices.
    u = ((azimuth + math.pi) / (2.0 * math.pi) * num_azimuth).astype(np.int64)
    v = ((elevation - el_min) / vertical_fov_rad * num_elevation).astype(np.int64)
    # Clip to valid range (guards against +pi azimuth rounding to num_azimuth).
    u = np.clip(u, 0, num_azimuth - 1)
    v = np.clip(v, 0, num_elevation - 1)

    # Keep min range per pixel. Vectorised via numpy: combine (v, u)
    # into a single flat index, then sort by range ascending and use
    # np.unique on the flat index with `return_index=True` to pick
    # the first (smallest-range) entry per pixel.
    flat_idx = v * num_azimuth + u
    order = np.argsort(ranges)              # ascending
    flat_sorted = flat_idx[order]
    range_sorted = ranges[order]
    _, first_idx = np.unique(flat_sorted, return_index=True)
    pick_flat = flat_sorted[first_idx]
    pick_range = range_sorted[first_idx]

    depth = np.zeros(num_elevation * num_azimuth, dtype=np.float32)
    depth[pick_flat] = pick_range.astype(np.float32)
    return depth.reshape(num_elevation, num_azimuth)


# Cache of precomputed range-image stacks, keyed by (id(scene),
# num_azimuth, num_elevation, vertical_fov_rad). Each value holds a
# strong reference to `scene` (so a dead object's id can't be reused
# by a different scene) and to the TemporaryDirectory (so the .npy
# files outlive the call; they are removed at interpreter exit).
_RANGE_IMAGE_CACHE: Dict[tuple, tuple] = {}


def precompute_lidar_range_images(
    scene,
    num_azimuth: int,
    num_elevation: int,
    vertical_fov_rad: float,
) -> Tuple[str, str]:
    """Reproject every frame of `scene` to a spherical range image
    ONCE and stash the stack + poses as raw .npy files (mmap-friendly).

    Returns (range_images_npy, cam_to_world_npy) paths for the spec
    keys `lidar_range_images_npy` / `lidar_poses_npy` understood by
    `nvblox_runner.py`. Repeated calls with the same scene object and
    lidar intrinsics are cache hits for the lifetime of the process,
    so a whole voxel-size sweep shares one precompute.
    """
    key = (id(scene), int(num_azimuth), int(num_elevation),
           float(vertical_fov_rad))
    hit = _RANGE_IMAGE_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1]

    t0 = time.perf_counter()
    n_frames = scene.n_frames
    imgs = np.zeros((n_frames, num_elevation, num_azimuth), dtype=np.float32)
    for i in range(n_frames):
        pts_s = world_points_to_sensor_frame(
            scene.points_per_frame[i], scene.cam_to_world[i])
        imgs[i] = points_to_spherical_range_image(
            pts_s, num_azimuth, num_elevation, vertical_fov_rad)

    tmpdir = tempfile.TemporaryDirectory(prefix="nvblox_range_images_")
    imgs_path = os.path.join(tmpdir.name, "range_images.npy")
    poses_path = os.path.join(tmpdir.name, "cam_to_world.npy")
    np.save(imgs_path, imgs)
    np.save(poses_path,
            np.ascontiguousarray(scene.cam_to_world, dtype=np.float32))
    print(f"[nvblox-prep] precomputed {n_frames} range images "
          f"({num_elevation}x{num_azimuth}) in "
          f"{time.perf_counter() - t0:.1f} s; shared across the sweep")
    _RANGE_IMAGE_CACHE[key] = (imgs_path, poses_path, tmpdir, scene)
    return imgs_path, poses_path
