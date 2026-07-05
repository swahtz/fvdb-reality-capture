# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Standalone nvblox TSDF-fusion runner for cross-library benchmarks.

Designed to be invoked as a subprocess from a benchmark driver that
runs in a different conda env. This script runs in the `nvblox`
conda env (torch==2.6.0+cu124, CUDA toolkit 12.4, the nvblox_torch
wheel we built from source in install_nvblox.sh). It reads a
dataset-specific JSON spec from stdin or a file, integrates the
sweeps/frames via nvblox's `Mapper`, and writes the timing + voxel
stats JSON to stdout or an output file.

Supported workloads:
  - LiDAR sweeps (Mai City, KITTI): the spec provides a list of
    3D point clouds (in sensor or world frame; we always send
    sensor-frame to nvblox and pass the world pose alongside) plus
    HDL-64-style LiDAR intrinsics (num_azimuth, num_elevation,
    vertical_fov_rad). The runner reprojects each sweep to a
    (num_elevation, num_azimuth) depth image and hands it to
    `Mapper.add_depth_frame`. Nvblox's LiDAR integrator expects
    exactly this representation.
  - Depth frames (Replica, 7-Scenes): the spec provides depth
    image paths + camera intrinsics. The runner loads each depth
    image and hands it to `Mapper.add_depth_frame`.

Spec format (JSON):
    {
        "workload": "lidar" | "depth",
        "voxel_size_m": 0.2,
        "truncation_distance_m": 0.6,
        # LiDAR only:
        "lidar_num_azimuth": 1800,
        "lidar_num_elevation": 64,
        "lidar_vertical_fov_rad": 0.4712,  # ~27 deg for HDL-64
        "lidar_min_valid_range_m": 1.0,
        "lidar_sweeps_npz": "/path/to/mai_city_sweeps.npz",  # holds (points_per_frame, sensor_origins, cam_to_world)
        # LiDAR fast path (preferred; overrides lidar_sweeps_npz):
        # the driver precomputes the spherical range images once per
        # sweep (see sensitivity_utils.precompute_lidar_range_images)
        # so per-voxel-size subprocesses skip the reprojection.
        "lidar_range_images_npy": "/path/to/range_images_NxHxW.npy",
        "lidar_poses_npy": "/path/to/cam_to_world_Nx4x4.npy",
        # Depth only:
        "depth_image_paths": [...],
        "depth_intrinsics": {"fu": ..., "fv": ..., "cu": ..., "cv": ..., "width": ..., "height": ...},
        "depth_poses_npy": "/path/to/poses_Nx4x4.npy",
        "depth_scale": 1000.0,  # for uint16 mm depth
        "depth_max_m": 8.0,
        "warmup_frames": 2,
    }

Output (JSON to stdout or `--output`):
    {
        "ok": true,
        "ms_per_f": 123.45,
        "wall_s": 78.9,
        "peak_rss_gb": 3.14,
        "n_frames": 700,
        "n_voxels": 123456,      # nvblox's activeBlockCount * 512 (approximate)
        "n_mesh_verts": 10000,
        "n_mesh_tris": 20000,
        "voxel_size_m": 0.2,
        "failure": null,
    }
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.sensor import Sensor

from sensitivity_utils import (
    apply_gpu_mem_cap,
    squat_nbytes,
    points_to_spherical_range_image as _points_to_spherical_range_image,
    world_points_to_sensor_frame as _world_points_to_sensor_frame,
)


def _device_used_bytes() -> int:
    """Driver-reported used bytes on the visible device (all processes)."""
    free_b, total_b = torch.cuda.mem_get_info()
    return total_b - free_b


def _configure_mapper_params(spec: Dict[str, Any], workload: str) -> MapperParams:
    """Build MapperParams from the spec, honouring the sensitivity-
    analysis knobs.

    - Truncation: `truncation_distance_m` / voxel_size, set in voxels.
      Omit `truncation_distance_m` (or set it <= 0) to leave nvblox's
      upstream default (4.0 x voxel) untouched.
    - Max integration distance: optional `max_integration_distance_m`.
      The paper's workload-matched config uses 60 m for LiDAR; nvblox's
      upstream defaults are 10 m (LiDAR) / 7 m (depth). Omit to leave
      the upstream default untouched.

    On nvblox_torch >= 0.0.10 the integrator params live on a
    `ProjectiveIntegratorParams` sub-object reachable only through
    `MapperParams.get_/set_projective_integrator_params` (verified on
    the 0.0.10 build: the flat attribute spellings all hasattr-False).
    Older builds exposed flat attributes on MapperParams directly; we
    support both, and a candidate is only applied if the target object
    already exposes it (`hasattr`), so a plain-Python params class
    can't silently absorb a misspelled name.

    Returns (params, effective) where `effective` is a disclosure dict
    of the parameter values this run will use (explicitly set AND
    readable defaults) for the result JSON. A run that configures
    nothing still discloses the build's defaults.
    """
    params = MapperParams()
    effective: Dict[str, float] = {}

    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec.get("truncation_distance_m", -1.0))
    max_dist_m = spec.get("max_integration_distance_m", None)

    # Locate the object that carries the integrator params: the
    # ProjectiveIntegratorParams sub-object on new builds, MapperParams
    # itself on old ones.
    sub_params = None
    try:
        sub_params = params.get_projective_integrator_params()
    except AttributeError:
        pass
    target = sub_params if sub_params is not None else params

    trunc_attrs = {
        "lidar": ("projective_integrator_truncation_distance_vox",
                  "tsdf_integrator_truncation_distance_vox"),
        "depth": ("projective_integrator_truncation_distance_vox",
                  "tsdf_integrator_truncation_distance_vox"),
        "lidar_occupancy": ("occupancy_integrator_truncation_distance_vox",
                            "projective_integrator_truncation_distance_vox"),
    }[workload]
    if trunc_m > 0:
        for attr in trunc_attrs:
            if hasattr(target, attr):
                setattr(target, attr, float(trunc_m / voxel_size_m))
                break

    if max_dist_m is not None and float(max_dist_m) > 0:
        dist_attrs = (
            "lidar_projective_integrator_max_integration_distance_m",
            "projective_integrator_max_integration_distance_m",
        ) if workload in ("lidar", "lidar_occupancy") else (
            "projective_integrator_max_integration_distance_m",
        )
        for attr in dist_attrs:
            if hasattr(target, attr):
                setattr(target, attr, float(max_dist_m))
                break

    if sub_params is not None:
        params.set_projective_integrator_params(sub_params)

    # Echo the params this nvblox build will actually use (defaults
    # included, where readable) so every result JSON is self-disclosing.
    for attr in ("projective_integrator_truncation_distance_vox",
                 "tsdf_integrator_truncation_distance_vox",
                 "occupancy_integrator_truncation_distance_vox",
                 "lidar_projective_integrator_max_integration_distance_m",
                 "projective_integrator_max_integration_distance_m"):
        if hasattr(target, attr):
            try:
                effective[attr] = float(getattr(target, attr))
            except Exception:
                pass
    return params, effective


# NOTE: `_points_to_spherical_range_image` and
# `_world_points_to_sensor_frame` moved to `sensitivity_utils` (see
# the aliased imports at the top) so the bench drivers can precompute
# range images once per sweep and share them across subprocesses.


def _load_lidar_frames(spec: Dict[str, Any]):
    """Load the sweep data for a LiDAR workload spec.

    Returns (range_images, pts_concat, pts_offsets, cam_to_world,
    n_frames). Exactly one of `range_images` (fast path: the driver
    precomputed the spherical range images once per sweep; mmapped so
    frames page in on demand) or `pts_concat`+`pts_offsets` (legacy
    path: raw world-frame points, reprojected per frame here) is
    non-None.
    """
    if spec.get("lidar_range_images_npy"):
        range_images = np.load(spec["lidar_range_images_npy"], mmap_mode="r")
        cam_to_world_np = np.load(spec["lidar_poses_npy"])    # [n_frames, 4, 4]
        return range_images, None, None, cam_to_world_np, int(range_images.shape[0])
    data = np.load(spec["lidar_sweeps_npz"])
    pts_concat = data["points_per_frame_concat"]          # [N_total, 3]
    pts_offsets = data["points_per_frame_offsets"]        # [n_frames + 1]
    cam_to_world_np = data["cam_to_world"]                # [n_frames, 4, 4]
    return None, pts_concat, pts_offsets, cam_to_world_np, len(pts_offsets) - 1


def run_lidar(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox LiDAR TSDF on a Mai City-style sweep sequence.

    The npz at `lidar_sweeps_npz` is produced by `nvblox_bench_setup_mai_city`
    in the driver -- it's a dict with:
      - `points_per_frame_concat`: [total_points, 3] fp32, all frames concatenated
      - `points_per_frame_offsets`: [n_frames + 1] int64 (csr-style offsets)
      - `sensor_origins`: [n_frames, 3] fp32
      - `cam_to_world`: [n_frames, 4, 4] fp32
    This indirection is much faster than JSON-encoding 700 x 100k points.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    num_azimuth = int(spec["lidar_num_azimuth"])
    num_elevation = int(spec["lidar_num_elevation"])
    vertical_fov_rad = float(spec["lidar_vertical_fov_rad"])
    min_valid_range_m = float(spec.get("lidar_min_valid_range_m", 1.0))
    warmup_frames = int(spec.get("warmup_frames", 2))

    (range_images, pts_concat, pts_offsets,
     cam_to_world_np, n_frames) = _load_lidar_frames(spec)

    # Build the nvblox mapper up-front.
    apply_gpu_mem_cap(spec.get("gpu_mem_cap_gb"))
    # Baseline AFTER any in-process squat: the reported gpu_used_gb is
    # the delta this workload adds, excluding emulation squatters (in
    # this process or the driver's) and pre-existing contexts.
    base_used_b = _device_used_bytes()
    params, effective_params = _configure_mapper_params(spec,"lidar")

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.TSDF],
        mapper_parameters=params,
    )

    sensor = Sensor.from_lidar(
        num_azimuth_divisions=num_azimuth,
        num_elevation_divisions=num_elevation,
        vertical_fov_rad=vertical_fov_rad,
        min_valid_range_m=min_valid_range_m,
    )

    # Warmup on the first two frames to pre-allocate nvblox's block pool.
    def one_frame(i: int) -> None:
        pose = cam_to_world_np[i]
        if range_images is not None:
            # ascontiguousarray also copies out of the read-only mmap
            # so torch.from_numpy gets a writable buffer.
            depth_img = np.ascontiguousarray(range_images[i], dtype=np.float32)
        else:
            start, end = int(pts_offsets[i]), int(pts_offsets[i + 1])
            pts_s = _world_points_to_sensor_frame(pts_concat[start:end], pose)
            depth_img = _points_to_spherical_range_image(
                pts_s, num_azimuth, num_elevation, vertical_fov_rad)
        depth_t = torch.from_numpy(depth_img).cuda()
        pose_t = torch.from_numpy(pose).float()
        mapper.add_depth_frame(
            depth_frame=depth_t,
            t_w_c=pose_t,
            sensor=sensor,
            mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Voxel count: use the TSDF layer's active-block count * 512 as a
    # conservative upper bound on active voxels. nvblox stores an 8^3
    # block as a unit; voxels inside a block with zero weight still
    # "count" toward the allocation, so this inflates vs fvdb's
    # strict surface-only narrow-band number. We report it but call
    # out the caveat in the bench driver.
    tsdf_view = mapper.tsdf_layer_view(mapper_id=0)
    n_blocks = None
    try:
        n_blocks = int(tsdf_view.num_allocated_blocks())
    except Exception:
        try:
            n_blocks = int(tsdf_view.num_blocks())
        except Exception:
            n_blocks = None
    # nvblox uses 8^3 voxels per block by default; let the layer
    # report the actual block-dim-in-voxels so we do the right math
    # even if someone changes the block size in the future.
    block_dim_vox = 8
    try:
        block_dim_vox = int(tsdf_view.block_dim_in_voxels())
    except Exception:
        pass
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    # Approximate GPU-mem peak. `num_allocated_bytes()` reports the
    # live size of the TSDF layer only; we want the total nvblox
    # allocator footprint. Torch's cuda memory API works in-process
    # even though nvblox uses its own allocator pool, because nvblox
    # goes through the same CUDA driver -- `mem_get_info()` sees the
    # aggregate used memory.
    gpu_used_gb = -1.0
    try:
        gpu_used_gb = (_device_used_bytes() - base_used_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    # Mesh extraction for quality reference.
    n_mesh_verts = n_mesh_tris = -1
    try:
        mapper.update_color_mesh(mapper_id=0)
        mesh = mapper.get_color_mesh(mapper_id=0)
        n_mesh_verts = int(mesh.vertices.shape[0])
        n_mesh_tris = int(mesh.triangles.shape[0])
    except Exception:
        pass

    # Optional: time nvblox's ESDF-from-TSDF update step.
    #
    # Important caveat about `Mapper.update_esdf`: nvblox's ESDF is
    # INCREMENTAL BY DEFAULT. The first call after a new TSDF state
    # does the real work (building the ESDF across all dirty blocks);
    # subsequent calls on the same unchanged TSDF state hit an
    # internal "no dirty blocks" fast-path and take ~0.05 ms (just
    # the dirty-block check). So we time two things separately:
    #
    #   - `esdf_cold_ms`: the very first `update_esdf` call after
    #     TSDF fusion. This is the cost of "build the whole ESDF
    #     from scratch" — directly comparable to fvdb's stateless
    #     `compute_esdf`.
    #   - `esdf_warm_ms_*`: subsequent calls on the same TSDF state.
    #     These are the dirty-block-check no-op cost — directly
    #     comparable to fvdb's `compute_esdf_incremental` on a static
    #     scene (idempotent warm-start).
    esdf_cold_ms = -1.0
    esdf_warm_ms_min = -1.0
    esdf_warm_ms_median = -1.0
    esdf_warm_calls  = int(spec.get("esdf_warm_calls", 5))
    if bool(spec.get("with_esdf", False)):
        try:
            # Cold call: the only call where nvblox actually builds
            # the ESDF across all blocks. This is the cost we want
            # to compare against fvdb's one-shot `compute_esdf`.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mapper.update_esdf(mapper_id=0)
            torch.cuda.synchronize()
            esdf_cold_ms = (time.perf_counter() - t0) * 1000.0

            # Warm calls: same TSDF state, no dirty blocks. Should
            # all be near-zero because nvblox short-circuits on the
            # dirty-block check.
            warm_samples = []
            for _ in range(esdf_warm_calls):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mapper.update_esdf(mapper_id=0)
                torch.cuda.synchronize()
                warm_samples.append((time.perf_counter() - t0) * 1000.0)
            if warm_samples:
                warm_samples.sort()
                esdf_warm_ms_min = warm_samples[0]
                esdf_warm_ms_median = warm_samples[len(warm_samples) // 2]
        except Exception:
            # ESDF update may not be available on every mapper config.
            esdf_cold_ms = -2.0
            esdf_warm_ms_min = -2.0
            esdf_warm_ms_median = -2.0

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "n_mesh_verts": n_mesh_verts,
        "n_mesh_tris": n_mesh_tris,
        "voxel_size_m": voxel_size_m,
        "nvblox_params": effective_params,
        "gpu_mem_cap_gb": spec.get("gpu_mem_cap_gb"),
        "esdf_cold_ms": esdf_cold_ms,
        "esdf_warm_ms_min": esdf_warm_ms_min,
        "esdf_warm_ms_median": esdf_warm_ms_median,
        "esdf_warm_calls": esdf_warm_calls,
    }


def run_lidar_occupancy(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox LiDAR occupancy (log-odds) integration on a Mai
    City-style sweep sequence. Simpler than `run_lidar` (which does
    TSDF + optional ESDF + optional mesh) — occupancy is a single
    integrator type with a single per-voxel quantity, so we just:

      1. Build a Mapper with ProjectiveIntegratorType.OCCUPANCY.
      2. Feed each sweep through add_depth_frame (same spherical-
         range-image proxy as run_lidar).
      3. Report per-frame wall-clock, final block count, GPU memory.

    No ESDF, no mesh. For the paper's scale-ceiling comparison
    against fvdb's `integrate_occupancy_from_points_frames`.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    num_azimuth = int(spec["lidar_num_azimuth"])
    num_elevation = int(spec["lidar_num_elevation"])
    vertical_fov_rad = float(spec["lidar_vertical_fov_rad"])
    min_valid_range_m = float(spec.get("lidar_min_valid_range_m", 1.0))
    warmup_frames = int(spec.get("warmup_frames", 2))

    (range_images, pts_concat, pts_offsets,
     cam_to_world_np, n_frames) = _load_lidar_frames(spec)

    apply_gpu_mem_cap(spec.get("gpu_mem_cap_gb"))
    # Baseline AFTER any in-process squat: the reported gpu_used_gb is
    # the delta this workload adds, excluding emulation squatters (in
    # this process or the driver's) and pre-existing contexts.
    base_used_b = _device_used_bytes()
    params, effective_params = _configure_mapper_params(spec,"lidar_occupancy")

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.OCCUPANCY],
        mapper_parameters=params,
    )

    sensor = Sensor.from_lidar(
        num_azimuth_divisions=num_azimuth,
        num_elevation_divisions=num_elevation,
        vertical_fov_rad=vertical_fov_rad,
        min_valid_range_m=min_valid_range_m,
    )

    def one_frame(i: int) -> None:
        pose = cam_to_world_np[i]
        if range_images is not None:
            depth_img = np.ascontiguousarray(range_images[i], dtype=np.float32)
        else:
            start, end = int(pts_offsets[i]), int(pts_offsets[i + 1])
            pts_s = _world_points_to_sensor_frame(pts_concat[start:end], pose)
            depth_img = _points_to_spherical_range_image(
                pts_s, num_azimuth, num_elevation, vertical_fov_rad)
        depth_t = torch.from_numpy(depth_img).cuda()
        pose_t = torch.from_numpy(pose).float()
        mapper.add_depth_frame(
            depth_frame=depth_t, t_w_c=pose_t,
            sensor=sensor, mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Layer-size reporting. Occupancy uses a different layer view
    # than TSDF. Try the likely method names; tolerate absence.
    n_blocks = None
    block_dim_vox = 8
    for view_getter in ("occupancy_layer_view", "tsdf_layer_view"):
        try:
            v = getattr(mapper, view_getter)(mapper_id=0)
            try:
                n_blocks = int(v.num_allocated_blocks())
            except Exception:
                try:
                    n_blocks = int(v.num_blocks())
                except Exception:
                    continue
            try:
                block_dim_vox = int(v.block_dim_in_voxels())
            except Exception:
                pass
            break
        except Exception:
            continue
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    gpu_used_gb = -1.0
    try:
        gpu_used_gb = (_device_used_bytes() - base_used_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "voxel_size_m": voxel_size_m,
        "nvblox_params": effective_params,
        "gpu_mem_cap_gb": spec.get("gpu_mem_cap_gb"),
    }


def run_depth(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run nvblox depth-TSDF on a Replica / 7-Scenes-style sequence.

    The npz at `depth_npz` is produced by the driver and holds:
      - `depth_images`: [N, H, W] fp32 in METRES (0 = no measurement)
      - `cam_to_world`: [N, 4, 4] fp32 camera-to-world poses
      - `K`:            [3, 3] fp32 intrinsics matrix
            K[0,0]=fx, K[1,1]=fy, K[0,2]=cx, K[1,2]=cy
    Mirrors the indirection used by `run_lidar` — JSON-encoding large
    depth arrays is too slow.

    nvblox input format:
      - `Sensor.from_camera_matrix(K, W, H)`.
      - `Mapper.add_depth_frame(depth_frame, t_w_c, sensor)`, where
        `depth_frame` is a CUDA fp32 tensor and `t_w_c` is a 4x4
        host tensor (matching the test_mapper_add_frames example).

    Timing protocol and ESDF handling are identical to `run_lidar`.
    """
    voxel_size_m = float(spec["voxel_size_m"])
    trunc_m = float(spec["truncation_distance_m"])
    warmup_frames = int(spec.get("warmup_frames", 2))

    data = np.load(spec["depth_npz"])
    depth_images = data["depth_images"]   # [N, H, W] fp32 metres
    cam_to_world = data["cam_to_world"]   # [N, 4, 4] fp32
    K_np         = data["K"]              # [3, 3] fp32
    n_frames, height, width = depth_images.shape

    apply_gpu_mem_cap(spec.get("gpu_mem_cap_gb"))
    # Baseline AFTER any in-process squat: the reported gpu_used_gb is
    # the delta this workload adds, excluding emulation squatters (in
    # this process or the driver's) and pre-existing contexts.
    base_used_b = _device_used_bytes()
    params, effective_params = _configure_mapper_params(spec,"depth")

    mapper = Mapper(
        voxel_sizes_m=[voxel_size_m],
        integrator_types=[ProjectiveIntegratorType.TSDF],
        mapper_parameters=params,
    )

    K_t = torch.from_numpy(K_np).to(dtype=torch.float32)
    sensor = Sensor.from_camera_matrix(K_t, width, height)

    def one_frame(i: int) -> None:
        depth_t = torch.from_numpy(depth_images[i]).cuda()
        pose_t  = torch.from_numpy(cam_to_world[i]).float()
        mapper.add_depth_frame(
            depth_frame=depth_t, t_w_c=pose_t,
            sensor=sensor, mapper_id=0,
        )

    for i in range(min(warmup_frames, n_frames)):
        one_frame(i)
    mapper.clear(mapper_id=0)
    torch.cuda.synchronize()
    gc.collect()

    base_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_frames):
        one_frame(i)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    ms_per_f = wall_s * 1000 / n_frames

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    tsdf_view = mapper.tsdf_layer_view(mapper_id=0)
    n_blocks = None
    try:
        n_blocks = int(tsdf_view.num_allocated_blocks())
    except Exception:
        try:
            n_blocks = int(tsdf_view.num_blocks())
        except Exception:
            n_blocks = None
    block_dim_vox = 8
    try:
        block_dim_vox = int(tsdf_view.block_dim_in_voxels())
    except Exception:
        pass
    n_voxels_upper = (
        n_blocks * (block_dim_vox ** 3) if n_blocks is not None else -1)

    gpu_used_gb = -1.0
    try:
        gpu_used_gb = (_device_used_bytes() - base_used_b) / 1e9
    except Exception:
        pass
    peak_torch_gb = -1.0
    try:
        peak_torch_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass

    n_mesh_verts = n_mesh_tris = -1
    try:
        mapper.update_color_mesh(mapper_id=0)
        mesh = mapper.get_color_mesh(mapper_id=0)
        n_mesh_verts = int(mesh.vertices.shape[0])
        n_mesh_tris  = int(mesh.triangles.shape[0])
    except Exception:
        pass

    # ESDF timing (same cold/warm split as run_lidar).
    esdf_cold_ms = -1.0
    esdf_warm_ms_min = -1.0
    esdf_warm_ms_median = -1.0
    esdf_warm_calls = int(spec.get("esdf_warm_calls", 5))
    if bool(spec.get("with_esdf", False)):
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mapper.update_esdf(mapper_id=0)
            torch.cuda.synchronize()
            esdf_cold_ms = (time.perf_counter() - t0) * 1000.0

            warm_samples = []
            for _ in range(esdf_warm_calls):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mapper.update_esdf(mapper_id=0)
                torch.cuda.synchronize()
                warm_samples.append((time.perf_counter() - t0) * 1000.0)
            if warm_samples:
                warm_samples.sort()
                esdf_warm_ms_min = warm_samples[0]
                esdf_warm_ms_median = warm_samples[len(warm_samples) // 2]
        except Exception:
            esdf_cold_ms = -2.0
            esdf_warm_ms_min = -2.0
            esdf_warm_ms_median = -2.0

    return {
        "ok": True,
        "ms_per_f": ms_per_f,
        "wall_s": wall_s,
        "peak_rss_gb": peak_rss_kb / 1e6,
        "peak_rss_delta_gb": max(0.0, (peak_rss_kb - base_rss_kb) / 1e6),
        "gpu_used_gb": gpu_used_gb,
        "peak_torch_gb": peak_torch_gb,
        "n_frames": n_frames,
        "n_voxels": n_voxels_upper,
        "n_blocks": n_blocks if n_blocks is not None else -1,
        "n_mesh_verts": n_mesh_verts,
        "n_mesh_tris": n_mesh_tris,
        "voxel_size_m": voxel_size_m,
        "nvblox_params": effective_params,
        "gpu_mem_cap_gb": spec.get("gpu_mem_cap_gb"),
        "esdf_cold_ms": esdf_cold_ms,
        "esdf_warm_ms_min": esdf_warm_ms_min,
        "esdf_warm_ms_median": esdf_warm_ms_median,
        "esdf_warm_calls": esdf_warm_calls,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, help="Path to JSON spec file")
    p.add_argument("--output", required=True, help="Path to JSON output file")
    args = p.parse_args()

    with open(args.spec, "r") as f:
        spec = json.load(f)

    try:
        if spec["workload"] == "lidar":
            result = run_lidar(spec)
        elif spec["workload"] == "lidar_occupancy":
            result = run_lidar_occupancy(spec)
        elif spec["workload"] == "depth":
            result = run_depth(spec)
        else:
            raise NotImplementedError(
                f"workload {spec['workload']!r} not implemented (supported: "
                "'lidar', 'lidar_occupancy', 'depth')")
    except Exception as e:  # noqa: BLE001
        result = {
            "ok": False,
            "failure": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "voxel_size_m": spec.get("voxel_size_m", -1),
        }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
