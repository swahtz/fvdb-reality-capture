# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
KITTI Odometry counterpart to `bench_esdf_vs_nvblox.py`.

Reuses the per-config ESDF runner from `bench_esdf_vs_nvblox.py`
(loader-agnostic — it only depends on the duck-typed scene
interface satisfied by both `MaiCityScene` and `KittiScene`),
sweeps over multiple sequences and multiple voxel sizes, and
writes a combined JSON suitable for paper-table generation.

Default voxel sweep aligns with the LiDAR ESDF table in
`PAPER_SECTION.md` §3.4. Default sequences are the standard
NICE-SLAM trio (00, 02, 05).

Usage:

    cd fvdb-reality-capture/tests/benchmarks/tsdf_fusion/cross_library
    CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \\
    /home/fwilliams/bin/miniconda3/envs/fvdb/bin/python \\
        bench_esdf_kitti.py \\
        --root .../data/KITTI \\
        --sequences 00 02 05 \\
        --n-frames 100 \\
        --voxel-sizes-m 0.2 0.1 0.05 0.03 0.02 \\
        --json-out ./results/kitti_esdf.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_loader import load_kitti_scene  # noqa: E402
import bench_esdf_vs_nvblox as _esdf_bench  # noqa: E402
from bench_esdf_vs_nvblox import (  # noqa: E402
    _format_scale_table,
    _run_one_config,
)
from sensitivity_utils import (  # noqa: E402
    apply_gpu_mem_cap,
    clip_scene_to_max_range,
)


# Mai City evidence: nvblox ESDF OOMs at 2 cm. KITTI has more frames
# per sequence -> denser working set -> nvblox will OOM at least as
# early. Skip the cells we know will fail to save wall time.
_KNOWN_OOM_NVBLOX = lambda vs: vs <= 0.02   # 2 cm and below


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="Path to KITTI root (contains dataset/sequences/...)")
    ap.add_argument("--sequences", nargs="+", default=["00", "02", "05"])
    ap.add_argument("--n-frames", type=int, default=100,
                    help="frames per sequence used to build TSDF before "
                         "ESDF timing (default 100, matching Mai City)")
    ap.add_argument("--voxel-sizes-m", type=float, nargs="+",
                    default=[0.2, 0.1, 0.05, 0.03, 0.02])
    ap.add_argument("--trunc-voxel-multiplier", type=float, default=3.0)
    ap.add_argument("--max-distance-voxel-multiplier", type=float, default=10.0)
    ap.add_argument("--esdf-warm-calls", type=int, default=5)
    ap.add_argument("--skip-known-oom", action="store_true", default=True)
    ap.add_argument("--no-skip-known-oom", dest="skip_known_oom",
                    action="store_false")
    ap.add_argument("--skip-nvblox", action="store_true",
                    help="skip nvblox at every voxel (fvdb-only run)")
    ap.add_argument("--json-out", type=Path, default=None)
    # --- Sensitivity-analysis knobs (see bench_mai_city.py) ---
    ap.add_argument("--nvblox-python", type=str, default=None)
    ap.add_argument("--nvblox-max-integration-distance-m", type=float,
                    default=None,
                    help="60 = paper's workload-matched config; omit for "
                         "nvblox's upstream default (10 m)")
    ap.add_argument("--max-range-m", type=float, default=None)
    ap.add_argument("--gpu-mem-cap-gb", type=float, default=None,
                    help="emulate a smaller GPU (51.5 = paper's 48 GB Ada)")
    args = ap.parse_args()

    if args.nvblox_python is not None:
        _esdf_bench.NVBLOX_ENV_PYTHON = args.nvblox_python
    if (args.skip_known_oom
            and args.nvblox_max_integration_distance_m is None):
        # The known-OOM cutoff was measured at the paper's 60 m
        # workload-matched config; at nvblox's upstream defaults the
        # OOM threshold is exactly what we're trying to measure.
        print("[note] --skip-known-oom disabled: nvblox is running at its "
              "upstream defaults.", flush=True)
        args.skip_known_oom = False

    all_results: list[dict[str, Any]] = []
    t_start = time.time()

    for seq in args.sequences:
        print(f"\n##### KITTI seq={seq!r} #####")
        scene = load_kitti_scene(
            root_dir=args.root, sequence=seq,
            max_frames=args.n_frames,
        )
        print(f"[load] done. {scene.n_frames} frames, "
              f"{scene.total_points / 1e6:.2f} M points total")
        if args.max_range_m is not None:
            scene = clip_scene_to_max_range(scene, args.max_range_m)
        apply_gpu_mem_cap(args.gpu_mem_cap_gb)

        for vs in args.voxel_sizes_m:
            cfg_skip_nvblox = args.skip_nvblox or (
                args.skip_known_oom and _KNOWN_OOM_NVBLOX(vs))
            r = _run_one_config(
                scene,
                voxel_size=float(vs),
                truncation=float(vs) * args.trunc_voxel_multiplier,
                max_distance=float(vs) * args.max_distance_voxel_multiplier,
                esdf_warm_calls=args.esdf_warm_calls,
                skip_nvblox=cfg_skip_nvblox,
                nvblox_max_integration_distance_m=(
                    args.nvblox_max_integration_distance_m),
                gpu_mem_cap_gb=args.gpu_mem_cap_gb,
            )
            r["sequence"] = seq
            r["dataset"] = "kitti"
            all_results.append(r)
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                with args.json_out.open("w") as f:
                    json.dump({
                        "config": {
                            "sequences": args.sequences,
                            "n_frames": args.n_frames,
                            "esdf_warm_calls": args.esdf_warm_calls,
                            "trunc_voxel_multiplier":
                                args.trunc_voxel_multiplier,
                            "max_distance_voxel_multiplier":
                                args.max_distance_voxel_multiplier,
                        },
                        "results": all_results,
                    }, f, indent=2)

    print("")
    print(_format_scale_table(all_results))
    print(f"\n##### Total wall time: {(time.time() - t_start) / 60:.1f} min #####")


if __name__ == "__main__":
    main()
