# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Per-cell isolated runner for the KITTI TSDF fvdb sweep.

fvdb's OOM at multi-billion-voxel KITTI grids aborts inside nanovdb's
CUDA error check (`CUDA error 2 ... Util.h`), killing the whole
driver process — a single bench_kitti.py invocation therefore loses
every cell after the first hard OOM. This wrapper runs ONE
(config, sequence, voxel) cell per subprocess, records a synthetic
OOM result when the subprocess dies with an out-of-memory abort, and
merges everything into the kitti_tsdf_{A,B}.fvdb.json files the
summarizer expects. Cells already present in those files are skipped,
so it is resumable and only fills the holes.

Usage:
    FVDB_PYTHON=... python run_kitti_fvdb_cells.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results_sensitivity"
KITTI_ROOT = HERE.parent / "data" / "kitti"
FVDB_PYTHON = os.environ.get(
    "FVDB_PYTHON", "/home/jswartz/Development/fvdb-core-pr656/.venv/bin/python")

CONFIGS = {
    "A": {
        "voxels": [0.4, 0.2, 0.1, 0.05, 0.03, 0.02],
        "extra": ["--truncation-multiplier", "3.0",
                  "--nvblox-max-integration-distance-m", "60"],
    },
    "B": {
        "voxels": [0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01],
        "extra": ["--truncation-multiplier", "4.0", "--max-range-m", "10"],
    },
}
SEQUENCES = ["00", "02", "05"]
MEM_CAP = "51.5"


def load_results(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text()).get("results", [])
    return []


def main() -> None:
    for cfg, spec in CONFIGS.items():
        merged_path = OUT / f"kitti_tsdf_{cfg}.fvdb.json"
        results = load_results(merged_path)
        have = {(r.get("sequence"), float(r["voxel_size"]))
                for r in results if r.get("system", "").startswith("fvdb")}
        # Once a (seq) hits OOM, finer voxels in the same config are
        # also OOM; still run them so every cell is measured, but a
        # cheap ordering (coarse -> fine) means the OK cells complete
        # before the abort-and-record ones.
        for seq in SEQUENCES:
            for vox in spec["voxels"]:
                if (seq, vox) in have:
                    print(f"[skip] {cfg} seq={seq} vox={vox} (already recorded)",
                          flush=True)
                    continue
                cell_json = OUT / f"cell_{cfg}_{seq}_{vox}.json"
                cmd = [FVDB_PYTHON, str(HERE / "bench_kitti.py"),
                       "--root", str(KITTI_ROOT),
                       "--sequences", seq,
                       "--systems", "fvdb",
                       "--voxel-sizes", str(vox),
                       "--gpu-mem-cap-gb", MEM_CAP,
                       "--json-out", str(cell_json),
                       *spec["extra"]]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=7200)
                wall = time.time() - t0
                cell_results = load_results(cell_json)
                if cell_results:
                    r = cell_results[0]
                elif ("out of memory" in proc.stdout.lower()
                      or "out of memory" in proc.stderr.lower()):
                    r = {"system": "fvdb", "ok": False, "voxel_size": vox,
                         "sequence": seq, "dataset": "kitti",
                         "failure": "OOM: hard abort in nanovdb CUDA "
                                    "allocator (CUDA error 2: out of memory)",
                         "wall_s": wall}
                else:
                    r = {"system": "fvdb", "ok": False, "voxel_size": vox,
                         "sequence": seq, "dataset": "kitti",
                         "failure": f"driver died (rc={proc.returncode}): "
                                    f"{proc.stderr[-200:]!r}",
                         "wall_s": wall}
                r.setdefault("sequence", seq)
                results.append(r)
                merged_path.write_text(json.dumps(
                    {"results": results,
                     "args": {"per_cell_isolated": True, "config": cfg}},
                    indent=2))
                status = ("OK" if r.get("ok")
                          else ("OOM" if "OOM" in str(r.get("failure", ""))
                                else "FAIL"))
                extra = (f" {r.get('ms_per_f', 0):.2f} ms/f "
                         f"{r.get('peak_gb', 0):.2f} GB"
                         if r.get("ok") else f" {r.get('failure', '')[:60]}")
                print(f"[{cfg} seq={seq} vox={vox*100:g}cm] {status}"
                      f" ({wall:.0f}s){extra}", flush=True)
                cell_json.unlink(missing_ok=True)
    print("ALL KITTI FVDB CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
