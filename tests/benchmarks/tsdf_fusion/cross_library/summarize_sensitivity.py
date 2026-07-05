# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Summarize the nvblox-defaults sensitivity sweeps into rebuttal-ready
markdown tables.

Reads the JSON files produced by `run_sensitivity_sweeps.sh` from
`--results-dir` (default: ../results_sensitivity) and prints, per
dataset/workload, a table with the paper-matched config (A: 60 m,
tau=3x) side by side with the nvblox-upstream-defaults config (B:
10 m, tau=4x), flagging the OOM threshold of each system under each
config.

Usage:
    python summarize_sensitivity.py [--results-dir DIR] [--out FILE.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_cell(ms, gb) -> str:
    if ms is None:
        return "--"
    return f"{ms:.2f} / {gb:.2f}"


def _is_oom(failure: str | None) -> bool:
    if not failure:
        return False
    f = str(failure).lower()
    return ("oom" in f or "out of memory" in f or "outofmemory" in f
            or "cudamalloc" in f or "bad_alloc" in f)


def _tsdf_rows(path: Path) -> dict:
    """{(sequence or '', voxel): {system: cell}} from a TSDF sweep.

    The sweeps run fvdb and nvblox in separate driver invocations, so
    a config's results may be split across `<stem>.json`,
    `<stem>.fvdb.json` and `<stem>.nvblox.json`; merge whichever exist.
    """
    paths = [path] + sorted(path.parent.glob(path.stem + ".*.json"))
    results: list = []
    for p in paths:
        if p.exists():
            results.extend(json.loads(p.read_text()).get("results", []))
    rows: dict = {}
    for r in results:
        key = (r.get("sequence", ""), float(r["voxel_size"]))
        cell: dict = {"ok": bool(r.get("ok"))}
        if r.get("ok"):
            cell["ms"] = r.get("ms_per_f")
            cell["gb"] = r.get("peak_gb")
            cell["n_voxels"] = r.get("n_voxels")
        else:
            cell["oom"] = _is_oom(r.get("failure"))
            cell["skipped"] = bool(r.get("skipped"))
            cell["failure"] = str(r.get("failure", ""))[:80]
        rows.setdefault(key, {})[r.get("system", "?")] = cell
    return rows


def _esdf_rows(path: Path) -> dict:
    """{(sequence or '', voxel): {system: cell}} from an ESDF sweep json."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows: dict = {}
    for r in data.get("results", []):
        key = (r.get("sequence", ""), float(r["voxel_size_m"]))
        for system in ("fvdb", "nvblox"):
            sr = r.get(system, {})
            cell: dict = {"ok": bool(sr.get("ok"))}
            if sr.get("ok"):
                cell["ms"] = sr.get("esdf_cold_ms")
                cell["gb"] = (sr.get("peak_torch_gb") if system == "fvdb"
                              else sr.get("gpu_used_gb"))
                cell["n_voxels"] = sr.get("esdf_n_voxels", sr.get("n_voxels"))
            else:
                cell["oom"] = _is_oom(sr.get("failure"))
                cell["skipped"] = "skip" in str(sr.get("failure", "")).lower()
                cell["failure"] = str(sr.get("failure", ""))[:80]
            rows.setdefault(key, {})[system] = cell
    return rows


def _render_cell(cell: dict | None) -> str:
    if cell is None:
        return "--"
    if cell.get("ok"):
        ms = cell.get("ms")
        gb = cell.get("gb")
        ms_s = f"{ms:.2f}" if isinstance(ms, (int, float)) else "?"
        gb_s = f"{gb:.2f}" if isinstance(gb, (int, float)) else "?"
        return f"{ms_s} / {gb_s}"
    if cell.get("skipped"):
        return "skip"
    if cell.get("oom"):
        return "**OOM**"
    return "FAIL"


def _render_table(title: str, rows_a: dict, rows_b: dict,
                  unit_note: str) -> str:
    keys = sorted(set(rows_a) | set(rows_b),
                  key=lambda k: (k[0], -k[1]))
    if not keys:
        return f"### {title}\n\n_(no results yet)_\n"
    lines = [f"### {title}", "",
             f"_{unit_note}; A = paper-matched (60 m, τ=3×), "
             f"B = nvblox upstream defaults (10 m, τ=4×, both systems "
             f"matched); 48 GB-device emulation._", "",
             "| seq | voxel | ours A | nvblox A | ours B | nvblox B |",
             "|-----|-------|--------|----------|--------|----------|"]
    for seq, vox in keys:
        a = rows_a.get((seq, vox), {})
        b = rows_b.get((seq, vox), {})
        lines.append(
            f"| {seq or '-'} | {vox * 100:g} cm "
            f"| {_render_cell(a.get('fvdb'))} "
            f"| {_render_cell(a.get('nvblox'))} "
            f"| {_render_cell(b.get('fvdb'))} "
            f"| {_render_cell(b.get('nvblox'))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results_sensitivity")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    d = args.results_dir

    sections = [
        _render_table(
            "Mai City 700 f LiDAR TSDF (Table 2 sensitivity)",
            _tsdf_rows(d / "mai_city_tsdf_A.json"),
            _tsdf_rows(d / "mai_city_tsdf_B.json"),
            "cells: ms/frame / peak GB"),
        _render_table(
            "KITTI full-sequence LiDAR TSDF (Table 2 sensitivity)",
            _tsdf_rows(d / "kitti_tsdf_A.json"),
            _tsdf_rows(d / "kitti_tsdf_B.json"),
            "cells: ms/frame / peak GB"),
        _render_table(
            "Mai City 700 f LiDAR ESDF (Table 5 sensitivity)",
            _esdf_rows(d / "mai_city_esdf_A.json"),
            _esdf_rows(d / "mai_city_esdf_B.json"),
            "cells: cold ESDF ms / peak GB"),
        _render_table(
            "KITTI 100 f LiDAR ESDF (Table 5 sensitivity)",
            _esdf_rows(d / "kitti_esdf_A.json"),
            _esdf_rows(d / "kitti_esdf_B.json"),
            "cells: cold ESDF ms / peak GB"),
    ]
    text = "\n".join(sections)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
