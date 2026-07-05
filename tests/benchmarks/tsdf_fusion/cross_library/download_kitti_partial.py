# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Partial (per-sequence) extractor for the KITTI Odometry Velodyne archive.

The canonical `data_odometry_velodyne.zip` is a single ~80 GB zip holding
all 22 sequences. When you only need a few sequences (the standard TSDF
benchmark triple is 00 / 02 / 05, ~23 GB) and disk is tight, downloading
the whole archive is not an option. This script extracts *only* the
requested sequences directly from S3 using HTTP Range requests, never
storing the zip itself:

  1. HEAD the archive for its total size.
  2. Fetch the tail, parse the (zip64) End Of Central Directory record,
     then fetch and parse the full central directory (a few MB).
  3. Select the entries under `sequences/<seq>/velodyne/*.bin` for the
     requested sequences.
  4. Coalesce byte-adjacent entries into large Range requests (default
     ~64 MiB per request; each .bin is ~2 MB so tiny per-file requests
     would be dominated by per-request latency), download them with a
     pool of parallel workers (KITTI's S3 bucket throttles individual
     connections hard -- see `download_kitti.py` -- so parallelism is
     load-bearing), parse local file headers from the stream, inflate
     if needed (entries are STORED or DEFLATEd), and write each file to
     `<out_root>/dataset/sequences/<seq>/velodyne/<frame>.bin`.

Resumable: files that already exist on disk with the expected
uncompressed size are skipped, so re-running after an interruption
only fetches what is missing.

Disk-safety: refuses to start -- and cooperatively stops mid-run -- if
free space on the destination filesystem would drop below
`--min-free-gb` (default 15 GB).

Typical use (from `cross_library/`):

    python3 download_kitti_partial.py \
        --out-root ../data/kitti --sequences 00 02 05

The output tree matches what `kitti_loader.load_kitti_scene` expects
for `root_dir=<out_root>` (i.e. `<out_root>/dataset/sequences/...`).
calib.txt / times.txt / poses come from the small full archives
(data_odometry_calib.zip, data_odometry_poses.zip) -- download and
unzip those separately (they are ~1-4 MB).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import os
import re
import shutil
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

KITTI_S3_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
VELODYNE_URL = f"{KITTI_S3_BASE}/data_odometry_velodyne.zip"

# Zip format signatures.
_EOCD_SIG = 0x06054B50          # End Of Central Directory
_Z64_EOCD_LOC_SIG = 0x07064B50  # Zip64 EOCD locator
_Z64_EOCD_SIG = 0x06064B50      # Zip64 EOCD record
_CDH_SIG = 0x02014B50           # Central Directory file header
_LFH_SIG = 0x04034B50           # Local File Header


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def http_head_size(url: str, timeout: float = 30.0, max_retries: int = 5) -> int:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                if cl is None:
                    raise RuntimeError(f"no Content-Length for {url}")
                return int(cl)
        except (urllib.error.URLError, OSError) as e:
            last = e
            time.sleep(min(30.0, 2.0 ** attempt))
    assert last is not None
    raise last


def http_range_bytes(url: str, start: int, end_inclusive: int,
                     timeout: float = 60.0, max_retries: int = 8) -> bytes:
    """Fetch bytes [start, end_inclusive] in one shot (for small ranges)."""
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{end_inclusive}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"expected 206, got {resp.status}")
                data = resp.read()
                want = end_inclusive - start + 1
                if len(data) != want:
                    raise RuntimeError(f"short range read: {len(data)} != {want}")
                return data
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            last = e
            time.sleep(min(30.0, 2.0 ** attempt))
    assert last is not None
    raise last


# ------------------------------------------------------------------
# Central directory parsing
# ------------------------------------------------------------------

@dataclasses.dataclass
class ZipEntry:
    name: str
    method: int          # 0 = STORED, 8 = DEFLATE
    comp_size: int
    uncomp_size: int
    local_offset: int    # absolute offset of the local file header
    name_len: int

    @property
    def span_end_bound(self) -> int:
        """Conservative upper bound on the absolute end offset (exclusive)
        of this entry's data. The local header's extra-field length can
        differ from the central directory's, so pad generously; the
        streaming parser reads the true lengths from the local header."""
        return (self.local_offset + 30 + self.name_len + 512
                + self.comp_size + 32)


def _find_eocd(tail: bytes, tail_abs_start: int) -> Tuple[int, int, int]:
    """Locate the EOCD (and zip64 records if present) in `tail`.

    Returns (cd_offset, cd_size, n_entries). `tail_abs_start` is the
    absolute file offset of tail[0].
    """
    idx = tail.rfind(struct.pack("<I", _EOCD_SIG))
    if idx < 0:
        raise RuntimeError("EOCD signature not found in archive tail; "
                           "fetch a larger tail")
    (sig, _disk, _cd_disk, _n_disk, n_total,
     cd_size, cd_offset, _comment_len) = struct.unpack(
        "<IHHHHIIH", tail[idx:idx + 22])
    assert sig == _EOCD_SIG

    need64 = (n_total == 0xFFFF or cd_size == 0xFFFFFFFF
              or cd_offset == 0xFFFFFFFF)

    # Zip64 EOCD locator sits immediately before the EOCD.
    loc_idx = idx - 20
    if loc_idx >= 0:
        loc_sig, = struct.unpack("<I", tail[loc_idx:loc_idx + 4])
        if loc_sig == _Z64_EOCD_LOC_SIG:
            (_sig, _disk_z64, z64_eocd_abs, _n_disks) = struct.unpack(
                "<IIQI", tail[loc_idx:loc_idx + 20])
            z64_rel = z64_eocd_abs - tail_abs_start
            if z64_rel < 0:
                raise RuntimeError(
                    f"zip64 EOCD at absolute {z64_eocd_abs} lies before the "
                    f"fetched tail (starts {tail_abs_start}); fetch more tail")
            (z_sig, _z_size, _vm, _vn, _dsk, _cddsk,
             _n_disk64, n_total64, cd_size64, cd_offset64) = struct.unpack(
                "<IQHHIIQQQQ", tail[z64_rel:z64_rel + 56])
            if z_sig != _Z64_EOCD_SIG:
                raise RuntimeError("bad zip64 EOCD signature")
            return cd_offset64, cd_size64, n_total64

    if need64:
        raise RuntimeError("EOCD needs zip64 but no zip64 locator found")
    return cd_offset, cd_size, n_total


def fetch_central_directory(url: str, total_size: int,
                            tail_bytes: int = 4 << 20) -> Tuple[bytes, int]:
    """Fetch the archive tail, locate the EOCD, then return
    (central_directory_bytes, n_entries)."""
    tail_start = max(0, total_size - tail_bytes)
    sys.stderr.write(f"[cd] fetching {total_size - tail_start:,} tail bytes\n")
    tail = http_range_bytes(url, tail_start, total_size - 1)
    cd_offset, cd_size, n_entries = _find_eocd(tail, tail_start)
    sys.stderr.write(f"[cd] central directory: offset={cd_offset:,} "
                     f"size={cd_size:,} entries={n_entries:,}\n")
    if cd_offset >= tail_start:
        cd = tail[cd_offset - tail_start: cd_offset - tail_start + cd_size]
    else:
        sys.stderr.write(f"[cd] fetching full central directory "
                         f"({cd_size / 1e6:.1f} MB)\n")
        cd = http_range_bytes(url, cd_offset, cd_offset + cd_size - 1)
    if len(cd) != cd_size:
        raise RuntimeError(f"central directory short read: {len(cd)} != {cd_size}")
    return cd, n_entries


def parse_central_directory(cd: bytes, n_entries: int) -> List[ZipEntry]:
    entries: List[ZipEntry] = []
    pos = 0
    for _ in range(n_entries):
        (sig, _vm, _vn, _flags, method, _mt, _md, _crc,
         comp_size, uncomp_size, name_len, extra_len, comment_len,
         _disk, _iattr, _eattr, local_offset) = struct.unpack_from(
            "<IHHHHHHIIIHHHHHII", cd, pos)
        if sig != _CDH_SIG:
            raise RuntimeError(f"bad central directory signature at {pos}")
        name = cd[pos + 46: pos + 46 + name_len].decode("utf-8")
        extra = cd[pos + 46 + name_len: pos + 46 + name_len + extra_len]

        # Zip64 extra field (id 0x0001): 8-byte values appear in a fixed
        # order, but ONLY for the fields that are 0xFFFFFFFF in the
        # fixed-size header.
        if 0xFFFFFFFF in (comp_size, uncomp_size, local_offset):
            e = 0
            while e + 4 <= len(extra):
                fid, flen = struct.unpack_from("<HH", extra, e)
                if fid == 0x0001:
                    body = extra[e + 4: e + 4 + flen]
                    b = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size, = struct.unpack_from("<Q", body, b); b += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size, = struct.unpack_from("<Q", body, b); b += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset, = struct.unpack_from("<Q", body, b); b += 8
                    break
                e += 4 + flen
            else:
                raise RuntimeError(f"entry {name!r} needs zip64 extra "
                                   f"field but none found")

        entries.append(ZipEntry(name=name, method=method,
                                comp_size=comp_size, uncomp_size=uncomp_size,
                                local_offset=local_offset, name_len=name_len))
        pos += 46 + name_len + extra_len + comment_len
    return entries


# ------------------------------------------------------------------
# Entry selection / output mapping
# ------------------------------------------------------------------

def select_velodyne_entries(entries: List[ZipEntry],
                            sequences: List[str]) -> List[ZipEntry]:
    pat = re.compile(
        r"(?:^|/)sequences/(" + "|".join(re.escape(s) for s in sequences)
        + r")/velodyne/\d+\.bin$")
    picked = [e for e in entries if pat.search(e.name)]
    picked.sort(key=lambda e: e.local_offset)
    return picked


def entry_out_path(out_root: Path, name: str) -> Path:
    """Map a zip-internal name to the on-disk path the loader expects
    (`<out_root>/dataset/...`). The official archives use a `dataset/`
    prefix already; tolerate its absence."""
    rel = name if name.startswith("dataset/") else f"dataset/{name}"
    return out_root / rel


# ------------------------------------------------------------------
# Run planning (coalesce adjacent entries into large Range requests)
# ------------------------------------------------------------------

@dataclasses.dataclass
class Run:
    entries: List[ZipEntry]

    @property
    def start(self) -> int:
        return self.entries[0].local_offset

    @property
    def end_bound(self) -> int:
        return self.entries[-1].span_end_bound


def plan_runs(pending: List[ZipEntry], total_size: int,
              target_run_bytes: int, max_gap_bytes: int = 8 << 20) -> List[Run]:
    """Group offset-sorted pending entries into runs of ~target_run_bytes.
    A new run is started when the next entry is far from the previous
    one (so we never download big irrelevant gaps, e.g. skipped
    sequences) or when the run would exceed the target size."""
    runs: List[Run] = []
    cur: List[ZipEntry] = []
    for e in pending:
        if cur:
            gap = e.local_offset - cur[-1].span_end_bound
            run_len = e.span_end_bound - cur[0].local_offset
            if gap > max_gap_bytes or run_len > target_run_bytes:
                runs.append(Run(cur))
                cur = []
        cur.append(e)
    if cur:
        runs.append(Run(cur))
    for r in runs:
        if r.end_bound > total_size:
            # span_end_bound is a padded bound; clamp to the file size.
            pass
    return runs


# ------------------------------------------------------------------
# Streaming run download
# ------------------------------------------------------------------

class Progress:
    def __init__(self, n_files_total: int, bytes_total: int):
        self.lock = threading.Lock()
        self.files_done = 0
        self.bytes_done = 0
        self.n_files_total = n_files_total
        self.bytes_total = bytes_total
        self.stop_event = threading.Event()
        self.first_error: Optional[BaseException] = None

    def file_done(self, nbytes: int) -> None:
        with self.lock:
            self.files_done += 1
            self.bytes_done += nbytes

    def record_error(self, exc: BaseException) -> None:
        with self.lock:
            if self.first_error is None:
                self.first_error = exc
        self.stop_event.set()


class _StreamReader:
    """Sequential reader over an HTTP response with absolute-position
    tracking and skip support."""

    def __init__(self, resp, abs_start: int):
        self.resp = resp
        self.pos = abs_start

    def read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            c = self.resp.read(min(remaining, 1 << 20))
            if not c:
                raise EOFError(f"stream ended {remaining} bytes early")
            chunks.append(c)
            remaining -= len(c)
        self.pos += n
        return b"".join(chunks)

    def skip_to(self, abs_offset: int) -> None:
        if abs_offset < self.pos:
            raise RuntimeError(f"cannot seek backwards ({abs_offset} < {self.pos})")
        remaining = abs_offset - self.pos
        while remaining > 0:
            c = self.resp.read(min(remaining, 1 << 20))
            if not c:
                raise EOFError("stream ended during skip")
            remaining -= len(c)
        self.pos = abs_offset

    def stream_into(self, n: int, sink) -> None:
        remaining = n
        while remaining > 0:
            c = self.resp.read(min(remaining, 1 << 20))
            if not c:
                raise EOFError(f"stream ended {remaining} bytes early")
            sink(c)
            remaining -= len(c)
        self.pos += n


def _extract_one(reader: _StreamReader, entry: ZipEntry, out_path: Path) -> None:
    """Parse the local header at the current stream position (must equal
    entry.local_offset) and write the decompressed payload to out_path."""
    hdr = reader.read_exact(30)
    (sig, _ver, _flags, method, _mt, _md, _crc,
     _csize, _usize, name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", hdr)
    if sig != _LFH_SIG:
        raise RuntimeError(f"{entry.name}: bad local header signature "
                           f"0x{sig:08x} at offset {entry.local_offset}")
    reader.read_exact(name_len + extra_len)

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        if method == 0:                      # STORED
            reader.stream_into(entry.comp_size, f.write)
        elif method == 8:                    # DEFLATE (raw)
            d = zlib.decompressobj(-15)
            reader.stream_into(entry.comp_size,
                               lambda c: f.write(d.decompress(c)))
            f.write(d.flush())
        else:
            raise RuntimeError(f"{entry.name}: unsupported compression "
                               f"method {method}")
    got = tmp_path.stat().st_size
    if got != entry.uncomp_size:
        tmp_path.unlink()
        raise RuntimeError(f"{entry.name}: wrote {got} bytes, "
                           f"expected {entry.uncomp_size}")
    os.replace(tmp_path, out_path)


def _download_run(run: Run, url: str, out_root: Path, total_size: int,
                  progress: Progress, min_free_bytes: int,
                  max_retries: int = 10, timeout: float = 60.0) -> None:
    """Fetch one coalesced Range and extract every entry in it.
    Retries restart the run but skip entries already completed."""
    retries = 0
    while not progress.stop_event.is_set():
        pending = [e for e in run.entries
                   if not _is_done(entry_out_path(out_root, e.name), e)]
        if not pending:
            return
        if shutil.disk_usage(out_root).free < min_free_bytes:
            progress.record_error(RuntimeError(
                f"free disk space dropped below the safety floor "
                f"({min_free_bytes / 1e9:.0f} GB); stopping"))
            return
        start = pending[0].local_offset
        end = min(pending[-1].span_end_bound, total_size) - 1
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"expected 206, got {resp.status}")
                reader = _StreamReader(resp, start)
                for e in pending:
                    if progress.stop_event.is_set():
                        return
                    out_path = entry_out_path(out_root, e.name)
                    if _is_done(out_path, e):
                        continue
                    reader.skip_to(e.local_offset)
                    _extract_one(reader, e, out_path)
                    progress.file_done(e.uncomp_size)
            return
        except (urllib.error.URLError, OSError, EOFError, RuntimeError) as exc:
            retries += 1
            if retries > max_retries:
                progress.record_error(RuntimeError(
                    f"run at offset {run.start}: exhausted {max_retries} "
                    f"retries (last: {type(exc).__name__}: {exc})"))
                return
            time.sleep(min(60.0, 2.0 ** min(retries, 6)))


def _is_done(out_path: Path, entry: ZipEntry) -> bool:
    try:
        return out_path.stat().st_size == entry.uncomp_size
    except OSError:
        return False


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[2],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "data" / "kitti",
                    help="Output root; files land in <out-root>/dataset/... "
                         "(pass this same path to kitti_loader as root_dir).")
    ap.add_argument("--sequences", nargs="+", default=["00", "02", "05"],
                    help="KITTI odometry sequence numbers to extract.")
    ap.add_argument("--url", default=VELODYNE_URL)
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallel Range-request streams. KITTI's S3 "
                         "bucket throttles per-connection bandwidth "
                         "(~0.2 MB/s), so parallelism is essential.")
    ap.add_argument("--run-mb", type=int, default=64,
                    help="Target size of each coalesced Range request in MiB.")
    ap.add_argument("--min-free-gb", type=float, default=15.0,
                    help="Abort if destination free space would drop below this.")
    ap.add_argument("--progress-every", type=float, default=15.0,
                    help="Seconds between progress lines.")
    args = ap.parse_args()

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    min_free_bytes = int(args.min_free_gb * 1e9)

    total_size = http_head_size(args.url)
    sys.stderr.write(f"[head] {args.url}\n  total {total_size:,} bytes "
                     f"({total_size / 1e9:.2f} GB)\n")

    cd, n_entries = fetch_central_directory(args.url, total_size)
    entries = parse_central_directory(cd, n_entries)
    picked = select_velodyne_entries(entries, args.sequences)
    if not picked:
        sample = [e.name for e in entries[:5]]
        sys.stderr.write(f"[fatal] no entries matched sequences "
                         f"{args.sequences}; sample names: {sample}\n")
        return 2

    per_seq: dict[str, int] = {}
    for e in picked:
        m = re.search(r"sequences/(\d+)/velodyne", e.name)
        per_seq[m.group(1)] = per_seq.get(m.group(1), 0) + 1
    want_bytes = sum(e.uncomp_size for e in picked)
    sys.stderr.write(f"[plan] {len(picked):,} files selected "
                     f"({want_bytes / 1e9:.2f} GB uncompressed): "
                     + ", ".join(f"seq {s}: {n}" for s, n in sorted(per_seq.items()))
                     + "\n")

    free = shutil.disk_usage(out_root).free
    already = [e for e in picked if _is_done(entry_out_path(out_root, e.name), e)]
    need_bytes = sum(e.uncomp_size for e in picked) - sum(
        e.uncomp_size for e in already)
    sys.stderr.write(f"[disk] free {free / 1e9:.1f} GB, still need "
                     f"{need_bytes / 1e9:.2f} GB, floor "
                     f"{args.min_free_gb:.0f} GB\n")
    if free - need_bytes < min_free_bytes:
        sys.stderr.write("[fatal] not enough disk space to proceed safely\n")
        return 2

    pending = [e for e in picked
               if not _is_done(entry_out_path(out_root, e.name), e)]
    if already:
        sys.stderr.write(f"[resume] {len(already):,} files already on disk; "
                         f"{len(pending):,} to fetch\n")
    if not pending:
        sys.stderr.write("[done] nothing to do\n")
        return 0

    runs = plan_runs(pending, total_size, target_run_bytes=args.run_mb << 20)
    sys.stderr.write(f"[plan] {len(runs)} coalesced range requests "
                     f"(~{args.run_mb} MiB each), {args.workers} workers\n")

    progress = Progress(n_files_total=len(picked), bytes_total=want_bytes)
    progress.files_done = len(already)
    progress.bytes_done = sum(e.uncomp_size for e in already)

    run_queue: List[Run] = list(runs)
    queue_lock = threading.Lock()

    def worker() -> None:
        while not progress.stop_event.is_set():
            with queue_lock:
                if not run_queue:
                    return
                run = run_queue.pop(0)
            _download_run(run, args.url, out_root, total_size,
                          progress, min_free_bytes)

    threads = [threading.Thread(target=worker, name=f"w{i}", daemon=True)
               for i in range(args.workers)]
    t0 = time.time()
    b0 = progress.bytes_done
    for t in threads:
        t.start()

    last_files = progress.files_done
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(args.progress_every)
            with progress.lock:
                fd, bd = progress.files_done, progress.bytes_done
            el = time.time() - t0
            mbps = (bd - b0) / el / 1e6 if el > 0 else 0.0
            eta_s = ((progress.bytes_total - bd) / (mbps * 1e6)
                     if mbps > 1e-3 else float("inf"))
            free_gb = shutil.disk_usage(out_root).free / 1e9
            if fd != last_files or True:
                sys.stderr.write(
                    f"  {fd:>6}/{progress.n_files_total} files  "
                    f"{bd / 1e9:7.2f}/{progress.bytes_total / 1e9:.2f} GB  "
                    f"{mbps:6.1f} MB/s  ETA {eta_s / 60:6.1f} min  "
                    f"free {free_gb:.1f} GB\n")
                sys.stderr.flush()
                last_files = fd
            if free_gb * 1e9 < min_free_bytes:
                progress.record_error(RuntimeError(
                    "free space fell below the safety floor"))
    except KeyboardInterrupt:
        progress.stop_event.set()
        sys.stderr.write("\n[sigint] stopping; completed files are kept, "
                         "re-run to resume\n")
        for t in threads:
            t.join(timeout=10.0)
        return 130

    for t in threads:
        t.join()

    if progress.first_error is not None:
        sys.stderr.write(f"[fatal] {progress.first_error}\n")
        return 1

    missing = [e for e in picked
               if not _is_done(entry_out_path(out_root, e.name), e)]
    if missing:
        sys.stderr.write(f"[fatal] {len(missing)} files still missing after "
                         f"all runs completed; re-run to retry\n")
        return 1

    el = time.time() - t0
    sys.stderr.write(f"[done] {len(picked):,} files "
                     f"({progress.bytes_total / 1e9:.2f} GB) in "
                     f"{el / 60:.1f} min "
                     f"({(progress.bytes_done - b0) / el / 1e6:.1f} MB/s)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
