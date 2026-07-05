"""
Compress a DockM8 benchmark dataset for upload to Zenodo.

Produces the nested archive layout consumed by ``analysis/extract_zenodo.py``:

    <dataset>.tar.bz2  (store-only outer tar)
      └── phase2/
            ├── <target_1>.tar.bz2   (bzip2)
            │     └── <target_1>/ ... .sdf.gz + other files
            ├── <target_2>.tar.bz2
            └── ...

This keeps the EXACT archive format the extractor requires -- inner archives are
real bzip2 (opened ``r:bz2``), SDF members are real gzip (opened ``gzip.open``),
the outer tar is store-only -- while being as fast as possible on the real
workload (a few very large targets, individual SDFs up to ~9 GB).

Speed strategy (all format-preserving, all ratio-preserving so archive size is
unchanged -- important because Zenodo caps each file at 50 GB):

  * SDF files are gzipped by streaming (no intermediate copy) at level 9, using
    ``pigz`` (parallel gzip) when available so a single multi-GB SDF is spread
    across cores instead of pinned to one. Falls back to ``gzip`` transparently.
  * Per-target ``.tar.bz2`` archives use ``pbzip2``/``lbzip2`` (parallel bzip2)
    when available. This is the big win: with only a handful of large targets,
    single-threaded ``bzip2`` leaves most cores idle and the slowest target
    bounds the whole run. Parallel bzip2 emits standard bzip2 streams that
    ``tarfile.open(..., "r:bz2")`` reads without change. Falls back to ``bzip2``.
  * A global core budget is shared across concurrently-running compressors so
    the parallel encoders never oversubscribe the machine (N workers x M threads
    stays <= cores).
  * The outer archive is a store-only tar (no pointless recompression of the
    already-bzip2'd per-target archives).

Notes on choices that were deliberately NOT made, because they trade archive
size for speed and the 50 GB ceiling is the hard constraint:

  * gzip level stays at 9 (not 6). bzip2 cannot recompress DEFLATE output, so a
    lower gzip level flows ~1:1 into the final archive size.
  * bzip2 level stays at 9. The non-SDF members (e.g. multi-GB CSV) are packed
    raw and DO compress level-sensitively, so a lower bzip2 level would grow the
    archive for negligible speed.

A hard size guardrail fails the run (before the slow outer step, and again after)
if any per-target archive or the final archive exceeds ``--max-archive-gb``, so
an over-limit file is caught here instead of at upload time.

Usage:
    python -m analysis.compress_for_zenodo /path/to/DUD-E -o /out/DUD-E.tar.bz2
    python -m analysis.compress_for_zenodo /path/to/lit-pcba/PART_1 \
        -o /out/lit-pcba_1.tar.bz2 -j 32
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    tqdm = None


def human(nbytes: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TB"


def dir_size(path: Path) -> int:
    """Return total bytes of regular (non-symlink) files under path."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def detect_encoders(force_serial: bool):
    """Pick the gzip/bzip2 encoders to use, preferring parallel ones."""
    if force_serial:
        return "gzip", "bzip2"
    gzip_enc = "pigz" if shutil.which("pigz") else "gzip"
    if shutil.which("pbzip2"):
        bzip_enc = "pbzip2"
    elif shutil.which("lbzip2"):
        bzip_enc = "lbzip2"
    else:
        bzip_enc = "bzip2"
    return gzip_enc, bzip_enc


def _gzip_cmd(encoder: str, level: int, threads: int, src: str):
    if encoder == "pigz":
        return ["pigz", f"-{level}", "-p", str(max(1, threads)), "-c", src]
    return ["gzip", f"-{level}", "-c", src]


def _bzip_prog(encoder: str, level: int, threads: int) -> str:
    """Compress-program string handed to ``tar --use-compress-program``."""
    threads = max(1, threads)
    if encoder == "pbzip2":
        return f"pbzip2 -{level} -p{threads}"
    if encoder == "lbzip2":
        return f"lbzip2 -{level} -n{threads}"
    return f"bzip2 -{level}"


def stage_non_sdf(src: str, dst: str):
    """Hard-link (fast) a non-SDF file into staging, falling back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _reap(running, free, on_finish):
    """Block until >=1 running job finishes; reclaim its cores, call on_finish.

    ``running`` is a list of job dicts each with 'proc' and 'cores'. Returns the
    updated free-core count.
    """
    while True:
        still = []
        finished = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
            else:
                finished.append(job)
        if finished:
            running[:] = still
            for job in finished:
                free += job["cores"]
                on_finish(job)
            return free
        time.sleep(0.05)


def gzip_stage(gzip_jobs, jobs, level, encoder, bar):
    """Gzip every SDF into staging under a shared core budget.

    Smallest files launch first so the few multi-GB SDFs start late, when most
    of the budget is free, and (with pigz) get many threads -- collapsing the
    single-file tail. Returns (gz_in, gz_out, failures).
    """
    ordered = sorted(gzip_jobs, key=lambda j: os.path.getsize(j[0]))
    n = len(ordered)
    free = jobs
    idx = 0
    running = []
    gz_in = gz_out = 0
    failures = []
    done = 0

    def finish(job):
        nonlocal gz_in, gz_out, done
        rc = job["proc"].returncode
        job["fh"].close()
        job["errf"].seek(0)
        err = job["errf"].read().decode(errors="replace").strip()
        job["errf"].close()
        done += 1
        if bar:
            bar.update(1)
        elif done % 25 == 0 or done == n:
            print(f"  gzip {done}/{n}", flush=True)
        if rc == 0:
            gz_in += job["in_size"]
            gz_out += os.path.getsize(job["dst"])
        else:
            failures.append(f"{job['src']}: gzip exit {rc} {err}")

    while idx < n or running:
        while idx < n and free > 0:
            remaining = n - idx
            threads = max(1, free // remaining) if encoder == "pigz" else 1
            threads = min(threads, free)
            src, dst, _target = ordered[idx]
            fh = open(dst, "wb")
            errf = tempfile.TemporaryFile()
            proc = subprocess.Popen(
                _gzip_cmd(encoder, level, threads, src),
                stdout=fh, stderr=errf,
            )
            running.append({"proc": proc, "fh": fh, "errf": errf, "src": src,
                            "dst": dst, "cores": threads,
                            "in_size": os.path.getsize(src)})
            free -= threads
            idx += 1
        if running:
            free = _reap(running, free, finish)

    return gz_in, gz_out, failures


def pack_stage(targets, staging, phase2, jobs, level, encoder, bar):
    """Build each phase2/<target>.tar.bz2 under a shared core budget.

    Largest target first so the heavy packs claim threads early. Returns
    (archives dict target->size, failures).
    """
    sized = sorted(targets, key=lambda t: dir_size(staging / t), reverse=True)
    n = len(sized)
    free = jobs
    idx = 0
    running = []
    archives = {}
    failures = []

    def finish(job):
        rc = job["proc"].returncode
        job["errf"].seek(0)
        err = job["errf"].read().decode(errors="replace").strip()
        job["errf"].close()
        target = job["target"]
        if rc == 0:
            size = os.path.getsize(job["out"])
            archives[target] = size
            msg = f"  packed {target} ({human(size)})"
            (bar.write(msg) if bar else print(msg, flush=True))
        else:
            failures.append(f"{target}: pack exit {rc} {err}")

    while idx < n or running:
        while idx < n and free > 0:
            remaining = n - idx
            threads = max(1, free // remaining)
            threads = min(threads, free)
            target = sized[idx]
            out = str(phase2 / f"{target}.tar.bz2")
            prog = _bzip_prog(encoder, level, threads)
            errf = tempfile.TemporaryFile()
            proc = subprocess.Popen(
                ["tar", "--use-compress-program", prog, "-cf", out,
                 "-C", str(staging), target],
                stdout=subprocess.DEVNULL, stderr=errf,
            )
            running.append({"proc": proc, "errf": errf, "out": out,
                            "target": target, "cores": threads})
            free -= threads
            idx += 1
        if running:
            free = _reap(running, free, finish)

    return archives, failures


def compress_dataset(base_dir: Path, output_file: Path, jobs: int, level: int,
                     bzip_level: int, max_archive_gb: float, force_serial: bool,
                     keep_staging: bool, only_targets=None, extra_dirs=None,
                     work_dir=None) -> bool:
    """Build the nested Zenodo archive for one dataset; return True on success."""
    base_dir = base_dir.resolve()
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        return False

    # Resolve the set of targets to archive: subdirs of base_dir (optionally
    # restricted by --targets) plus any explicit --extra-dir targets pulled in
    # from elsewhere. Each becomes a top-level dir named after its basename.
    target_dirs = []            # (name, source Path)
    names = sorted(p.name for p in base_dir.iterdir() if p.is_dir())
    if only_targets:
        wanted = set(only_targets)
        missing = wanted - set(names)
        if missing:
            print(f"Error: --targets not found under {base_dir}: "
                  f"{', '.join(sorted(missing))}", file=sys.stderr)
            return False
        names = [n for n in names if n in wanted]
    target_dirs += [(n, base_dir / n) for n in names]

    for extra in (extra_dirs or []):
        ep = Path(extra).resolve()
        if not ep.is_dir():
            print(f"Error: --extra-dir is not a directory: {ep}", file=sys.stderr)
            return False
        target_dirs.append((ep.name, ep))

    dupes = [n for n in {t[0] for t in target_dirs}
             if [x[0] for x in target_dirs].count(n) > 1]
    if dupes:
        print(f"Error: duplicate target name(s): {', '.join(sorted(dupes))}",
              file=sys.stderr)
        return False
    if not target_dirs:
        print("Error: no target subdirectories to archive", file=sys.stderr)
        return False

    if not only_targets and not extra_dirs:
        loose = [p.name for p in base_dir.iterdir() if p.is_file()]
        if loose:
            print(f"Warning: {len(loose)} file(s) directly under {base_dir.name}/ are "
                  f"not inside a target subdirectory and will NOT be archived: "
                  f"{', '.join(loose[:5])}{' ...' if len(loose) > 5 else ''}")

    gzip_enc, bzip_enc = detect_encoders(force_serial)
    max_bytes = int(max_archive_gb * 1024 ** 3)

    original_size = sum(dir_size(tdir) for _n, tdir in target_dirs)
    print(f"Dataset: {base_dir}")
    print(f"Targets: {len(target_dirs)} ({', '.join(n for n, _ in target_dirs)})"
          f"  |  Raw size: {human(original_size)}")
    print(f"Output:  {output_file}")
    print(f"Encoders: SDF gzip -> {gzip_enc} -{level}  |  "
          f"pack bzip2 -> {bzip_enc} -{bzip_level}  |  budget {jobs} cores")
    print(f"Size guardrail: {max_archive_gb:.1f} GB per archive")

    overall_start = time.time()
    # Stage on a fast filesystem (--work-dir) so hard-links work and /tmp
    # (often small) does not overflow with tens of GB of staged data. Defaults
    # to the data's own parent.
    stage_parent = Path(work_dir).resolve() if work_dir else base_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix=".zenodo_pack_", dir=stage_parent))
    staging = work_root / "staged"
    phase2 = work_root / "phase2"
    phase2.mkdir(parents=True, exist_ok=True)
    targets = [n for n, _ in target_dirs]

    try:
        # --- Build staging tree: queue SDFs for gzip, hard-link the rest. ---
        gzip_jobs = []          # (src, dst, target) for SDF files
        n_sdf = 0
        for target, tdir in target_dirs:
            for root, _dirs, files in os.walk(tdir):
                subrel = os.path.relpath(root, tdir)
                rel = target if subrel == "." else os.path.join(target, subrel)
                dest_dir = staging / rel
                dest_dir.mkdir(parents=True, exist_ok=True)
                for name in files:
                    src = os.path.join(root, name)
                    if name.endswith(".sdf"):
                        gzip_jobs.append((src, str(dest_dir / (name + ".gz")), target))
                        n_sdf += 1
                    else:
                        stage_non_sdf(src, str(dest_dir / name))

        # --- Stage 1: gzip all SDFs. ---
        print(f"Stage 1/2: gzip {n_sdf} SDF files ({gzip_enc} -{level})...")
        bar = tqdm(total=n_sdf, desc="gzip SDF", unit="file") if tqdm else None
        gz_in, gz_out, gz_failures = gzip_stage(gzip_jobs, jobs, level, gzip_enc, bar)
        if bar:
            bar.close()

        # Never ship a format-valid but silently-incomplete archive.
        if gz_failures:
            print(f"\nAborting: {len(gz_failures)} SDF(s) failed to gzip:")
            for f in gz_failures[:10]:
                print(f"  - {f}")
            return False

        if gz_in:
            print(f"SDF gzip: {human(gz_in)} -> {human(gz_out)} "
                  f"({(1 - gz_out / gz_in) * 100:.1f}% smaller)")

        # --- Stage 2: pack each target into phase2/<target>.tar.bz2. ---
        print(f"Stage 2/2: pack {len(targets)} targets ({bzip_enc} -{bzip_level})...")
        bar = tqdm(total=len(targets), desc="pack", unit="tgt") if tqdm else None
        archives, pack_failures = pack_stage(targets, staging, phase2, jobs,
                                             bzip_level, bzip_enc, bar)
        if bar:
            bar.close()

        if pack_failures:
            print(f"\nAborting: {len(pack_failures)} target(s) failed to pack:")
            for f in pack_failures[:10]:
                print(f"  - {f}")
            return False

        staged_bz2_size = sum(archives.values())
        print(f"Per-target archives: {len(archives)}  |  {human(staged_bz2_size)}")

        # --- Size guardrail (before the slow outer step). ---
        over = {t: s for t, s in archives.items() if s > max_bytes}
        if over or staged_bz2_size > max_bytes:
            print(f"\nAborting: size guardrail exceeded ({max_archive_gb:.1f} GB).")
            if over:
                for t, s in sorted(over.items(), key=lambda kv: -kv[1]):
                    print(f"  - target {t}: {human(s)}")
            if staged_bz2_size > max_bytes:
                print(f"  - predicted final archive: {human(staged_bz2_size)} "
                      f"(sum of per-target archives)")
            print("  Split this dataset into more parts and re-run.")
            return False

        # --- Outer archive: store-only tar of phase2/ (no recompression). ---
        print("Building outer archive (store, no recompression)...")
        subprocess.run(
            ["tar", "-cf", str(output_file), "-C", str(work_root), "phase2"],
            check=True,
        )

        final_size = output_file.stat().st_size if output_file.exists() else 0
        if final_size > max_bytes:
            print(f"\nError: final archive {human(final_size)} exceeds the "
                  f"{max_archive_gb:.1f} GB guardrail.")
            return False

    finally:
        if keep_staging:
            print(f"Staging kept at: {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)

    final_size = output_file.stat().st_size if output_file.exists() else 0
    ratio = (1 - final_size / original_size) * 100 if original_size else 0
    elapsed = time.time() - overall_start
    print("-" * 60)
    print(f"Done in {timedelta(seconds=int(elapsed))}")
    print(f"Raw {human(original_size)} -> archive {human(final_size)} "
          f"({ratio:.2f}% smaller)")
    print(f"Ready for Zenodo upload: {output_file}")
    return final_size > 0


def main():
    """Parse CLI arguments and run the compression."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("base_dir", type=Path,
                        help="Dataset directory whose immediate subdirectories are targets")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output .tar.bz2 path (default: <parent>/<dataset>.tar.bz2)")
    parser.add_argument("-j", "--jobs", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Core budget shared across compressors (default: cores - 1)")
    parser.add_argument("--level", type=int, default=9, choices=range(1, 10),
                        metavar="1-9", help="gzip level for SDF files (default: 9)")
    parser.add_argument("--bzip-level", type=int, default=9, choices=range(1, 10),
                        metavar="1-9", help="bzip2 level for per-target archives (default: 9)")
    parser.add_argument("--max-archive-gb", type=float, default=48.0,
                        help="Fail if any per-target or final archive exceeds this "
                             "(default: 48, i.e. margin under Zenodo's 50 GB limit)")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated subset of base_dir subdirs to archive "
                             "(default: all)")
    parser.add_argument("--extra-dir", action="append", default=None, metavar="DIR",
                        help="Additional target directory to include (repeatable); "
                             "archived under its basename, e.g. a target moved from "
                             "another part")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Where to build the staging tree (default: base_dir's "
                             "parent). Point at a fast disk when the data lives on a "
                             "slow mount")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Force serial gzip/bzip2 (ignore pigz/pbzip2/lbzip2)")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Do not delete the temporary staging tree")
    args = parser.parse_args()

    output = args.output
    if output is None:
        name = os.path.basename(os.path.normpath(str(args.base_dir)))
        output = args.base_dir.resolve().parent / f"{name}.tar.bz2"
    output.parent.mkdir(parents=True, exist_ok=True)

    only_targets = ([t.strip() for t in args.targets.split(",") if t.strip()]
                    if args.targets else None)

    ok = compress_dataset(args.base_dir, output, args.jobs, args.level,
                          args.bzip_level, args.max_archive_gb,
                          args.no_parallel, args.keep_staging,
                          only_targets=only_targets, extra_dirs=args.extra_dir,
                          work_dir=args.work_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
