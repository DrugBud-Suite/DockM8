"""
Extract DockM8 benchmark data downloaded from Zenodo.

The Zenodo archives use a nested compression scheme:
  outer .tar.bz2 → phase2/ → per-target .tar.bz2 → target dirs with .sdf.gz files

The outer tar may be either bzip2-compressed (older archives) or store-only
(newer archives from compress_for_zenodo.py, which skips the pointless outer
recompression); the auto-detecting "r:*" mode below handles both. The inner
per-target archives and the .sdf.gz members are standard bzip2/gzip regardless
of which encoder produced them (compress_for_zenodo.py uses pbzip2/pigz when
available, which emit standard streams), so no special handling is needed here.

The per-target archives (Phase 2) and .sdf.gz members (Phase 3) are decompressed
in parallel to match the parallel compressor. This stays pure-Python -- no pigz
or pbzip2 required on the extracting machine -- because CPython releases the GIL
during zlib/bz2 decompression, so a thread pool genuinely uses multiple cores.

Which target lands in which lit-pcba part is not fixed (parts are just <50 GB
upload shards; e.g. a target may be packed into a different part than its raw
data suggests). The analysis pipeline finds targets by scanning, not by part,
so this does not matter for extraction.

This script reverses all three phases to produce the directory layout expected
by the analysis pipeline.

Usage:
    python scripts/extract_zenodo.py /path/to/downloads /path/to/output [-j N]

    The downloads directory should contain:
        DEKOIS.tar.bz2, DUD-E.tar.bz2,
        lit-pcba_1.tar.bz2, lit-pcba_2.tar.bz2, lit-pcba_3.tar.bz2

    After extraction, the output directory will contain:
        DEKOIS_2.0x/  DUD-E/  lit-pcba/PART_1/  lit-pcba/PART_2/  lit-pcba/PART_3/
"""

import argparse
import gzip
import os
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ARCHIVE_MAP = {
    "DEKOIS.tar.bz2": "DEKOIS_2.0x",
    "DUD-E.tar.bz2": "DUD-E",
    "lit-pcba_1.tar.bz2": "lit-pcba/PART_1",
    "lit-pcba_2.tar.bz2": "lit-pcba/PART_2",
    "lit-pcba_3.tar.bz2": "lit-pcba/PART_3",
}


def decompress_sdf_gz(directory: Path, jobs: int = 1) -> int:
    """Decompress every .sdf.gz under directory (in parallel); return the count."""
    gz_paths = list(directory.rglob("*.sdf.gz"))

    def _one(gz_path: Path) -> None:
        sdf_path = gz_path.with_suffix("")
        with gzip.open(gz_path, "rb") as f_in, open(sdf_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        gz_path.unlink()

    if gz_paths:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for _ in pool.map(_one, gz_paths):
                pass
    return len(gz_paths)


def extract_archive(archive_path: Path, dest_dir: Path, jobs: int = 1) -> None:
    """Reverse the three-phase archive for one dataset into dest_dir."""
    archive_name = archive_path.name
    dataset_subdir = ARCHIVE_MAP.get(archive_name)
    if dataset_subdir is None:
        print(f"  Skipping unknown archive: {archive_name}")
        return

    final_dir = dest_dir / dataset_subdir
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Extracting {archive_name} → {final_dir}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("  Phase 1/3: Extracting outer archive...")
        with tarfile.open(archive_path, "r:*") as outer:
            outer.extractall(tmp_path)

        phase2_dir = tmp_path / "phase2"
        if not phase2_dir.exists():
            candidates = list(tmp_path.iterdir())
            if len(candidates) == 1 and candidates[0].is_dir():
                phase2_dir = candidates[0]
            else:
                print(f"  ERROR: Expected phase2/ inside archive, found: {[c.name for c in candidates]}")
                return

        inner_tarballs = sorted(phase2_dir.glob("*.tar.bz2"))
        print(f"  Phase 2/3: Extracting {len(inner_tarballs)} target archives "
              f"({jobs} workers)...")

        # Each inner tarball unpacks a distinct <target>/ subtree, so extracting
        # them concurrently into final_dir never collides.
        def _extract_inner(inner_tar: Path) -> None:
            with tarfile.open(inner_tar, "r:bz2") as tf:
                tf.extractall(final_dir)

        if inner_tarballs:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                for _ in pool.map(_extract_inner, inner_tarballs):
                    pass
        print(f"    Extracted {len(inner_tarballs)} targets.")

        print(f"  Phase 3/3: Decompressing .sdf.gz files ({jobs} workers)...")
        count = decompress_sdf_gz(final_dir, jobs)
        print(f"    Decompressed {count} SDF files.")

    print(f"  Done: {final_dir}")


def main():
    """Parse CLI arguments and extract the requested Zenodo archives."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "download_dir",
        type=Path,
        help="Directory containing the 5 Zenodo .tar.bz2 archives",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for extracted data (use as --base-path value)",
    )
    parser.add_argument(
        "--archives",
        type=str,
        default=None,
        help="Comma-separated list of specific archives to extract (default: all)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel workers for decompression (default: CPU count)",
    )
    args = parser.parse_args()

    if not args.download_dir.is_dir():
        print(f"Error: {args.download_dir} is not a directory")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.archives:
        names = [n.strip() for n in args.archives.split(",")]
    else:
        names = list(ARCHIVE_MAP.keys())

    found = []
    missing = []
    for name in names:
        path = args.download_dir / name
        if path.exists():
            found.append(path)
        else:
            missing.append(name)

    if missing:
        print(f"Warning: Missing archives: {', '.join(missing)}")
    if not found:
        print("Error: No archives found to extract.")
        sys.exit(1)

    print(f"Archives to extract: {len(found)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel workers: {args.jobs}")

    for archive_path in found:
        extract_archive(archive_path, args.output_dir, args.jobs)

    print(f"\n{'='*60}")
    print("Extraction complete!")
    print("Run the analysis pipeline with:")
    print(f"  python -m analysis.run_all all --base-path {args.output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
