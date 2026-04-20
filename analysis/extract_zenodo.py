"""
Extract DockM8 benchmark data downloaded from Zenodo.

The Zenodo archives use a nested compression scheme:
  outer .tar.bz2 → phase2/ → per-target .tar.bz2 → target dirs with .sdf.gz files

This script reverses all three phases to produce the directory layout expected
by the analysis pipeline.

Usage:
    python scripts/extract_zenodo.py /path/to/downloads /path/to/output

    The downloads directory should contain:
        DEKOIS.tar.bz2, DUD-E.tar.bz2,
        lit-pcba_1.tar.bz2, lit-pcba_2.tar.bz2, lit-pcba_3.tar.bz2

    After extraction, the output directory will contain:
        DEKOIS_2.0x/  DUD-E/  lit-pcba/PART_1/  lit-pcba/PART_2/  lit-pcba/PART_3/
"""

import argparse
import gzip
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


ARCHIVE_MAP = {
    "DEKOIS.tar.bz2": "DEKOIS_2.0x",
    "DUD-E.tar.bz2": "DUD-E",
    "lit-pcba_1.tar.bz2": "lit-pcba/PART_1",
    "lit-pcba_2.tar.bz2": "lit-pcba/PART_2",
    "lit-pcba_3.tar.bz2": "lit-pcba/PART_3",
}


def decompress_sdf_gz(directory: Path) -> int:
    count = 0
    for gz_path in directory.rglob("*.sdf.gz"):
        sdf_path = gz_path.with_suffix("")
        with gzip.open(gz_path, "rb") as f_in, open(sdf_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        count += 1
    return count


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
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

        print(f"  Phase 1/3: Extracting outer archive...")
        with tarfile.open(archive_path, "r:bz2") as outer:
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
        print(f"  Phase 2/3: Extracting {len(inner_tarballs)} target archives...")

        for i, inner_tar in enumerate(inner_tarballs, 1):
            target_name = inner_tar.stem.replace(".tar", "")
            sys.stdout.write(f"\r    [{i}/{len(inner_tarballs)}] {target_name}...")
            sys.stdout.flush()

            with tarfile.open(inner_tar, "r:bz2") as tf:
                tf.extractall(final_dir)

        print(f"\r    Extracted {len(inner_tarballs)} targets.{' '*20}")

        print(f"  Phase 3/3: Decompressing .sdf.gz files...")
        count = decompress_sdf_gz(final_dir)
        print(f"    Decompressed {count} SDF files.")

    print(f"  Done: {final_dir}")


def main():
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

    for archive_path in found:
        extract_archive(archive_path, args.output_dir)

    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"Run the analysis pipeline with:")
    print(f"  python -m analysis.run_all all --base-path {args.output_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
