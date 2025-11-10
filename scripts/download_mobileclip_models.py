#!/usr/bin/env python3
"""Download MobileCLIP and MobileCLIP2 checkpoints from Hugging Face.

This script relies on the `huggingface_hub` package. Make sure you have run

    pip install "huggingface_hub[cli]"

and authenticated with `huggingface-cli login` if the repos require it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

from huggingface_hub import snapshot_download

MOBILECLIP_SERIES: Mapping[str, Sequence[str]] = {
    "mobileclip": ("S0", "S1", "S2", "B", "B-LT", "S3", "L-14", "S4"),
    "mobileclip2": ("S0", "S2", "B", "S3", "L-14", "S4"),
}

REPO_PREFIX = {
    "mobileclip": "apple/MobileCLIP",
    "mobileclip2": "apple/MobileCLIP2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MobileCLIP and MobileCLIP2 model weights from Hugging Face"
    )
    parser.add_argument(
        "--series",
        choices=sorted(MOBILECLIP_SERIES.keys()),
        nargs="*",
        default=list(MOBILECLIP_SERIES.keys()),
        help="Which model series to download (default: both)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional subset of models to download. Values should match the suffixes "
            "used by apple/MobileCLIP[-2]-<model>. When omitted, all models in the "
            "selected series are downloaded."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("models_cache"),
        help="Directory to place downloaded checkpoints (default: ./models_cache)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Redownload even if the target folder already exists",
    )
    parser.add_argument(
        "--disable-symlinks",
        action="store_true",
        help="Pass local_dir_use_symlinks=False to snapshot_download",
    )
    return parser.parse_args()


def download_series(series: str, models: Sequence[str], base_dir: Path, force: bool, use_symlinks: bool) -> None:
    prefix = REPO_PREFIX[series]
    for model in models:
        repo_id = f"{prefix}-{model}"
        target_dir = base_dir / series / f"{series.upper()}-{model}"
        if target_dir.exists() and not force:
            print(f"Skip {repo_id}: already present at {target_dir}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=use_symlinks,
        )
        print(f"Finished {repo_id}")


def main() -> int:
    args = parse_args()
    base_dir: Path = args.base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        all_allowed = {name for names in MOBILECLIP_SERIES.values() for name in names}
        invalid = [name for name in args.models if name not in all_allowed]
        if invalid:
            print(
                "Unknown model names: " + ", ".join(invalid),
                file=sys.stderr,
            )
            return 1

    series_list = args.series or list(MOBILECLIP_SERIES.keys())
    for series in series_list:
        available = MOBILECLIP_SERIES[series]
        if args.models:
            models = [m for m in args.models if m in available]
            if not models:
                continue
        else:
            models = available
        download_series(
            series=series,
            models=models,
            base_dir=base_dir,
            force=args.force,
            use_symlinks=not args.disable_symlinks,
        )

    print("All requested model downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
