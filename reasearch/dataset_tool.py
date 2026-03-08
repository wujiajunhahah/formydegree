#!/usr/bin/env python3
"""Export and import EMG datasets (raw CSV + trained model)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_profile(root: Path, profile: str | None) -> tuple[Path, Path, str]:
    if profile:
        base = root / "profiles" / profile
        return base / "data", base / "model", profile
    return root / "data", root / "model", "default"


def _copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source missing: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def export_dataset(args: argparse.Namespace) -> None:
    root = _repo_root()
    exports_dir = root / "exports"
    exports_dir.mkdir(exist_ok=True)

    data_dir, model_dir, profile_name = _resolve_profile(root, args.profile)
    name = args.name or _timestamp_name()
    dest = exports_dir / name

    if dest.exists() and not args.overwrite:
        raise FileExistsError(f"export already exists: {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    exported = {}
    if data_dir.exists():
        _copy_tree(data_dir, dest / "data", args.overwrite)
        exported["data"] = str(data_dir)
    if model_dir.exists():
        _copy_tree(model_dir, dest / "model", args.overwrite)
        exported["model"] = str(model_dir)

    meta = {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile_name,
        "sources": exported,
    }
    with (dest / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"[dataset] Exported to {dest}")


def import_dataset(args: argparse.Namespace) -> None:
    root = _repo_root()
    source = Path(args.source)
    if not source.is_absolute():
        source = (root / "exports" / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"export not found: {source}")

    data_dir, model_dir, profile_name = _resolve_profile(root, args.profile)
    data_src = source / "data"
    model_src = source / "model"

    if data_src.exists():
        _copy_tree(data_src, data_dir, args.overwrite)
    if model_src.exists():
        _copy_tree(model_src, model_dir, args.overwrite)

    print(f"[dataset] Imported '{source.name}' into profile '{profile_name}'")


def list_exports(_: argparse.Namespace) -> None:
    root = _repo_root()
    exports_dir = root / "exports"
    if not exports_dir.exists():
        print("[dataset] No exports yet.")
        return
    items = sorted(p for p in exports_dir.iterdir() if p.is_dir())
    if not items:
        print("[dataset] No exports yet.")
        return
    for item in items:
        print(item.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export/import EMG datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Export data + model")
    export.add_argument("--profile", help="Profile name (profiles/<name>)")
    export.add_argument("--name", help="Export name (default: timestamp)")
    export.add_argument("--overwrite", action="store_true", help="Overwrite existing export")
    export.set_defaults(func=export_dataset)

    imp = sub.add_parser("import", help="Import data + model")
    imp.add_argument("source", help="Export name or path")
    imp.add_argument("--profile", help="Target profile name")
    imp.add_argument("--overwrite", action="store_true", help="Overwrite existing target")
    imp.set_defaults(func=import_dataset)

    listing = sub.add_parser("list", help="List available exports")
    listing.set_defaults(func=list_exports)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
