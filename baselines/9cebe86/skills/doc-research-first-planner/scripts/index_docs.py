#!/usr/bin/env python3
"""
Index design/docs files for fast lookup.

Outputs:
  ./.cache/design_index.json
  ./.cache/docs_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


DESIGN_BASES = [
    "design",
    "docs/design",
    "adr",
    "adrs",
    "architecture",
    "specs",
    "rfcs",
]
DOCS_BASES = ["docs"]
ROOT_GLOBS = ["README*", "CONTRIBUTING*"]

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".cache",
}


def _iter_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.exists():
        return []
    files: list[Path] = []
    for root, dirs, filenames in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            files.append(Path(root) / name)
    return files


def _record(path: Path, root: Path, category: str) -> dict[str, object]:
    stat = path.stat()
    rel = path.relative_to(root).as_posix()
    return {
        "path": rel,
        "category": category,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def _collect_design(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for base in DESIGN_BASES:
        path = root / base
        for file_path in _iter_files(path):
            records.append(_record(file_path, root, base))
    for pattern in ROOT_GLOBS:
        for file_path in root.glob(pattern):
            if file_path.is_file():
                records.append(_record(file_path, root, "root"))
    return records


def _collect_docs(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for base in DOCS_BASES:
        path = root / base
        for file_path in _iter_files(path):
            rel = file_path.relative_to(root).as_posix()
            if rel.startswith("docs/design/"):
                continue
            records.append(_record(file_path, root, base))
    return records


def _write_index(path: Path, records: list[dict[str, object]], root: Path) -> None:
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "root": root.as_posix(),
        "count": len(records),
        "files": sorted(records, key=lambda item: item["path"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index design/docs files.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--out-dir",
        default=".cache",
        help="Output directory for index files (default: .cache).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir

    design_records = _collect_design(root)
    docs_records = _collect_docs(root)

    _write_index(out_dir / "design_index.json", design_records, root)
    _write_index(out_dir / "docs_index.json", docs_records, root)

    print(
        f"Indexed {len(design_records)} design files and "
        f"{len(docs_records)} docs files into {out_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
