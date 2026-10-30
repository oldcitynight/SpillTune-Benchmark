#!/usr/bin/env python3
"""Create immutable file inventory and SHA256SUMS for a release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classification(rel: str) -> tuple[str, str, str]:
    if rel.startswith("source/frozen/"):
        return "frozen source code", "R112 frozen source snapshot", "GPL-3.0-or-later"
    if rel.startswith("scripts/"):
        return "release utility source", "authored from this release workflow", "GPL-3.0-or-later"
    if rel in {"COPYING", "LICENSE.md", "NOTICE", "SOURCE_ATTRIBUTION.md"} or rel.startswith("legal/") or rel.startswith("LICENSES/"):
        return "license or attribution material", "SpillTune public source tree", "see file"
    if rel == "data/licenses/GSM8K-MIT.txt":
        return "third-party license text", "openai/grade-school-math pinned revision", "MIT"
    if rel in {"data/evaluation_items.csv", "data/training_tokens.jsonl"}:
        return "sanitized derived dataset export", "R112 local data preparation; see data/source_provenance.json", "GSM8K MIT attribution"
    if rel in {"data/model_outputs.jsonl", "data/judgments.csv"}:
        return "sanitized derived user-authorized test records", "R112 POST_TRAIN results; see data/source_provenance.json", "no separate license granted"
    if rel.startswith("data/"):
        return "sanitized derived benchmark evidence", "R112 run records; see data/source_provenance.json", "see DATA_LICENSES.md"
    if rel.startswith("source/"):
        return "source provenance index", "R112 frozen source snapshot", "GPL-3.0-or-later source metadata"
    if rel.startswith("reports/"):
        return "descriptive report and visualization", "R112 sanitized data and frozen analysis report", "project repository license for report artifacts"
    if rel.startswith("protocol/"):
        return "sanitized protocol configuration", "R112 plan and run-profile fields", "project repository license"
    return "repository documentation", "release candidate", "project repository license"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "RELEASE_MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    if manifest_path.exists() or sums_path.exists():
        raise FileExistsError("Refusing to overwrite an existing release manifest or checksum file")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in {"RELEASE_MANIFEST.json", "SHA256SUMS"}:
            continue
        category, origin, license_info = classification(rel)
        files.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "classification": category,
            "origin": origin,
            "license": license_info,
        })
    manifest = {
        "schema": "spilltune.benchmark.release_manifest.v1",
        "file_count": len(files),
        "total_file_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
        "note": "The manifest omits its own digest to avoid recursion. SHA256SUMS covers this manifest and every other payload file, excluding itself.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_paths = sorted(p for p in root.rglob("*") if p.is_file() and p != sums_path)
    with sums_path.open("w", encoding="utf-8", newline="\n") as stream:
        for path in all_paths:
            stream.write(f"{digest(path)}  {path.relative_to(root).as_posix()}\n")
    print(json.dumps({"manifest_file_count": len(files), "total_file_bytes": manifest["total_file_bytes"], "checksummed_files_including_manifest": len(all_paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
