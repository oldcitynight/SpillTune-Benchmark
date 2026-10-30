#!/usr/bin/env python3
"""Verify the packaged evidence and recompute descriptive R112 summaries.

This script reads only this repository's sanitized data. It never runs a model,
training loop, inference, or scoring provider.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ARMS = ("independent_native_resident", "spilltune_streaming")
PRIVATE_PATTERNS = [
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"/(?:run/media|home|mnt|Users)/[^\s\"']+"),
    re.compile(r"(?i)\bGPU-[0-9a-f-]{20,}\b"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pct_change(new: float, old: float) -> float:
    return round((new / old - 1.0) * 100.0, 1)


def calculate(root: Path) -> dict[str, Any]:
    evaluation = read_csv(root / "data/evaluation_items.csv")
    training = read_jsonl(root / "data/training_tokens.jsonl")
    outputs = read_jsonl(root / "data/model_outputs.jsonl")
    grades = read_csv(root / "data/judgments.csv")
    updates = read_jsonl(root / "data/training_updates.jsonl")
    timings = read_json(root / "data/training_timing.json")
    samples = read_csv(root / "data/resource_samples.csv")
    checkpoints = read_csv(root / "data/checkpoint_hashes.csv")

    require(len(evaluation) == 500, f"expected 500 evaluation items, got {len(evaluation)}")
    require(len({r["item_id"] for r in evaluation}) == 500, "evaluation item IDs are not unique")
    require(len(training) == 1500, f"expected 1500 training rows, got {len(training)}")
    require(len({r["row_id"] for r in training}) == 1500, "training row IDs are not unique")
    require(all(isinstance(r["input_ids"], list) and isinstance(r["labels"], list) for r in training), "training token arrays malformed")
    require(len(outputs) == 1000, f"expected 1000 generated outputs, got {len(outputs)}")
    require(len(grades) == 5000, f"expected 5000 judgment rows, got {len(grades)}")
    require(len(updates) == 1000, f"expected 1000 update rows, got {len(updates)}")
    require(len(checkpoints) == 12, f"expected 12 checkpoint fingerprints, got {len(checkpoints)}")

    evaluation_ids = {r["item_id"] for r in evaluation}
    output_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outputs:
        require(row["arm"] in ARMS, "unexpected output arm")
        require(row["item_id"] in evaluation_ids, "output item missing from evaluation data")
        require(row["checkpoint"] == 500 and row["split"] == "final", "unexpected output evaluation slice")
        require(row["repeat_index"] == 0 and row["repeat_count"] == 1, "repeat index/count mismatch")
        key = (row["arm"], row["item_id"])
        require(key not in output_map, f"duplicate output row {key}")
        output_map[key] = row
    for arm in ARMS:
        require(sum(1 for a, _ in output_map if a == arm) == 500, f"wrong output count for {arm}")
    exact_output_pairs = 0
    stop_counts = {arm: {"eos": 0, "max_new_tokens": 0} for arm in ARMS}
    for item_id in evaluation_ids:
        left = output_map[(ARMS[0], item_id)]
        right = output_map[(ARMS[1], item_id)]
        require(left["text"] == right["text"], f"generated text differs for {item_id}")
        require(left["output_ids"] == right["output_ids"], f"generated token IDs differ for {item_id}")
        require(left["stop_reason"] == right["stop_reason"], f"stop reason differs for {item_id}")
        exact_output_pairs += 1
    for row in outputs:
        stop_counts[row["arm"]][row["stop_reason"]] = stop_counts[row["arm"]].get(row["stop_reason"], 0) + 1
    require(all(stop_counts[arm] == {"eos": 499, "max_new_tokens": 1} for arm in ARMS), "unexpected generation stop counts")

    channel_names = sorted({r["judge_channel"] for r in grades})
    require(len(channel_names) == 5, f"expected five judgment channels, got {len(channel_names)}")
    grade_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_channel: dict[str, dict[str, dict[str, int]]] = {}
    for row in grades:
        require(row["arm"] in ARMS, "unexpected judgment arm")
        require(row["item_id"] in evaluation_ids, "judgment item missing from evaluation data")
        require(row["split"] == "final" and int(row["checkpoint"]) == 500, "unexpected judgment slice")
        key = (row["arm"], row["item_id"], row["judge_channel"])
        require(key not in grade_map, f"duplicate judgment row {key}")
        grade_map[key] = row
        by_channel.setdefault(row["judge_channel"], {}).setdefault(row["arm"], {"n": 0, "correct": 0, "effective_correct": 0})
        stats = by_channel[row["judge_channel"]][row["arm"]]
        stats["n"] += 1
        stats["correct"] += int(row["raw_correct"] == "True")
        stats["effective_correct"] += int(row["effective_correct"] == "True")
        expected_effective = row["raw_correct"] == "True" and row["stop_reason"] == "eos"
        require((row["effective_correct"] == "True") == expected_effective, "effective correctness does not match raw score and stop rule")
        expected_stop = output_map[(row["arm"], row["item_id"])]["stop_reason"]
        require(row["stop_reason"] == expected_stop, "judgment stop reason differs from generated output")
    require(len(grade_map) == 5000, "judgments are not unique at arm/item/channel grain")
    for channel in channel_names:
        require(by_channel[channel][ARMS[0]]["n"] == 500 and by_channel[channel][ARMS[1]]["n"] == 500, f"wrong denominator for {channel}")
    require(all(row["effective_correct"] == row["raw_correct"] for row in grades), "effective correctness differs from raw correctness")

    score_summary = []
    for channel in channel_names:
        native = by_channel[channel][ARMS[0]]
        streaming = by_channel[channel][ARMS[1]]
        delta = 100.0 * (streaming["correct"] / 500 - native["correct"] / 500)
        paired = [0, 0, 0]
        for item_id in evaluation_ids:
            n = grade_map[(ARMS[0], item_id, channel)]["raw_correct"] == "True"
            s = grade_map[(ARMS[1], item_id, channel)]["raw_correct"] == "True"
            paired[0 if s < n else 2 if s > n else 1] += 1
        score_summary.append({
            "channel": channel,
            "native_correct": native["correct"],
            "streaming_correct": streaming["correct"],
            "n_per_arm": 500,
            "native_accuracy": native["correct"] / 500,
            "streaming_accuracy": streaming["correct"] / 500,
            "difference_percentage_points": round(delta, 1),
            "paired_streaming_lower_equal_higher": paired,
        })
    effective_equals_raw = all(row["effective_correct"] == row["raw_correct"] for row in grades)

    update_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for row in updates:
        require(row["arm"] in ARMS, "unexpected training update arm")
        update_by_arm[row["arm"]].append(row)
    for arm in ARMS:
        update_by_arm[arm].sort(key=lambda row: row["update"])
        require(len(update_by_arm[arm]) == 500, f"wrong update count for {arm}")
    native_losses = [row["loss"] for row in update_by_arm[ARMS[0]]]
    streaming_losses = [row["loss"] for row in update_by_arm[ARMS[1]]]
    require(native_losses == streaming_losses, "loss sequences differ")
    require([r["update"] for r in update_by_arm[ARMS[0]]] == [r["update"] for r in update_by_arm[ARMS[1]]], "update indices differ")

    timing_by_arm = {row["arm"]: row for row in timings}
    require(set(timing_by_arm) == set(ARMS), "timing file does not contain exactly two arms")
    for arm in ARMS:
        require(timing_by_arm[arm]["updates"] == 500 and timing_by_arm[arm]["microbatches"] == 3000, f"wrong timing cursor for {arm}")

    resource_by_arm: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    for row in samples:
        require(row["arm"] in ARMS, "unexpected resource sample arm")
        resource_by_arm[row["arm"]].append(row)
    require(len(resource_by_arm[ARMS[0]]) == 3514, "native resource sample count mismatch")
    require(len(resource_by_arm[ARMS[1]]) == 4411, "streaming resource sample count mismatch")
    for arm in ARMS:
        require(all(row["gpu_status"] == "ok" for row in resource_by_arm[arm]), f"GPU sample errors in {arm}")
        require(all(row["host_code"] == "E_CGROUP_IO_MISSING" for row in resource_by_arm[arm]), f"unexpected host resource status in {arm}")
    def peak(arm: str, field: str) -> int:
        return max(int(float(row[field])) for row in resource_by_arm[arm] if row[field] not in ("", "None", "null"))

    gpu = {arm: peak(arm, "gpu_memory_used_bytes") for arm in ARMS}
    rss = {arm: peak(arm, "host_rss_bytes") for arm in ARMS}
    cgroup = {arm: peak(arm, "cgroup_memory_bytes") for arm in ARMS}
    swap = {arm: peak(arm, "cgroup_swap_bytes") for arm in ARMS}
    read_null = sum(row["cgroup_io_read_bytes"] in ("", "None", "null") for arm in ARMS for row in resource_by_arm[arm])
    write_null = sum(row["cgroup_io_write_bytes"] in ("", "None", "null") for arm in ARMS for row in resource_by_arm[arm])
    cgroup_io_missing = sum(row["host_code"] == "E_CGROUP_IO_MISSING" for arm in ARMS for row in resource_by_arm[arm])

    checkpoint_by_arm: dict[str, dict[int, dict[str, str]]] = {arm: {} for arm in ARMS}
    for row in checkpoints:
        require(row["arm"] in ARMS, "unexpected checkpoint arm")
        checkpoint_by_arm[row["arm"]][int(row["update"])] = row
    expected_updates = {0, 100, 200, 300, 400, 500}
    for arm in ARMS:
        require(set(checkpoint_by_arm[arm]) == expected_updates, f"checkpoint update set wrong for {arm}")
    for update in expected_updates:
        n = checkpoint_by_arm[ARMS[0]][update]
        s = checkpoint_by_arm[ARMS[1]][update]
        require(n["sha256"] == s["sha256"], f"published checkpoint fingerprints differ at {update}")
        require(n["bytes"] == s["bytes"], f"published checkpoint lengths differ at {update}")
        require(n["bytes"] == "41987296", f"unexpected checkpoint state size at {update}")
    checkpoint_fingerprints_equal = all(
        checkpoint_by_arm[ARMS[0]][update]["sha256"] == checkpoint_by_arm[ARMS[1]][update]["sha256"]
        and checkpoint_by_arm[ARMS[0]][update]["bytes"] == checkpoint_by_arm[ARMS[1]][update]["bytes"]
        for update in expected_updates
    )
    require(checkpoint_fingerprints_equal, "published checkpoint fingerprints are not equal")
    require(checkpoint_by_arm[ARMS[0]][500]["sha256"] == "6864efa30bd61960a80a4446e266ca25a0c5c64cb0f05a1468168112553580b1", "final checkpoint hash differs from audited source value")

    summary = {
        "scope": "R112 formal final POST_TRAIN; sanitized release-data recomputation",
        "dataset": {"training_rows": len(training), "evaluation_items": len(evaluation), "source_revision": "openai/grade-school-math@" + "3101c7d5072418e28b9008a6636bde82a006892c"},
        "protocol": {"updates_per_arm": 500, "microbatches_per_arm": 3000, "gradient_accumulation": 6, "repeat_count": 1, "repeat_index": 0, "checkpoint_updates": sorted(expected_updates)},
        "outputs": {"rows": len(outputs), "paired_outputs": exact_output_pairs, "text_token_stop_exactly_equal": True, "stop_reason_counts_by_arm": stop_counts},
        "judgments": {"rows": len(grades), "channels": channel_names, "effective_equals_raw": effective_equals_raw, "score_summary": score_summary},
        "training_loss": {"sequence_identical": True, "count_per_arm": 500, "first": native_losses[0], "last": native_losses[-1], "mean": round(statistics.mean(native_losses), 10), "median": round(statistics.median(native_losses), 10)},
        "training_time": {
            "training_with_saves_host_seconds": {arm: timing_by_arm[arm]["training_with_saves_host_seconds"] for arm in ARMS},
            "startup_train_save_close_host_seconds": {arm: timing_by_arm[arm]["startup_train_save_close_host_seconds"] for arm in ARMS},
            "streaming_vs_native_percent_change": pct_change(timing_by_arm[ARMS[1]]["training_with_saves_host_seconds"], timing_by_arm[ARMS[0]]["training_with_saves_host_seconds"]),
        },
        "sampled_resources": {
            "sample_count": {arm: len(resource_by_arm[arm]) for arm in ARMS},
            "whole_device_smi_memory_used_peak_bytes": gpu,
            "whole_device_smi_peak_streaming_vs_native_percent_change": pct_change(gpu[ARMS[1]], gpu[ARMS[0]]),
            "worker_rss_peak_bytes": rss,
            "worker_rss_peak_streaming_vs_native_percent_change": pct_change(rss[ARMS[1]], rss[ARMS[0]]),
            "cgroup_memory_peak_bytes": cgroup,
            "cgroup_memory_peak_streaming_vs_native_percent_change": pct_change(cgroup[ARMS[1]], cgroup[ARMS[0]]),
            "cgroup_swap_peak_bytes": swap,
            "cgroup_io_read_null_samples": read_null,
            "cgroup_io_write_null_samples": write_null,
            "cgroup_io_missing_samples": cgroup_io_missing,
        },
        "checkpoint_state": {
            "published_fingerprints_equal_between_arms": checkpoint_fingerprints_equal,
            "tensor_payloads_included": False,
            "binary_equality_independently_verifiable_from_release": False,
            "sha256_by_update": {str(update): checkpoint_by_arm[ARMS[0]][update]["sha256"] for update in sorted(expected_updates)},
            "bytes_per_state": {str(update): int(checkpoint_by_arm[ARMS[0]][update]["bytes"]) for update in sorted(expected_updates)},
        },
        "interpretation_limits": [
            "The comparison is a single local run pair, not a statistical equivalence or generalization result.",
            "The comparator is independent_native_resident: packed NF4 records and QuantState remain CUDA resident, with native non-reentrant activation checkpointing. It is not a GPU offload comparator.",
            "SMI memory is a whole-device sampled peak, not process-exclusive allocation or an instantaneous guaranteed peak.",
            "The run is tagged LOCAL_OS_STARTUP_NOT_GPU_QUALIFICATION; no independent GPU qualification is claimed.",
            "The selected model repository revision was not pinned and weights are not included; exact rerun of the model computation is not guaranteed.",
            "Only 19 frozen source bodies are included from the 559-file source snapshot; the full-run environment cannot be reconstructed from the source index alone.",
            "Checkpoint equality is recomputed from published SHA-256/length fingerprints; checkpoint tensor payloads are not included for independent re-hashing.",
            "The judgment lane labels reproduce stored labels and do not independently authenticate backend identities.",
            "Cgroup I/O metrics are unavailable in the source record; no I/O throughput is claimed.",
        ],
    }
    return summary


def verify_sources(root: Path) -> int:
    source_manifest = read_json(root / "source/frozen_sources.json")
    source_index = read_json(root / "source/snapshot_file_index.json")
    require(source_index["snapshot_sha256"] == source_manifest["source_snapshot_sha256"], "source snapshot hash binding differs")
    require(source_manifest["source_snapshot_sha256"] == "5c64acd760d4c2f990f3bb403a75b1f7e05dcfe5182fa54df491cf97a6ed1f9f", "unexpected source snapshot")
    require(len(source_index["files"]) == 559, "frozen source index file count mismatch")
    count = 0
    for entry in source_manifest["included_files"]:
        indexed = source_index["files"].get(entry["path"])
        require(indexed == {"bytes": entry["bytes"], "sha256": entry["sha256"]}, f"source index mismatch for {entry['path']}")
        path = root / "source/frozen" / entry["path"]
        require(path.is_file(), f"missing frozen source file {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"size mismatch for frozen source file {entry['path']}")
        require(sha256(path) == entry["sha256"], f"hash mismatch for frozen source file {entry['path']}")
        count += 1
    return count


def verify_manifest(root: Path) -> int:
    manifest_path = root / "RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        return 0
    manifest = read_json(manifest_path)
    count = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        require(path.is_file(), f"manifest file missing: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"manifest size mismatch: {entry['path']}")
        require(sha256(path) == entry["sha256"], f"manifest SHA-256 mismatch: {entry['path']}")
        count += 1
    checksum_path = root / "SHA256SUMS"
    if checksum_path.exists():
        listed: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            listed[name] = digest
        for rel, digest in listed.items():
            require(sha256(root / rel) == digest, f"SHA256SUMS mismatch: {rel}")
    return count


def safety_scan(root: Path) -> list[str]:
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                hits.append(f"{path.relative_to(root)} matches {pattern.pattern}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-summary", action="store_true", help="write reports/summary.json from the packaged data")
    args = parser.parse_args()
    root = args.root.resolve()
    summary = calculate(root)
    source_count = verify_sources(root)
    manifest_count = verify_manifest(root)
    hits = safety_scan(root)
    require(not hits, "safety scan found possible sensitive content: " + "; ".join(hits[:10]))
    summary_path = root / "reports/summary.json"
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.write_summary:
        summary_path.write_text(encoded, encoding="utf-8")
        print(f"Wrote {summary_path.relative_to(root)}")
    elif summary_path.exists():
        require(summary_path.read_text(encoding="utf-8") == encoded, "reports/summary.json does not match recomputed sanitized data")
    print(json.dumps({
        "status": "PASS",
        "evaluation_items": summary["dataset"]["evaluation_items"],
        "training_rows": summary["dataset"]["training_rows"],
        "paired_exact_outputs": summary["outputs"]["paired_outputs"],
        "judgment_rows": summary["judgments"]["rows"],
        "frozen_source_files_verified": source_count,
        "manifest_files_verified": manifest_count,
        "sensitive_scan_hits": 0,
        "training_time_change_percent": summary["training_time"]["streaming_vs_native_percent_change"],
        "sampled_smi_peak_change_percent": summary["sampled_resources"]["whole_device_smi_peak_streaming_vs_native_percent_change"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFY_FAILED: {exc}", file=sys.stderr)
        raise
