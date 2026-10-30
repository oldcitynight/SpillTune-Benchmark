#!/usr/bin/env python3
"""Extract the public-safe R112 evidence subset from local source records.

This script reads existing records only. It never trains, generates, scores, or
changes the input databases. It deliberately omits SQLite containers and
machine/session identifiers from the public files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SHA256 = {
    "training.sqlite": "251b6101daafd2dbeb341fe1420cd46c36e9a4a670635d20beb71a0e46db1bf9",
    "scoring.sqlite": "00e5ed45d9d4a91591c3f40fda69f97ae4948f9b92cdde90a8ca12121a8daaa9",
    "results.sqlite": "7668b4b4172fb7a15286c022cbd82e9885ffd9a349bcbc639e58854246852ceb",
    "SOURCE_SNAPSHOT.json": "5c64acd760d4c2f990f3bb403a75b1f7e05dcfe5182fa54df491cf97a6ed1f9f",
}
SOURCE_SNAPSHOT_SHA256 = EXPECTED_SHA256["SOURCE_SNAPSHOT.json"]
TRAINING_DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
R112_SOURCE_FILES = [
    "src/ram_vram_finetune/experiments/nf4_stage_worker.py",
    "src/ram_vram_finetune/experiments/nf4_full_train.py",
    "src/ram_vram_finetune/experiments/nf4_execution.py",
    "src/ram_vram_finetune/experiments/nf4_train_data.py",
    "src/ram_vram_finetune/experiments/nf4_inference_stage.py",
    "src/ram_vram_finetune/experiments/nf4_inference_runtime.py",
    "src/ram_vram_finetune/experiments/nf4_scoring_stage.py",
    "src/ram_vram_finetune/experiments/nf4_final_analysis.py",
    "src/ram_vram_finetune/adapters/qwen35/nf4_native.py",
    "src/ram_vram_finetune/adapters/qwen35/nf4_cuda.py",
    "src/ram_vram_finetune/training/nf4_native_memory.py",
    "src/ram_vram_finetune/training/nf4_cuda_recompute.py",
    "src/ram_vram_finetune/training/nf4_cuda_pool.py",
    "src/ram_vram_finetune/training/nf4_preflight.py",
    "src/ram_vram_finetune/training/nf4_streaming.py",
    "src/ram_vram_finetune/quantization/nf4_backend_cuda.py",
    "src/ram_vram_finetune/quantization/native_terminal.py",
    "src/ram_vram_finetune/checkpoint_io/nf4_cuda_training.py",
    "src/ram_vram_finetune/checkpoint_io/nf4_training.py",
]

PRIVATE_TEXT_PATTERNS = [
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"/(?:run/media|home|mnt|Users)/[^\s\"']+"),
    re.compile(r"(?i)\bGPU-[0-9a-f-]{20,}\b"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path.name}")


def db_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def parse_json_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


def check_public_text(value: str, label: str) -> None:
    for pattern in PRIVATE_TEXT_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"Potential private identifier found in {label}; no output was published")


def natural_id(value: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([^0-9]*)([0-9]+)", str(value))
    if not match:
        return str(value), -1, str(value)
    return match.group(1), int(match.group(2)), str(value)


def emit_public_data(run_root: Path, output_root: Path) -> None:
    training_db = run_root / "round" / "training.sqlite"
    scoring_db = run_root / "round" / "scoring.sqlite"
    results_db = run_root / "round" / "POST_TRAIN" / "results.sqlite"
    for path in (training_db, scoring_db, results_db):
        if sha256(path) != EXPECTED_SHA256[path.name]:
            raise ValueError(f"Source hash mismatch for {path.name}")

    evaluation_rows: list[dict[str, Any]] = []
    con = db_readonly(scoring_db)
    try:
        raw = con.execute("SELECT id, question, standard_answer FROM scoring").fetchall()
    finally:
        con.close()
    for item_id, question, answer in sorted(raw, key=lambda row: natural_id(row[0])):
        check_public_text(question, "evaluation question")
        check_public_text(answer, "evaluation answer")
        evaluation_rows.append({"item_id": item_id, "question": question, "standard_answer": answer})
    write_csv(output_root / "data" / "evaluation_items.csv", ["item_id", "question", "standard_answer"], evaluation_rows)

    training_rows: list[dict[str, Any]] = []
    con = db_readonly(training_db)
    try:
        raw = con.execute("SELECT id, input_ids, labels FROM training").fetchall()
    finally:
        con.close()
    for row_id, input_ids, labels in sorted(raw, key=lambda row: natural_id(row[0])):
        training_rows.append({"row_id": row_id, "input_ids": parse_json_cell(input_ids), "labels": parse_json_cell(labels)})
    dump_jsonl(output_root / "data" / "training_tokens.jsonl", training_rows)

    output_rows: list[dict[str, Any]] = []
    grade_rows: list[dict[str, Any]] = []
    con = db_readonly(results_db)
    try:
        for payload, in con.execute("SELECT payload FROM results ORDER BY slot"):
            item = json.loads(payload)
            observation, result = item["observation"], item["result"]
            text = result["text"]
            check_public_text(text, "generated output")
            output_rows.append({
                "item_id": observation["question_id"],
                "arm": observation["arm"],
                "split": observation["split"],
                "checkpoint": observation["checkpoint"],
                "repeat_index": observation["repeat_index"],
                "repeat_count": 1,
                "stop_reason": result["stop_reason"],
                "output_ids": result["output_ids"],
                "text": text,
            })
        for payload, in con.execute("SELECT payload FROM judge_grades ORDER BY grade_index"):
            item = json.loads(payload)
            observation = item["observation"]
            grade_rows.append({
                "item_id": observation["question_id"],
                "arm": observation["arm"],
                "split": observation["split"],
                "checkpoint": observation["checkpoint"],
                "repeat_index": observation["repeat_index"],
                "judge_channel": item["judge_id"],
                "raw_correct": item["raw_correct"],
                "effective_correct": item["effective_correct"],
                "stop_reason": item["stop_reason"],
            })
    finally:
        con.close()
    output_rows.sort(key=lambda r: (r["arm"], natural_id(r["item_id"])))
    grade_rows.sort(key=lambda r: (r["arm"], natural_id(r["item_id"]), r["judge_channel"]))
    dump_jsonl(output_root / "data" / "model_outputs.jsonl", output_rows)
    write_csv(output_root / "data" / "judgments.csv", list(grade_rows[0]), grade_rows)

    update_rows: list[dict[str, Any]] = []
    for arm, dirname in (("independent_native_resident", "native-control"), ("spilltune_streaming", "streaming-control")):
        source = run_root / dirname / "work" / "UPDATES.jsonl"
        for line in source.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            update_rows.append({
                "arm": arm,
                "update": item["update"],
                "host_interval_seconds": item["host_interval_seconds"],
                "input_tokens": item["input_tokens"],
                "loss": item["loss"],
                "microbatches": item["microbatches"],
                "supervised_tokens": item["supervised_tokens"],
                "timing_scope": item["timing_scope"],
            })
    dump_jsonl(output_root / "data" / "training_updates.jsonl", update_rows)

    sample_rows: list[dict[str, Any]] = []
    for arm, dirname in (("independent_native_resident", "native-control"), ("spilltune_streaming", "streaming-control")):
        source = run_root / dirname / "hardware" / "SAMPLES.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        origin = records[0]["start_monotonic"]
        for item in records:
            gpu, host = item.get("gpu", {}), item.get("host", {})
            gv, hv = gpu.get("values") or {}, host.get("values") or {}
            sample_rows.append({
                "arm": arm,
                "sample_index": item["index"],
                "elapsed_start_seconds": item["start_monotonic"] - origin,
                "elapsed_end_seconds": item["end_monotonic"] - origin,
                "gpu_status": gpu.get("status"),
                "gpu_code": gpu.get("code"),
                "gpu_command_seconds": gpu.get("host_command_seconds"),
                "gpu_memory_used_bytes": gv.get("memory_used_bytes"),
                "gpu_memory_total_bytes": gv.get("memory_total_bytes"),
                "gpu_utilization_percent": gv.get("utilization_percent"),
                "gpu_power_watts": gv.get("power_watts"),
                "gpu_sm_clock_mhz": gv.get("sm_clock_mhz"),
                "gpu_memory_clock_mhz": gv.get("memory_clock_mhz"),
                "gpu_temperature_celsius": gv.get("temperature_celsius"),
                "host_status": host.get("status"),
                "host_code": host.get("code"),
                "host_rss_bytes": hv.get("rss_bytes"),
                "cgroup_memory_bytes": hv.get("cgroup_memory_bytes"),
                "cgroup_swap_bytes": hv.get("cgroup_swap_bytes"),
                "host_available_bytes": hv.get("host_available_bytes"),
                "process_read_bytes": hv.get("read_bytes"),
                "process_write_bytes": hv.get("write_bytes"),
                "cgroup_io_read_bytes": hv.get("cgroup_io_read_bytes"),
                "cgroup_io_write_bytes": hv.get("cgroup_io_write_bytes"),
                "cpu_system_ticks": hv.get("cpu_system_ticks"),
                "cpu_user_ticks": hv.get("cpu_user_ticks"),
            })
    write_csv(output_root / "data" / "resource_samples.csv", list(sample_rows[0]), sample_rows)

    timing_records = []
    for arm, dirname in (("independent_native_resident", "native-control"), ("spilltune_streaming", "streaming-control")):
        handoff = json_load(run_root / dirname / "work" / "TRAIN_HANDOFF.json")
        timing_records.append({
            "arm": arm,
            "training_with_saves_host_seconds": handoff["training_with_saves_host_seconds"],
            "startup_train_save_close_host_seconds": handoff["startup_train_save_close_host_seconds"],
            "checkpoint_save_host_seconds": handoff["save_host_seconds"],
            "updates": handoff["cursor"]["updates"],
            "microbatches": handoff["cursor"]["microbatches"],
            "execution_domain": handoff["execution_domain"],
            "execution_permission": handoff["execution_permission"],
            "status": handoff["status"],
            "run_profile_sha256": handoff["run_profile_sha256"],
            "updates_sha256": handoff["updates_sha256"],
        })
    dump_json(output_root / "data" / "training_timing.json", timing_records)

    checkpoint_rows: list[dict[str, Any]] = []
    for arm, dirname in (("independent_native_resident", "native-control"), ("spilltune_streaming", "streaming-control")):
        for update in (0, 100, 200, 300, 400, 500):
            path = run_root / dirname / "work" / "checkpoints" / f"update-{update:06d}" / "state.safetensors"
            checkpoint_rows.append({"arm": arm, "update": update, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(output_root / "data" / "checkpoint_hashes.csv", ["arm", "update", "bytes", "sha256"], checkpoint_rows)

    source_hashes = []
    named_sources = [
        ("training.sqlite", run_root / "round" / "training.sqlite"),
        ("scoring.sqlite", run_root / "round" / "scoring.sqlite"),
        ("POST_TRAIN/results.sqlite", run_root / "round" / "POST_TRAIN" / "results.sqlite"),
        ("TRAINING_MANIFEST.json", run_root / "round" / "TRAINING_MANIFEST.json"),
        ("POST_TRAIN/MANIFEST.json", run_root / "round" / "POST_TRAIN" / "MANIFEST.json"),
        ("NATIVE_PLAN.json", run_root / "NATIVE_PLAN.json"),
        ("STREAMING_PLAN.json", run_root / "STREAMING_PLAN.json"),
        ("PAIR.json", run_root / "PAIR.json"),
        ("SHARED_BINDING.json", run_root / "SHARED_BINDING.json"),
        ("native UPDATES.jsonl", run_root / "native-control" / "work" / "UPDATES.jsonl"),
        ("streaming UPDATES.jsonl", run_root / "streaming-control" / "work" / "UPDATES.jsonl"),
        ("native SAMPLES.jsonl", run_root / "native-control" / "hardware" / "SAMPLES.jsonl"),
        ("streaming SAMPLES.jsonl", run_root / "streaming-control" / "hardware" / "SAMPLES.jsonl"),
        ("native TRAIN_HANDOFF.json", run_root / "native-control" / "work" / "TRAIN_HANDOFF.json"),
        ("streaming TRAIN_HANDOFF.json", run_root / "streaming-control" / "work" / "TRAIN_HANDOFF.json"),
    ]
    for label, path in named_sources:
        source_hashes.append({"source": label, "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump_json(output_root / "data" / "source_provenance.json", {
        "classification": "sanitized-derived release metadata; original local files not included",
        "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
        "dataset_repository": "https://github.com/openai/grade-school-math",
        "dataset_revision": TRAINING_DATASET_REVISION,
        "original_sources": source_hashes,
        "transformations": [
            "SQLite containers were not copied. Evaluation question/answer columns and training token columns were extracted into stable UTF-8 CSV/JSONL; their field values are unchanged.",
            "Inference rows retain item, arm, split, checkpoint, repeat index, stop reason, generated token IDs, and exact output text. Run identity objects, job IDs, and hashes were dropped.",
            "Judgment rows retain channel label, item, arm, raw/effective correctness, and stop reason. Actor, transport, session, request, and backend identity fields were dropped.",
            "Training update UTC timestamps and absolute monotonic clocks were dropped; measured interval, loss, token counts, microbatch count, and update number remain.",
            "Hardware samples retain numeric measurements and relative monotonic offsets from each arm's first sample. Absolute UTC, GPU UUID, host, process, cgroup path, binding, and collector identity fields were dropped. Process I/O counters are distinct from cgroup I/O counters; the latter are absent where the source sampler reports E_CGROUP_IO_MISSING.",
            "Checkpoint state bytes are not included; only the byte lengths and SHA-256 digests of the six saved states per arm are included.",
        ],
        "omitted": [
            "original SQLite database files and private manifests",
            "TUI frame payloads, actor/session records, provider request/session identifiers, and raw transport payloads",
            "absolute machine paths, PIDs, cgroup paths, GPU UUIDs, local caches, and environment secrets",
            "model weights, adapters, and checkpoint tensors",
        ],
    })


def emit_source_bundle(snapshot_path: Path, source_root_override: Path | None, output_root: Path) -> None:
    if sha256(snapshot_path) != SOURCE_SNAPSHOT_SHA256:
        raise ValueError("Frozen source snapshot hash mismatch")
    snapshot = json_load(snapshot_path)
    source_root = source_root_override or Path(snapshot["worktree"])
    selected: list[dict[str, Any]] = []
    target_root = output_root / "source" / "frozen"
    for rel in R112_SOURCE_FILES:
        expected = snapshot["files"].get(rel)
        if not expected:
            raise ValueError(f"Source snapshot is missing {rel}")
        src = source_root / rel
        actual_sha = sha256(src)
        if actual_sha != expected["sha256"] or src.stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen source file mismatch: {rel}")
        text = src.read_text(encoding="utf-8")
        check_public_text(text, f"source file {rel}")
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        ensure_new(dst)
        shutil.copyfile(src, dst)
        selected.append({"path": rel, "bytes": expected["bytes"], "sha256": expected["sha256"], "content": "frozen source, copied byte-for-byte"})
    public_file_index = {
        rel: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for rel, item in snapshot["files"].items()
    }
    dump_json(output_root / "source" / "snapshot_file_index.json", {
        "snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
        "files": public_file_index,
        "note": "The private worktree path is omitted. This index binds every relative path and blob hash from the frozen snapshot.",
    })
    dump_json(output_root / "source" / "frozen_sources.json", {
        "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
        "included_files": selected,
        "scope": "Training, inference, scoring, checkpoint, and NF4 native/streaming control-path source excerpt. This is a code evidence bundle, not a standalone replay package.",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Path to the local R112 run directory")
    parser.add_argument("--snapshot-json", type=Path, required=True, help="Path to the frozen SOURCE_SNAPSHOT.json")
    parser.add_argument("--source-root", type=Path, help="Override the worktree root recorded in the snapshot")
    parser.add_argument("--output-root", type=Path, required=True, help="New empty release-candidate directory")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (
        args.output_root / "data" / "evaluation_items.csv",
        args.output_root / "data" / "training_tokens.jsonl",
        args.output_root / "data" / "model_outputs.jsonl",
        args.output_root / "data" / "judgments.csv",
        args.output_root / "data" / "training_updates.jsonl",
        args.output_root / "data" / "resource_samples.csv",
        args.output_root / "data" / "checkpoint_hashes.csv",
        args.output_root / "data" / "training_timing.json",
        args.output_root / "data" / "source_provenance.json",
        args.output_root / "source" / "snapshot_file_index.json",
        args.output_root / "source" / "frozen_sources.json",
    ):
        ensure_new(path)
    emit_public_data(args.run_root, args.output_root)
    emit_source_bundle(args.snapshot_json, args.source_root, args.output_root)
    print("Extracted sanitized evidence and copied frozen source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
