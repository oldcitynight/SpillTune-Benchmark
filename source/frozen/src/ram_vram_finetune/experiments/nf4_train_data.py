"""Training-only startup projection for the new four-update diagnostic.

This module never reads a scoring database and never generates real samples.
Production sampling/provider qualification is a separate, still pending step.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

POLICY = "nf4_train_only_short_r60_v1"
SAMPLING = "fresh_random_each_round_fixed_recipe_v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_FIELDS = {"schema", "ready", "round_id", "generation_id", "data_id",
    "sampling_policy", "recipe_sha256", "environment_sha256", "environment_path", "source_adapter",
    "cpu_quota_percent", "minimum_review_remaining_seconds"}


class TrainStagePending(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def entry_exists(path):
    """Startup marker check: a dangling symlink is still an invalid entry."""
    path=Path(path)
    return path.exists() or path.is_symlink()


def file_bytes(path, limit=8 * 1024**2):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("E_PRIVATE_INPUT")
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("E_PRIVATE_INPUT")
    return data


def decode_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("E_PRIVATE_INPUT")
            result[k] = v
        return result
    return json.loads(data, object_pairs_hook=pairs)


def read_json(path, limit=8 * 1024**2):
    return decode_json(file_bytes(path, limit))


def require_committed_draft(path):
    """R94 producer bundles only; legacy independently selected files remain.

    A fully written document in a failed/partial draft is not a committed
    input. Normal single-machine storage/cooperating-owner boundary only.
    """
    path=Path(path).absolute();root=path.parent;marker=root/'DRAFT_BUNDLE.json'
    if not entry_exists(marker):return
    if any(entry_exists(root/k)for k in('PENDING.json','INVALID.json')):raise ValueError('E_DRAFT_NOT_COMMITTED')
    m=read_json(marker);complete=read_json(root/'COMPLETE.json')
    if(type(m)is not dict or set(m)!={'schema','files'}or m['schema']!='rvft.data_draft_bundle.r94.v1'
       or type(m['files'])is not dict or path.name not in m['files']
       or complete!={'draft_bundle_sha256':hashlib.sha256(file_bytes(marker)).hexdigest()}
       or m['files'][path.name]!=hashlib.sha256(file_bytes(path)).hexdigest()):raise ValueError('E_DRAFT_NOT_COMMITTED')


def is_train_only(plan):
    policy = plan.get("train_stage_policy")
    if policy is None:
        if "train_stage" in plan:
            raise ValueError("E_TRAIN_POLICY")
        return False
    if policy != POLICY:
        raise ValueError("E_TRAIN_POLICY")
    validate_plan(plan)
    return True


def validate_plan(plan):
    from .nf4_host_manual import validate_short_plan
    validate_short_plan(plan)  # original exact math, native policy and plan digest
    from .nf4_host_policy import is_host_short
    is_host_short(plan)  # original strict runtime/OS-memory/no-debug exclusions
    s, r = plan.get("train_stage"), plan["runtime"]
    if type(s) is not dict or set(s) != _STAGE_FIELDS or s.get("schema") != "rvft.nf4.train_stage.v1":
        raise ValueError("E_TRAIN_POLICY")
    if s.get("ready") is not True:
        raise TrainStagePending("E_PROJECTION_AND_REVIEW_PENDING")
    for key in ("round_id", "generation_id", "data_id"):
        if type(s[key]) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", s[key]) is None:
            raise ValueError("E_PRIVATE_IDENTITY")
    for key in ("recipe_sha256", "environment_sha256"):
        if type(s[key]) is not str or _HEX.fullmatch(s[key]) is None:
            raise ValueError("E_PRIVATE_IDENTITY")
    if type(s["environment_path"]) is not str or not Path(s["environment_path"]).is_absolute():
        raise ValueError("E_ENVIRONMENT_BINDING")
    if (s["sampling_policy"] != SAMPLING or s["source_adapter"] != "qwen35_nf4_dq_bf16c"
            or type(s["cpu_quota_percent"]) is not int or s["cpu_quota_percent"] != 100
            or type(s["minimum_review_remaining_seconds"]) is not int
            or s["minimum_review_remaining_seconds"] != 1800):
        raise ValueError("E_TRAIN_POLICY")
    if (plan.get("host_resource_policy") != "host_startup_only_functional4_r45_v1"
            or plan.get("host_resume_from") is not None or plan.get("phase") != "real_checkpoint"
            or r.get("synthetic_fixture") is not False or len(r["order_ids"]) != 24
            or len(set(r["order_ids"])) != 24):
        raise ValueError("E_TRAIN_POLICY")
    forbidden = ("diagnostic_rows_path", "diagnostic_rows_sha256", "protocol_path", "protocol_file_sha256")
    if any(k in plan for k in forbidden) or "validation_ids" in r:
        raise ValueError("E_SCORING_INPUT_FORBIDDEN")
    if any(plan.get(k) is not None for k in ("timing_probe_policy", "numeric_probe_policy", "full_chain_timer_options", "streaming_debug_policy")):
        raise ValueError("E_TRAIN_POLICY")
    return s


def _review_contract(plan, pair_admission):
    s = validate_plan(plan)
    return {"schema": "rvft.nf4.train_only_review.v1", "policy": POLICY,
        "plan_sha256": plan["plan_sha256"], "source_snapshot_sha256": plan["source_snapshot_sha256"],
        "arm": plan["arm"], "train_stage": s, "rows_sha256": plan["rows_sha256"],
        "protocol_sha256": plan["protocol_sha256"], "actions": ["construct", "update", "checkpoint"],
        "generation_allowed": False, "formal_allowed": False,
        "pair_admission": pair_admission}


def expected_review_binding(plan):
    from .nf4_train_pair import review_binding
    return _review_contract(plan, review_binding(plan))


def validate_review(plan, review):
    if canonical(review.get("train_only_binding")) != canonical(expected_review_binding(plan)):
        raise ValueError("E_TRAIN_REVIEW")
    if review.get("allowed_actions") != ["construct", "update", "checkpoint"]:
        raise ValueError("E_TRAIN_REVIEW")


def validate_pair(native, streaming):
    """Startup-only same-round/same-task contract; never launches either arm."""
    validate_plan(native);validate_plan(streaming)
    if native['arm'] != 'independent_native_resident' or streaming['arm'] != 'spilltune_streaming':
        raise ValueError('E_PAIR_IDENTITY')
    shared = ('train_stage', 'runtime', 'rows_path', 'rows_sha256', 'protocol_sha256',
              'source_snapshot_sha256', 'source_payload_sha256', 'source_config_sha256',
              'artifact_fingerprint', 'artifact_manifest_sha256', 'tokenizer_identity_sha256',
              'training_integrity_policy', 'budget', 'memory_swap_max_bytes')
    if any(canonical(native[k]) != canonical(streaming[k]) for k in shared):
        raise ValueError('E_PAIR_IDENTITY')
    if any(native[k] == streaming[k] for k in ('control', 'review_path', 'host_launch_nonce')):
        raise ValueError('E_PAIR_REPLAY')
    return {'status': 'PAIR_PREPARED_NOT_AUTHORIZED', 'updates_per_arm': 4}


@dataclass
class TrainProjection:
    rows: dict
    order: list
    counts: list
    identity: dict


def load_projection(plan, *, vocab_size=None):
    s = validate_plan(plan)
    path = Path(plan["rows_path"])
    if type(plan.get("rows_sha256")) is not str or _HEX.fullmatch(plan["rows_sha256"]) is None:
        raise ValueError("E_PRIVATE_INPUT")
    raw = file_bytes(path)
    if hashlib.sha256(raw).hexdigest() != plan["rows_sha256"]:
        raise ValueError("E_PRIVATE_INPUT")
    d = decode_json(raw)  # the exact bytes hashed above, no second content read
    fields = {"schema", "role", "round_id", "generation_id", "data_id", "sampling_policy",
              "recipe_sha256", "protocol_sha256", "order_ids", "rows_by_id"}
    if type(d) is not dict or set(d) != fields or d["schema"] != "rvft.private_train_projection.v1" or d["role"] != "training":
        raise ValueError("E_PRIVATE_INPUT")
    if any(d[k] != s[k] for k in ("round_id", "generation_id", "data_id", "sampling_policy", "recipe_sha256")):
        raise ValueError("E_PRIVATE_IDENTITY")
    order, rows = d["order_ids"], d["rows_by_id"]
    if d["protocol_sha256"] != plan["protocol_sha256"] or order != plan["runtime"]["order_ids"] or type(rows) is not dict or set(rows) != set(order):
        raise ValueError("E_PRIVATE_IDENTITY")
    counts = []
    for key in order:
        row = rows[key]
        if type(row) is not dict or set(row) != {"input_ids", "labels"}:
            raise ValueError("E_PRIVATE_INPUT")
        ids, labels = row["input_ids"], row["labels"]
        if type(ids) is not list or type(labels) is not list or not 2 <= len(ids) <= 1024 or len(labels) != len(ids):
            raise ValueError("E_PRIVATE_INPUT")
        if any(type(v) is not int or v < 0 or (vocab_size is not None and v >= vocab_size) for v in ids):
            raise ValueError("E_PRIVATE_INPUT")
        first = next((i for i, x in enumerate(labels) if x != -100), None)
        if (first is None or first < 1 or any(type(v) is not int for v in labels)
                or labels[:first] != [-100] * first or labels[first:] != ids[first:]):
            raise ValueError("E_PRIVATE_INPUT")
        counts.append({"input_tokens": len(ids), "supervised_tokens": sum(x != -100 for x in labels[1:])})
    return TrainProjection(rows, order, counts, {k: s[k] for k in ("round_id", "generation_id", "data_id", "recipe_sha256")})


def verify_plan_data(plan):
    """Central startup-only data admission; never reads old mixed row capsules."""
    projection = load_projection(plan)
    return verify_projection_identity(projection)


def verify_projection_identity(projection):
    """Startup identity for already-owned rows, matching real execution schema."""
    selected = [{"id": k, **projection.rows[k]} for k in projection.order]
    return digest(projection.order), digest(selected)
