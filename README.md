# SpillTune Benchmark — R112

This repository packages a focused source slice, sanitized exports, and reports for one formal SpillTune Q4 training run. Its verifier recomputes the published aggregates from the included exports; the package is not a complete replay environment for the original model run. It does not contain model weights, LoRA/checkpoint tensors, private SQLite files, TUI frames, actor/session records, or provider request payloads.

## What this run observed

- The `spilltune_streaming` arm sampled **2.03 GiB** whole-device SMI memory at peak, versus **3.67 GiB** for the `independent_native_resident` arm (**−44.7%** in this run). This is a whole-GPU sampled peak, not process-exclusive allocated memory or a guaranteed instantaneous maximum.
- Training with checkpoint saves took **76.27 minutes** for streaming and **60.98 minutes** for native resident (**+25.1%**). These paired run times are the same order of magnitude. There was no CPU baseline or GPU-offload baseline in this test.
- The two runs produced exactly matching output text, generated token IDs, and stop reasons for all **500** paired evaluation items. Their 500-step loss sequences matched. The package lists matching checkpoint SHA-256 and byte-length fingerprints at updates 0, 100, 200, 300, 400, and 500; the tensor files are not included, so readers cannot independently re-hash those binary payloads from this package.
- Stored scoring totals differed by at most **0.6 percentage points** across the five judgment channels. Since the paired generated answers were identical, those score differences are differences in rating records, not output changes.

This is a single local run pair. It does not establish statistical equivalence, universal quality preservation, or general performance across hardware. The run carries the `LOCAL_OS_STARTUP_NOT_GPU_QUALIFICATION` label; no independent GPU qualification is claimed.

## Contents

- [`reports/R112_REPORT.md`](reports/R112_REPORT.md): methods, results, and limits.
- [`reports/summary.json`](reports/summary.json): machine-readable recomputation from the packaged records.
- [`data/`](data/): evaluation items, tokenized training rows, exact generated outputs, channel-labeled judgment rows with actor/session fields stripped, update/loss records, normalized resource samples, and checkpoint hashes.
- [`source/frozen/`](source/frozen/): 19 relevant files copied byte-for-byte from the frozen R112 source snapshot. The public source index records relative paths and hashes for all 559 snapshot files, but contains source bodies for only those 19. [`source/README.md`](source/README.md) explains how this slice overlaps the later public SpillTune Q4 commit.
- [`protocol/r112_run_config.json`](protocol/r112_run_config.json): the public-safe frozen training and evaluation settings.
- The Q4 training API shipped in SpillTune commit [`d8764fbed6004897c74afd14ba5e34875dc3d6a2`](https://github.com/oldcitynight/SpillTune/commit/d8764fbed6004897c74afd14ba5e34875dc3d6a2).
- [`scripts/extract_public_release.py`](scripts/extract_public_release.py): extraction procedure, for authorized local source files only.
- [`scripts/verify_release.py`](scripts/verify_release.py): data integrity and aggregate verification, including equality of the published checkpoint fingerprints. It does not train, run inference, re-hash omitted tensor files, or call a judge.
- [`scripts/render_tradeoffs.py`](scripts/render_tradeoffs.py): optional vector-figure renderer from `reports/summary.json` (requires Matplotlib).
- [`DATA_LICENSES.md`](DATA_LICENSES.md): code, dataset, model, and generated-record provenance.

## Verify

Requires Python 3.10+ and only the standard library.

```bash
python scripts/verify_release.py
```

To regenerate the sanitized data from the original local files, use a new empty output directory and pass the local run root and frozen `SOURCE_SNAPSHOT.json` explicitly:

```bash
python scripts/extract_public_release.py \
  --run-root /path/to/r112-formal-run \
  --snapshot-json /path/to/SOURCE_SNAPSHOT.json \
  --output-root /path/to/new-release-candidate
```

The extractor checks the frozen source hashes and the original database hashes before reading. It never copies the SQLite databases into the release. The environment table describes fields captured in the historical R112 records; missing CPU model and total-RAM fields are marked as unavailable rather than filled from the current host.

## Scope and limits

The baseline is `independent_native_resident`: official bitsandbytes NF4 matmul with packed NF4 records and `QuantState` resident on CUDA, plus native non-reentrant activation checkpointing. Streaming stages layer records from pinned CPU memory to CUDA for each forward and backward pass, and its custom backward recomputes the native layer. This comparison is not against GPU native offload.

The model repository revision used by the run was not pinned. The weights are not included, so exact model reruns cannot be guaranteed. The original local audit recorded separate worker processes, plans, work directories, and checkpoint paths for the two arms, with a shared source snapshot and initialization/profile. The public package reports this audit finding but does not contain the original paths, inode metadata, or checkpoint binaries for independent filesystem-level revalidation.

The judgment channel labels are retained as recorded labels, not independent authentication of the backend model identity. Cgroup I/O deltas were unavailable. No p-values, equivalence tests, or majority-vote adjudication are provided.
