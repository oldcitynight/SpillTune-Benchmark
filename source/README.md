# Frozen execution source

`frozen/` contains a focused set of 19 exact source files from the formal run's frozen 559-file source snapshot. File bytes and SHA-256 values match `source/snapshot_file_index.json`. The snapshot as a whole is bound by SHA-256 `5c64acd760d4c2f990f3bb403a75b1f7e05dcfe5182fa54df491cf97a6ed1f9f`.

The included files cover the training worker and scheduler, frozen data reader and update loop, inference staging/runtime, scoring and final analysis, native-resident and streaming adapters, native activation-checkpoint policy, streaming lease pool and backward recomputation, CUDA NF4 backend, and checkpoint writing.

The complete snapshot index contains relative source paths and hashes only. It omits the original private worktree path and does not copy the other 540 file bodies, tests, machine-specific run manifests, or execution artifacts. This focused source excerpt is evidence of the executed code path; it is not a self-contained environment for replaying the original GPU run. The release verifier recomputes published aggregates only; a full run replay requires additional source, exact runtime dependencies, and the model revision/weights, which are not supplied here.

The Q4 library API was subsequently published in SpillTune commit `d8764fbed6004897c74afd14ba5e34875dc3d6a2` (linked from the root README). A local path-and-byte comparison against that commit found six of the 19 included paths byte-identical:

- `src/ram_vram_finetune/quantization/native_terminal.py`
- `src/ram_vram_finetune/quantization/nf4_backend_cuda.py`
- `src/ram_vram_finetune/training/nf4_cuda_pool.py`
- `src/ram_vram_finetune/training/nf4_cuda_recompute.py`
- `src/ram_vram_finetune/training/nf4_preflight.py`
- `src/ram_vram_finetune/training/nf4_streaming.py`

`src/ram_vram_finetune/adapters/qwen35/nf4_cuda.py` exists in the commit but differs from the frozen R112 bytes. The other 12 included paths are absent from that commit:

- `src/ram_vram_finetune/adapters/qwen35/nf4_native.py`
- `src/ram_vram_finetune/checkpoint_io/nf4_cuda_training.py`
- `src/ram_vram_finetune/checkpoint_io/nf4_training.py`
- `src/ram_vram_finetune/experiments/nf4_execution.py`
- `src/ram_vram_finetune/experiments/nf4_final_analysis.py`
- `src/ram_vram_finetune/experiments/nf4_full_train.py`
- `src/ram_vram_finetune/experiments/nf4_inference_runtime.py`
- `src/ram_vram_finetune/experiments/nf4_inference_stage.py`
- `src/ram_vram_finetune/experiments/nf4_scoring_stage.py`
- `src/ram_vram_finetune/experiments/nf4_stage_worker.py`
- `src/ram_vram_finetune/experiments/nf4_train_data.py`
- `src/ram_vram_finetune/training/nf4_native_memory.py`

The public commit therefore provides the maintained library API, not the complete R112 training/inference/scoring harness or its frozen source snapshot. The 559-file index cannot restore source bodies that are not included.
