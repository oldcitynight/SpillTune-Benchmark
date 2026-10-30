# Licenses and provenance

## SpillTune code

SpillTune project source and the extraction/verification scripts in this repository are provided under GPL-3.0-or-later. See [`LICENSE.md`](LICENSE.md) and the complete [`COPYING`](COPYING) text. Upstream attributions are retained in [`SOURCE_ATTRIBUTION.md`](SOURCE_ATTRIBUTION.md), [`NOTICE`](NOTICE), and [`legal/`](legal/).

## GSM8K-derived data

The 1,500 training rows and 500 evaluation questions/answers come from the locally frozen R112 data preparation based on [`openai/grade-school-math`](https://github.com/openai/grade-school-math), revision `3101c7d5072418e28b9008a6636bde82a006892c`. The dataset repository's license at that revision is MIT: [`LICENSE`](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE). The public files preserve attribution and the source database SHA-256 values in `data/source_provenance.json`.

`data/evaluation_items.csv` copies the selected question and standard-answer field values verbatim into CSV. `data/training_tokens.jsonl` serializes the selected training table's `input_ids` and `labels` values without changing those arrays. These are derived exports, not byte-for-byte copies of the original SQLite files.

## Model provenance

The recorded model identifier is `huihui-ai/Huihui-Qwopus3.5-4B-v3-abliterated`, based on `Jackrong/Qwopus3.5-4B-v3`. The model card declares Apache-2.0, but the formal run did not pin a model repository revision. No model weights, adapters, or checkpoint tensors are redistributed here; the model card's license is not asserted as a license for the generated outputs.

## Generated outputs and judgment records

The generated outputs and judgment rows are sanitized, user-authorized exports of the formal R112 run. The SQLite containers, raw TUI frames, reviewer/operator identifiers, session/provider request fields, and backend identity evidence are excluded. No separate license is granted for generated outputs or judgment records by this repository's code license.

## Included MIT license text

The GSM8K license for the pinned source revision is reproduced in [`data/licenses/GSM8K-MIT.txt`](data/licenses/GSM8K-MIT.txt).
