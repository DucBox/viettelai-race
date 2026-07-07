# Model directory layout

This repo does **not** download models — weights are assumed to already be
present. `scripts/01_check_model.sh` only *verifies* the directory is complete
before serving; it never fetches anything.

Place each model as a flat HuggingFace-format directory here:

```
serve/models/
  qwen3.5-2b/
    config.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors.index.json
    model.safetensors-00001-of-00001.safetensors   # (or however many shards the index lists)
    vocab.json                (optional but expected)
    merges.txt                (optional but expected)
    chat_template.jinja       (optional but expected)
    preprocessor_config.json  (optional — multimodal)
    video_preprocessor_config.json (optional — multimodal)
    LICENSE / README.md       (optional, informational)
```

Required for `scripts/01_check_model.sh` to pass:
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- Either `model.safetensors` (single-shard) or `model.safetensors.index.json`
  plus every shard file it references.

Everything else is optional — its absence only prints a warning.

## Getting the files onto the GPU box

However you obtain the weights (this project doesn't do it), get them into
`serve/models/<name>/` before serving, e.g.:

```bash
# from a machine that already has them cached (HF cache -> flat dir):
cp -RL ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ serve/models/qwen3.5-2b/

# or scp/rsync from wherever they live:
rsync -avP other-host:/path/to/qwen3.5-2b/ serve/models/qwen3.5-2b/
```

Then verify:

```bash
./scripts/01_check_model.sh
```
