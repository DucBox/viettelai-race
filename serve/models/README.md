# Model directory layout

This repo does **not** download models — weights are assumed to already be
present. `scripts/01_check_model.sh` only *verifies* the directory is complete
before serving; it never fetches anything.

Place each model as a flat HuggingFace-format directory here:

```
serve/models/
  lfm2.5-1.2b/
    config.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors            # LFM2.5-1.2B is a single-shard checkpoint (~2.2 GB, no index)
    special_tokens_map.json   (optional but expected)
    chat_template.jinja       (optional but expected)
    generation_config.json    (optional but expected)
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
# simplest — download straight from HuggingFace (public, not gated):
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir serve/models/lfm2.5-1.2b
#   (no `hf` CLI? python -c "from huggingface_hub import snapshot_download; \
#    snapshot_download('LiquidAI/LFM2.5-1.2B-Instruct', local_dir='serve/models/lfm2.5-1.2b')")

# or scp/rsync from wherever they live:
rsync -avP other-host:/path/to/lfm2.5-1.2b/ serve/models/lfm2.5-1.2b/
```

Then verify:

```bash
./scripts/01_check_model.sh
```

## Using a different model directory name, or a different image

Nothing here is hardcoded to `lfm2.5-1.2b` — both the model path and the vLLM
image are just values in `serve/.env` (copy `serve/.env.example` first):

```bash
# serve/.env

# If your weights live somewhere else / under a different name:
MODEL_DIR=./models/my-other-model     # host path, relative to serve/ (or absolute)
MODEL_PATH=/models/my-other-model     # path INSIDE the container — keep these two matching
SERVED_MODEL_NAME=my-other-model      # name clients (AIPerf, curl) will request

# If you're using an image from your own registry instead of Docker Hub:
IMAGE=registry.internal.example.com/team/vllm-openai:<lfm2-capable-tag>
```

`docker-compose.yml` reads all four from `.env` — no edits to the compose file
or to this repo's scripts are needed. `scripts/01_check_model.sh` also reads
`MODEL_DIR` from `serve/.env`, so it checks whatever path you point it at.

After changing `.env`, re-run:
```bash
./scripts/01_check_model.sh   # verifies the new MODEL_DIR
./scripts/serve_up.sh         # brings up vLLM with the new IMAGE / model path
```
