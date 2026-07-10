#!/usr/bin/env python3
"""Deep-inspect a (quantized) model checkpoint directory and write a single
Markdown report — no torch/transformers required for the core inspection
(pure stdlib: struct/json parse safetensors headers directly), so it runs
anywhere the checkpoint files sit, GPU pod or plain CPU box.

Built to answer one specific question with EVIDENCE instead of guessing:
"script 15's AWQ output is bigger than expected — why?" The report gives
everything needed to diff two runs of this SAME script against two different
directories (e.g. the real cyankiwi checkpoint vs. a scripts/15 output) side
by side, tensor-category by tensor-category, dtype by dtype.

Usage:
    python3 scripts/16_inspect_quantized_model.py --dir serve/models/qwen3.5-2b-awq4bit
    python3 scripts/16_inspect_quantized_model.py --dir serve/models/qwen3.5-2b-awq-int4 \
        --out /tmp/my_output_inspect.md

Run this against BOTH the reference checkpoint and your own scripts/15 output
(same script, same code — so any difference in the two reports is a real
difference in the checkpoints, not a difference in how they were measured),
then diff the two .md files or send both back.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone


def h(f, title, level=2):
    f.write(f"\n{'#' * level} {title}\n\n")


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:3.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def read_safetensors_header(path):
    """Parse ONLY the JSON header of a .safetensors file (first 8 bytes =
    little-endian uint64 header length, then that many bytes of JSON) —
    does not read/materialize any actual tensor data. Works with zero
    dependencies beyond stdlib."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    header.pop("__metadata__", None)
    return header


# Category classification, broad enough to bucket every tensor in this
# architecture (not just the ignored ones script 15 cares about) — see
# scripts/15_quantize_awq.py's IGNORE list / build_mappings() for why these
# specific module names matter.
CATEGORY_PATTERNS = [
    ("embed_tokens", r"embed_tokens"),
    ("lm_head", r"^lm_head"),
    ("mtp", r"(^|\.)mtp"),
    ("visual", r"\bvisual\b"),
    ("linear_attn", r"linear_attn"),
    ("self_attn.q_proj", r"self_attn\.q_proj"),
    ("self_attn.k_proj", r"self_attn\.k_proj"),
    ("self_attn.v_proj", r"self_attn\.v_proj"),
    ("self_attn.o_proj", r"self_attn\.o_proj"),
    ("mlp.gate_proj", r"mlp\.gate_proj"),
    ("mlp.up_proj", r"mlp\.up_proj"),
    ("mlp.down_proj", r"mlp\.down_proj"),
    ("norm/layernorm", r"norm"),
]


def classify(name):
    for label, pat in CATEGORY_PATTERNS:
        if re.search(pat, name):
            return label
    return "other"


def tensor_size(entry):
    lo, hi = entry["data_offsets"]
    return hi - lo


def gather_shards(model_dir):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(idx_path):
        with open(idx_path) as fh:
            idx = json.load(fh)
        shards = sorted(set(idx["weight_map"].values()))
        return [os.path.join(model_dir, s) for s in shards], idx.get("metadata", {})
    elif os.path.isfile(single_path):
        return [single_path], {}
    else:
        # AWQ output may use a different shard filename pattern; glob for it.
        found = sorted(
            fn for fn in os.listdir(model_dir)
            if fn.endswith(".safetensors")
        )
        if found:
            return [os.path.join(model_dir, fn) for fn in found], {}
        sys.exit(f"!! no model.safetensors / model.safetensors.index.json / *.safetensors in {model_dir}")


def try_versions():
    out = {}
    for mod in ("torch", "transformers", "compressed_tensors", "llmcompressor", "datasets", "accelerate"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"NOT IMPORTABLE ({e.__class__.__name__})"
    return out


def try_nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi not available ({e.__class__.__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Model directory to inspect")
    ap.add_argument("--out", default=None, help="Report path (default: <dir>/inspect_report.md)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.dir)
    if not os.path.isdir(model_dir):
        sys.exit(f"!! not a directory: {model_dir}")
    out_path = args.out or os.path.join(model_dir, "inspect_report.md")

    shard_paths, st_metadata = gather_shards(model_dir)

    with open(out_path, "w") as f:
        f.write(f"# Quantized model inspection report\n\n")
        f.write(f"- Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- Host: {os.uname().nodename}\n")
        f.write(f"- Directory inspected: `{model_dir}`\n")
        f.write(f"- Script: scripts/16_inspect_quantized_model.py\n")

        h(f, "1. Environment")
        for mod, ver in try_versions().items():
            f.write(f"- `{mod}`: {ver}\n")
        f.write(f"- `python`: {sys.version.split()[0]}\n")
        f.write(f"- `nvidia-smi`: {try_nvidia_smi()}\n")

        h(f, "2. Directory listing")
        total_dir_bytes = 0
        f.write("| file | bytes | human |\n|---|---:|---:|\n")
        for root, _, files in os.walk(model_dir):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                sz = os.path.getsize(p)
                total_dir_bytes += sz
                rel = os.path.relpath(p, model_dir)
                f.write(f"| `{rel}` | {sz:,} | {human(sz)} |\n")
        f.write(f"\n**Total directory size: {total_dir_bytes:,} bytes ({human(total_dir_bytes)})**\n")

        h(f, "3. config.json quantization_config")
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            f.write(f"- `transformers_version`: `{cfg.get('transformers_version')}`\n")
            f.write(f"- `tie_word_embeddings`: `{cfg.get('tie_word_embeddings')}`\n")
            qc = cfg.get("quantization_config")
            if qc:
                f.write("\n```json\n")
                f.write(json.dumps(qc, indent=2, default=str))
                f.write("\n```\n")
            else:
                f.write("\n**No `quantization_config` key in config.json** — this checkpoint may not be quantized at all.\n")
        else:
            f.write("**config.json not found.**\n")

        h(f, "4. safetensors header inventory")
        f.write(f"- Shard file(s): {[os.path.basename(p) for p in shard_paths]}\n")
        if st_metadata:
            f.write(f"- index `metadata`: {st_metadata}\n")

        all_entries = []  # (name, dtype, nbytes, shape)
        header_declared_total = 0
        for shard in shard_paths:
            header = read_safetensors_header(shard)
            for name, entry in header.items():
                nbytes = tensor_size(entry)
                all_entries.append((name, entry["dtype"], nbytes, entry["shape"]))
                header_declared_total += nbytes

        actual_shard_bytes = sum(os.path.getsize(p) for p in shard_paths)
        f.write(f"\n- Tensor count: {len(all_entries):,}\n")
        f.write(f"- Sum of tensor byte ranges (from headers): {header_declared_total:,} bytes ({human(header_declared_total)})\n")
        f.write(f"- Sum of actual shard file sizes on disk: {actual_shard_bytes:,} bytes ({human(actual_shard_bytes)})\n")
        f.write("  (difference between these two = header/footer overhead, should be tiny — large gaps mean something's off)\n")

        h(f, "5. Breakdown by dtype (whole checkpoint)")
        by_dtype = {}
        for name, dtype, nbytes, shape in all_entries:
            d = by_dtype.setdefault(dtype, {"count": 0, "bytes": 0})
            d["count"] += 1
            d["bytes"] += nbytes
        f.write("| dtype | tensor count | total bytes | total |\n|---|---:|---:|---:|\n")
        for dtype, d in sorted(by_dtype.items(), key=lambda kv: -kv[1]["bytes"]):
            f.write(f"| `{dtype}` | {d['count']:,} | {d['bytes']:,} | {human(d['bytes'])} |\n")

        h(f, "6. Breakdown by module category x dtype")
        by_cat = {}
        for name, dtype, nbytes, shape in all_entries:
            cat = classify(name)
            key = (cat, dtype)
            d = by_cat.setdefault(key, {"count": 0, "bytes": 0})
            d["count"] += 1
            d["bytes"] += nbytes
        f.write("| category | dtype | tensor count | total bytes | total |\n|---|---|---:|---:|---:|\n")
        for (cat, dtype), d in sorted(by_cat.items(), key=lambda kv: (kv[0][0], -kv[1]["bytes"])):
            f.write(f"| `{cat}` | `{dtype}` | {d['count']:,} | {d['bytes']:,} | {human(d['bytes'])} |\n")

        h(f, "7. Tied-embedding check (lm_head vs embed_tokens)")
        by_name = {name: (dtype, nbytes, shape) for name, dtype, nbytes, shape in all_entries}
        embed_keys = [n for n in by_name if "embed_tokens" in n]
        lm_head_keys = [n for n in by_name if re.search(r"^lm_head|(?<!\.)lm_head", n)]
        f.write(f"- `embed_tokens` tensor(s) found: {embed_keys or 'NONE'}\n")
        for n in embed_keys:
            dtype, nbytes, shape = by_name[n]
            f.write(f"  - `{n}`: dtype={dtype}, shape={shape}, bytes={nbytes:,} ({human(nbytes)})\n")
        f.write(f"- `lm_head` tensor(s) found: {lm_head_keys or 'NONE (expected if tied and save_pretrained correctly omitted it)'}\n")
        for n in lm_head_keys:
            dtype, nbytes, shape = by_name[n]
            f.write(f"  - `{n}`: dtype={dtype}, shape={shape}, bytes={nbytes:,} ({human(nbytes)})\n")
        if embed_keys and lm_head_keys:
            e_dtype, e_bytes, e_shape = by_name[embed_keys[0]]
            l_dtype, l_bytes, l_shape = by_name[lm_head_keys[0]]
            same_shape_dtype = (e_dtype, e_shape) == (l_dtype, l_shape)
            f.write(f"\n- **Both present** — if this model is meant to be tied "
                    f"(`tie_word_embeddings: true`), this is 1 full extra copy "
                    f"({human(l_bytes)}) that a properly-tied save would NOT write to disk.\n")
            f.write(f"- Same dtype+shape: {same_shape_dtype}\n")

        h(f, "8. Full per-tensor dump")
        f.write("<details><summary>click to expand — every tensor name/dtype/shape/bytes</summary>\n\n")
        f.write("| name | dtype | shape | bytes |\n|---|---|---|---:|\n")
        for name, dtype, nbytes, shape in sorted(all_entries, key=lambda e: e[0]):
            f.write(f"| `{name}` | `{dtype}` | {shape} | {nbytes:,} |\n")
        f.write("\n</details>\n")

    print(f">> wrote {out_path}")
    print(f">> total dir size: {human(total_dir_bytes)} ({total_dir_bytes:,} bytes)")
    print(f">> tensor count: {len(all_entries):,}")


if __name__ == "__main__":
    main()
