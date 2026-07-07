#!/usr/bin/env python3
"""Print the REAL nn.Module tree of Qwen3.5-2B, instantiated on the meta device
(structure only, no weights loaded → fast + low memory).

Needs transformers-from-source (qwen3_5 is not in any stable release yet) which
requires Python >=3.10 — so this is meant to run inside the python:3.11 Docker
container launched by scripts/06_module_tree_docker.sh, with the flat model
directory mounted read-only at /model.

Env:
    HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  MODEL_ID=/model
"""
import os
import torch
from transformers import AutoConfig

MODEL_ID = os.environ.get("MODEL_ID", "/model")

cfg = AutoConfig.from_pretrained(MODEL_ID)
print(f"model_type={cfg.model_type}  arch={cfg.architectures}\n")

# Instantiate on meta device — allocates no real tensors.
model = None
errors = []
for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
    try:
        import transformers
        Loader = getattr(transformers, loader_name)
        with torch.device("meta"):
            model = Loader.from_config(cfg)
        print(f"[instantiated via {loader_name}]\n")
        break
    except Exception as e:  # noqa: BLE001
        errors.append(f"{loader_name}: {e}")
if model is None:
    print("Could not instantiate. Attempts:")
    for e in errors:
        print("  -", e)
    raise SystemExit(1)

print("=" * 78)
print("FULL MODULE TREE")
print("=" * 78)
print(model)

print("\n" + "=" * 78)
print("PARAM COUNT BY TOP-LEVEL SUBMODULE (meta shapes → numel)")
print("=" * 78)


def count(m):
    return sum(p.numel() for p in m.parameters())


total = count(model)
for name, child in model.named_children():
    print(f"  {count(child)/1e9:7.3f} B  {100*count(child)/total:5.1f}%  {name}  ({type(child).__name__})")
    for sub_name, sub in child.named_children():
        print(f"      {count(sub)/1e9:7.3f} B  {sub_name}  ({type(sub).__name__})")
print(f"  ------\n  {total/1e9:.3f} B total")

# Show ONE GDN layer and ONE full-attention layer in detail (the two building blocks)
print("\n" + "=" * 78)
print("THE TWO DECODER BLOCK TYPES (detailed)")
print("=" * 78)
try:
    layers = model.model.language_model.layers
except AttributeError:
    try:
        layers = model.language_model.model.layers
    except AttributeError:
        layers = None
if layers is not None:
    print("\n--- Layer 0 (linear-attention / Gated DeltaNet) ---")
    print(layers[0])
    print("\n--- Layer 3 (full attention) ---")
    print(layers[3])
