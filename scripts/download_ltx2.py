"""Download the LTX-2 weights subset needed by diffusers LTX2Pipeline.

Repo: Lightricks/LTX-2 (not gated, no HF token needed). We fetch only the
components the diffusers pipeline loads:

* transformer/, text_encoder/ (transformers-named shards only), vae/,
  tokenizer/, scheduler/, audio_vae/, connectors/, vocoder/, model_index.json

Deliberately SKIPPED (dead weight for from_pretrained):
* ltx-2-19b-*.safetensors single-file variants at the repo root
* ltx-2-*-upscaler-x2-1.0.safetensors (only for the 2-stage pipeline)
* latent_upsampler/ (only for the 2-stage pipeline)
* text_encoder/diffusion_pytorch_model-* (diffusers-named dup of the
  transformers-named model-* shards; model_index assigns the text encoder
  to transformers.Gemma3ForConditionalGeneration, which ignores them)
* ltx-2-running-local.mp4 (promo video)

Usage:  python scripts/download_ltx2.py
Env:    LTX2_MODEL_DIR overrides the default target dir.
"""
import os
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get(
    "LTX2_MODEL_DIR", "/home/summit/avf-work/item11-ltx2/models/LTX-2"
)

d = snapshot_download(
    repo_id="Lightricks/LTX-2",
    local_dir=MODEL_DIR,
    allow_patterns=[
        "transformer/*",
        "text_encoder/*",
        "vae/*",
        "tokenizer/*",
        "scheduler/*",
        "audio_vae/*",
        "connectors/*",
        "vocoder/*",
        "model_index.json",
    ],
    ignore_patterns=[
        "text_encoder/diffusion_pytorch_model*",
        "latent_upsampler/*",
    ],
    max_workers=8,
)
print("DONE:", d)
