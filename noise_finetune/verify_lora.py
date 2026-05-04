#!/usr/bin/env python3
"""
Verify a trained noise-music LoRA by running Listener captioning on the
held-out tracks listed in noise_finetune/training_data/heldout_ids.json.

Reads the pre-extracted audio codes from merzbow_dataset_with_codes.json
(no DiT re-encoding needed), loads acestep-5Hz-lm-1.7B + the LoRA adapter,
and generates a caption per held-out track. Use --compare to A/B against the
base model — that's the meaningful test: did training shift the vocabulary
toward "harsh noise / feedback / drone / power electronics" instead of the
pop-music defaults the base LM falls back to on unfamiliar audio?

Planner direction (caption → codes → DiT decode → listen) requires audio
synthesis, not text inspection — out of scope here. Run that through the
normal generation pipeline manually.

Usage:
    python noise_finetune/verify_lora.py
    python noise_finetune/verify_lora.py --compare
    python noise_finetune/verify_lora.py --track <id>
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(PROJECT_ROOT))

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from acestep.constants import DEFAULT_LM_UNDERSTAND_INSTRUCTION


SYSTEM_LISTENER = f"# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n"


def generate_caption(model, tokenizer, audio_codes, temperature, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_LISTENER},
        {"role": "user", "content": audio_codes},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=str(PROJECT_ROOT / "checkpoints" / "acestep-5Hz-lm-1.7B"),
    )
    parser.add_argument(
        "--adapter",
        default=str(PROJECT_ROOT / "noise_finetune" / "lora_output" / "checkpoint-final"),
    )
    parser.add_argument(
        "--heldout-json",
        default=str(PROJECT_ROOT / "noise_finetune" / "training_data" / "heldout_ids.json"),
    )
    parser.add_argument(
        "--codes-json",
        default=str(PROJECT_ROOT / "merzbow" / "captioning" / "merzbow_dataset_with_codes.json"),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run the base model (no adapter) on each track for A/B inspection.",
    )
    parser.add_argument(
        "--track",
        default=None,
        help="Verify a single track id instead of the held-out list.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # --- Load codes JSON ---
    codes_path = Path(args.codes_json)
    if not codes_path.exists():
        logger.error(f"Missing {codes_path}. Run extract_audio_codes.py first.")
        sys.exit(1)
    with open(codes_path) as f:
        dataset_json = json.load(f)
    samples_by_id = {
        s["id"]: s for s in dataset_json.get("samples", []) if s.get("audio_codes")
    }
    logger.info(f"Loaded {len(samples_by_id)} samples with codes.")

    # --- Pick which tracks to verify ---
    if args.track:
        if args.track not in samples_by_id:
            logger.error(f"Track id {args.track!r} not found in {codes_path}")
            sys.exit(1)
        track_ids = [args.track]
    else:
        heldout_path = Path(args.heldout_json)
        if not heldout_path.exists():
            logger.error(f"Missing {heldout_path}. Run build_training_dataset.py first.")
            sys.exit(1)
        with open(heldout_path) as f:
            track_ids = json.load(f)
        track_ids = [t for t in track_ids if t in samples_by_id]
    logger.info(f"Verifying {len(track_ids)} track(s)")

    # --- Load tokenizer + base model + adapter ---
    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    base_model.eval()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)
    logger.info(f"Applying LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    # --- Run inference ---
    for track_id in track_ids:
        sample = samples_by_id[track_id]
        codes = sample["audio_codes"]
        ground = sample.get("caption", "(no caption in dataset)")

        print("\n" + "=" * 70)
        print(f"TRACK: {track_id}")
        print(f"GROUND TRUTH: {ground}")
        print("-" * 70)

        if args.compare:
            with model.disable_adapter():
                base_out = generate_caption(
                    model, tokenizer, codes, args.temperature, args.max_new_tokens
                )
            print("BASE MODEL:")
            print(base_out)
            print("-" * 70)

        adapter_out = generate_caption(
            model, tokenizer, codes, args.temperature, args.max_new_tokens
        )
        print("LORA ADAPTER:")
        print(adapter_out)
        print("=" * 70)


if __name__ == "__main__":
    main()
