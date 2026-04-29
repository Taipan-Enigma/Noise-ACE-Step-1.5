#!/usr/bin/env python3
"""
Build the HF Dataset used to LoRA fine-tune acestep-5Hz-lm-1.7B.

Reads:
  merzbow/captioning/merzbow_dataset_with_codes.json

Writes:
  noise_finetune/training_data/train/   (HF Dataset.save_to_disk)
  noise_finetune/training_data/eval/    (HF Dataset.save_to_disk)
  noise_finetune/training_data/heldout_ids.json  (ids withheld from training)

For each enriched sample we emit two rows:

  Listener  (codes -> caption+metadata)   _task = "listener"
  Planner   (caption+lyrics -> codes)      _task = "planner"

Both use the exact system strings + chat template the production model uses
(acestep/constants.py + acestep/llm_inference.py). Labels are pre-masked to
the assistant turn only so plain transformers.Trainer needs no custom loss.
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from loguru import logger
from transformers import AutoTokenizer

from acestep.constants import (
    DEFAULT_LM_INSTRUCTION,
    DEFAULT_LM_UNDERSTAND_INSTRUCTION,
)


SYSTEM_LISTENER = f"# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n"
SYSTEM_PLANNER = f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"


def build_cot_block(sample: dict) -> str:
    """Return the <think>...</think> metadata block used in both directions.

    Matches the production format parsed by LLMHandler.parse_lm_output
    (acestep/llm_inference.py:2747). Values are written as plain YAML-ish
    `key: value` lines; null values go out as `null`.
    """
    def fmt(v):
        if v is None or v == "":
            return "null" if v is None else ""
        return str(v)

    bpm = sample.get("bpm")
    caption = sample.get("caption", "")
    duration = sample.get("duration")
    genre = sample.get("genre", "") or ""
    keyscale = sample.get("keyscale", "") or ""
    language = sample.get("language", "") or ""
    timesig = sample.get("timesignature", "") or ""

    lines = [
        "<think>",
        f"bpm: {fmt(bpm)}",
        f"caption: {caption}",
        f"duration: {fmt(duration)}",
        f"genres: {genre}",
        f"keyscale: {keyscale}",
        f"language: {language}",
        f"timesignature: {timesig}",
        "</think>",
    ]
    return "\n".join(lines)


def build_rows(sample: dict, tokenizer, max_length: int) -> list[dict]:
    """Emit the Listener row and Planner row for a single enriched sample."""
    audio_codes = sample["audio_codes"]
    caption = sample.get("caption", "")
    lyrics = sample.get("formatted_lyrics") or sample.get("lyrics") or "[Instrumental]"
    cot = build_cot_block(sample)

    rows = []

    # --- Listener: codes -> <think>metadata</think> + lyrics ---
    listener_messages = [
        {"role": "system", "content": SYSTEM_LISTENER},
        {"role": "user", "content": audio_codes},
    ]
    listener_assistant = f"{cot}\n{lyrics}"
    rows.append(
        _tokenize_with_labels(
            tokenizer,
            messages=listener_messages,
            assistant_text=listener_assistant,
            max_length=max_length,
            task="listener",
            sample_id=sample["id"],
        )
    )

    # --- Planner: caption+lyrics -> <think>metadata</think> + audio codes ---
    planner_user = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    planner_messages = [
        {"role": "system", "content": SYSTEM_PLANNER},
        {"role": "user", "content": planner_user},
    ]
    planner_assistant = f"{cot}\n{audio_codes}"
    rows.append(
        _tokenize_with_labels(
            tokenizer,
            messages=planner_messages,
            assistant_text=planner_assistant,
            max_length=max_length,
            task="planner",
            sample_id=sample["id"],
        )
    )

    return rows


def _tokenize_with_labels(
    tokenizer,
    messages: list[dict],
    assistant_text: str,
    max_length: int,
    task: str,
    sample_id: str,
) -> dict:
    """Tokenize a two-turn prompt + assistant response, masking prompt to -100."""
    # Prefix: system + user + generation prompt marker. No assistant turn yet.
    prefix_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_messages = messages + [{"role": "assistant", "content": assistant_text}]
    full_str = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prefix_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_str, add_special_tokens=False).input_ids

    # Safety: prefix should be an actual prefix of full.
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # Chat template rendering of the two can differ in whitespace; find the
        # longest common prefix instead.
        common = 0
        for a, b in zip(prefix_ids, full_ids):
            if a != b:
                break
            common += 1
        prefix_len = common
    else:
        prefix_len = len(prefix_ids)

    if len(full_ids) > max_length:
        # Only the Planner side (audio codes) should ever hit this.
        full_ids = full_ids[:max_length]

    labels = [-100] * prefix_len + full_ids[prefix_len:]
    # Pad labels to match input_ids length exactly (in case of trimming).
    labels = labels[: len(full_ids)]

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "_task": task,
        "_id": sample_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes-json",
        default=str(PROJECT_ROOT / "merzbow" / "captioning" / "merzbow_dataset_with_codes.json"),
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "checkpoints" / "acestep-5Hz-lm-1.7B"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "noise_finetune" / "training_data"),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--heldout", type=int, default=5, help="Number of tracks to hold out for eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    codes_path = Path(args.codes_json)
    if not codes_path.exists():
        logger.error(f"Missing {codes_path}. Run extract_audio_codes.py first.")
        sys.exit(1)

    with open(codes_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    samples = [s for s in dataset_json.get("samples", []) if s.get("audio_codes")]
    if not samples:
        logger.error("No samples with audio_codes found.")
        sys.exit(1)
    logger.info(f"Loaded {len(samples)} samples with codes.")

    logger.info(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Deterministic holdout split — 5 whole tracks -> 10 eval rows.
    rng = random.Random(args.seed)
    all_ids = sorted(s["id"] for s in samples)
    rng.shuffle(all_ids)
    heldout_ids = set(all_ids[: args.heldout])
    train_samples = [s for s in samples if s["id"] not in heldout_ids]
    eval_samples = [s for s in samples if s["id"] in heldout_ids]
    logger.info(f"Train tracks: {len(train_samples)} | eval tracks: {len(eval_samples)}")

    train_rows: list[dict] = []
    for s in train_samples:
        train_rows.extend(build_rows(s, tokenizer, args.max_length))

    eval_rows: list[dict] = []
    for s in eval_samples:
        eval_rows.extend(build_rows(s, tokenizer, args.max_length))

    # Stats
    lens = [len(r["input_ids"]) for r in train_rows]
    logger.info(
        f"Train rows: {len(train_rows)} | "
        f"len min/avg/max = {min(lens)}/{sum(lens) // len(lens)}/{max(lens)}"
    )

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(out_dir / "train"))
    eval_ds.save_to_disk(str(out_dir / "eval"))
    with open(out_dir / "heldout_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted(heldout_ids), f, indent=2)

    logger.info(f"Saved train + eval datasets under {out_dir}")


if __name__ == "__main__":
    main()
