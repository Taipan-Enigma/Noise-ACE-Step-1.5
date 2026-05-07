#!/usr/bin/env python3
"""A/B inference using BOTH the LM Planner LoRA and the DiT LoRA stacked.

Generates the same noise + neo-soul prompts as inference_compare.py, but
routes the caption through the noise-finetuned LM Planner first to produce
audio codes, then feeds those codes to the DiT (with or without the merzbow
DiT LoRA loaded).

Why two stages:
  - The LM Planner LoRA shapes the *codes* (structural plan) from the caption.
  - The DiT LoRA shapes the *texture* (timbre/feel) of the rendered audio.
Stacking both is the strongest in-distribution setup for noise prompts.
For the OOD neo-soul prompt, the LM LoRA may bend things back toward noise —
that's the failure mode worth listening for.

Usage:
    python noise_finetune/inference_with_lm.py                       # both LoRAs
    python noise_finetune/inference_with_lm.py --no-dit-lora         # LM LoRA only
    python noise_finetune/inference_with_lm.py --no-lm-lora          # DiT LoRA only
    python noise_finetune/inference_with_lm.py --no-dit-lora --no-lm-lora  # base
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler


NEO_SOUL_PROMPT = {
    "caption": (
        "A groovy neo-soul track with warm Wurlitzer keys, tight pocket drums, "
        "and silky female vocals. Features rich harmonies, subtle guitar licks, "
        "and a head-nodding groove that blends classic soul with modern production."
    ),
    "lyrics": (
        "[Intro]\n\n"
        "[Verse 1]\nSunday morning golden light\nYou stayed over through the night\n"
        "Coffee brewing records spin\nThis is where our love begins\n\n"
        "[Pre-Chorus]\nNo rush no hurry\nNo stress no worry\n\n"
        "[Chorus]\nEasy like a Sunday morning\nLove without a warning\n"
        "You and me we flow so free\nEasy like it's meant to be\n\n"
        "[Verse 2]\nBarefoot dancing in the kitchen\nThis right here is what I'm missing\n"
        "Simple moments pure and true\nAll I need is me and you\n\n"
        "[Chorus]\nEasy like a Sunday morning\nLove without a warning\n"
        "You and me we flow so free\nEasy like it's meant to be\n\n"
        "[Bridge]\n\nLet the world keep spinning\nWe're just here winning\n\n"
        "[Outro]\nEasy easy\nSo easy"
    ),
    "bpm": 92,
    "keyscale": "Ab major",
    "timesignature": "4",
    "vocal_language": "en",
    "duration": 60.0,
}

NOISE_PROMPT = {
    "caption": (
        "A relentless wall of harsh feedback and distorted electronics. "
        "Dense granular textures, frantic gated noise screeches, and oppressive "
        "industrial machine-like loops. Power electronics with no melody — pure noise."
    ),
    "lyrics": "[Instrumental]",
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "vocal_language": "unknown",
    "duration": 60.0,
}


def save_audio(audio_tensor: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    if audio_tensor.dim() != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(audio_tensor.shape)}")
    arr = audio_tensor.detach().cpu().float().numpy().T
    sf.write(str(output_path), arr, sample_rate)
    secs = audio_tensor.shape[1] / sample_rate
    print(f"  wrote {output_path}  ({secs:.1f}s @ {sample_rate} Hz)")


def generate_one(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    label: str,
    prompt: dict,
    out_dir: Path,
    seed: int,
) -> bool:
    print(f"\n=== Generating: {label} ===")
    print(f"  caption: {prompt['caption'][:80]}...")

    user_metadata = {"duration": int(prompt["duration"])}
    if prompt.get("bpm") is not None:
        user_metadata["bpm"] = int(prompt["bpm"])
    if prompt.get("keyscale"):
        user_metadata["keyscale"] = prompt["keyscale"]
    if prompt.get("timesignature"):
        user_metadata["timesignature"] = prompt["timesignature"]

    print("  [LM Planner] generating audio codes...")
    lm_result = llm_handler.generate_with_stop_condition(
        caption=prompt["caption"],
        lyrics=prompt["lyrics"],
        infer_type="llm_dit",   # phase 1 (metas) + phase 2 (codes)
        temperature=0.85,
        target_duration=prompt["duration"],
        user_metadata=user_metadata,
        use_cot_metas=True,
        use_cot_caption=False,
        use_cot_language=False,
        use_constrained_decoding=True,
    )
    if not lm_result.get("success", False):
        print(f"  LM FAILED: {lm_result.get('error')}")
        return False
    audio_codes = lm_result.get("audio_codes", "") or ""
    n_codes = audio_codes.count("<|audio_code_")
    print(f"  [LM] {n_codes} code tokens (~{n_codes / 5:.1f}s @ 5Hz)")

    result = dit_handler.generate_music(
        captions=prompt["caption"],
        lyrics=prompt["lyrics"],
        bpm=prompt["bpm"],
        key_scale=prompt["keyscale"],
        time_signature=prompt["timesignature"],
        vocal_language=prompt["vocal_language"],
        audio_duration=prompt["duration"],
        inference_steps=8,
        guidance_scale=7.0,
        shift=3.0,
        use_random_seed=False,
        seed=seed,
        audio_code_string=audio_codes,   # ← what the LM Planner produced
    )
    if not result.get("success", False):
        print(f"  DiT FAILED: {result.get('error', result.get('status_message', '?'))}")
        return False
    audios = result.get("audios", []) or []
    if not audios:
        print("  FAILED: no audio in DiT payload")
        return False
    for i, audio in enumerate(audios):
        suffix = "" if i == 0 else f"_{i}"
        save_audio(audio["tensor"], audio["sample_rate"], out_dir / f"{label}{suffix}.wav")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-lora-path", default="noise_finetune/lora_output/checkpoint-final",
                    help="PEFT adapter dir from noise_finetune/train_lora.py")
    ap.add_argument("--dit-lora-path", default="lora_output/merzbow/final",
                    help="PEFT adapter dir from train.py fixed")
    ap.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B",
                    help="Base LM checkpoint folder under ./checkpoints/")
    ap.add_argument("--output-dir", default="noise_finetune/inference_output_lm_dit",
                    help="Where to write the .wav files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dit-lora", action="store_true",
                    help="Run base DiT (skip merzbow DiT LoRA)")
    ap.add_argument("--no-lm-lora", action="store_true",
                    help="Run base LM Planner (skip noise LM LoRA)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading base DiT (turbo)...")
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path="acestep-v15-turbo",
        device="auto",
    )
    if not ok:
        print(f"DiT init failed: {status}")
        return 1

    print(f"\n[2/4] Loading LM ({args.lm_model}, backend=pt)...")
    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
        lm_model_path=args.lm_model,
        backend="pt",      # PEFT requires the pytorch backend, not vllm
        device="auto",
    )
    if not ok:
        print(f"LM init failed: {status}")
        return 1

    if not args.no_lm_lora:
        if not Path(args.lm_lora_path).is_dir():
            print(f"  ❌ LM LoRA path not found: {args.lm_lora_path}")
            return 1
        print(f"  attaching LM LoRA: {args.lm_lora_path}")
        llm.llm = PeftModel.from_pretrained(llm.llm, args.lm_lora_path)
        llm.llm.eval()
        print("  LM LoRA active")
    else:
        print("  skipping LM LoRA (--no-lm-lora)")

    if not args.no_dit_lora:
        print(f"\n[3/4] Loading DiT LoRA: {args.dit_lora_path}")
        lora_status = dit.add_lora(args.dit_lora_path, adapter_name="merzbow")
        print(f"  {lora_status}")
        if lora_status.startswith("❌"):
            return 1
        dit.set_use_lora(True)
        dit.set_active_lora_adapter("merzbow")
        dit.set_lora_scale(1.0)
    else:
        print("\n[3/4] Skipping DiT LoRA (--no-dit-lora)")

    print("\n[4/4] Generating tracks...")
    ok_n = generate_one(dit, llm, "noise_track", NOISE_PROMPT, out_dir, args.seed)
    ok_s = generate_one(dit, llm, "neo_soul_track", NEO_SOUL_PROMPT, out_dir, args.seed)

    print(f"\nDone. Outputs in {out_dir}/")
    return 0 if (ok_n and ok_s) else 1


if __name__ == "__main__":
    sys.exit(main())
