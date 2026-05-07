#!/usr/bin/env python3
"""A/B inference: generate one noise track and one neo-soul track with the
merzbow LoRA loaded, so you can hear the LoRA's effect on both an
in-distribution prompt (harsh noise) and an out-of-distribution prompt
(neo-soul, copied verbatim from examples/text2music/example_76.json).

Usage:
    python noise_finetune/inference_compare.py \\
        --lora-path lora_output/merzbow/final

    # Baseline without LoRA (same prompts/seeds, base turbo only)
    python noise_finetune/inference_compare.py --no-lora \\
        --output-dir noise_finetune/inference_output_base
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acestep.handler import AceStepHandler


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
    "duration": 60.0,  # truncated from the example's 218s for quick A/B
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
    arr = audio_tensor.detach().cpu().float().numpy().T  # [samples, channels]
    sf.write(str(output_path), arr, sample_rate)
    secs = audio_tensor.shape[1] / sample_rate
    print(f"  wrote {output_path}  ({secs:.1f}s @ {sample_rate} Hz)")


def generate_one(handler: AceStepHandler, label: str, prompt: dict, out_dir: Path, seed: int) -> bool:
    print(f"\n=== Generating: {label} ===")
    print(f"  caption: {prompt['caption'][:80]}...")
    result = handler.generate_music(
        captions=prompt["caption"],
        lyrics=prompt["lyrics"],
        bpm=prompt["bpm"],
        key_scale=prompt["keyscale"],
        time_signature=prompt["timesignature"],
        vocal_language=prompt["vocal_language"],
        audio_duration=prompt["duration"],
        inference_steps=8,   # turbo
        guidance_scale=7.0,
        shift=3.0,           # turbo timestep shift; matches training
        use_random_seed=False,
        seed=seed,
    )
    if not result.get("success", False):
        print(f"  FAILED: {result.get('error', result.get('status_message', '?'))}")
        return False
    audios = result.get("audios", [])
    if not audios:
        print("  FAILED: no audio in result payload")
        return False
    for i, audio in enumerate(audios):
        suffix = "" if i == 0 else f"_{i}"
        save_audio(audio["tensor"], audio["sample_rate"], out_dir / f"{label}{suffix}.wav")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-path", default="lora_output/merzbow/final",
                    help="Path to LoRA adapter dir (PEFT format). Ignored with --no-lora.")
    ap.add_argument("--output-dir", default="noise_finetune/inference_output",
                    help="Where to write the .wav files")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lora", action="store_true",
                    help="Skip LoRA entirely (run base turbo as a baseline)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading base DiT (turbo)...")
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path="acestep-v15-turbo",
        device="auto",
    )
    if not ok:
        print(f"DiT init failed: {status}")
        return 1

    if not args.no_lora:
        print(f"\n[2/3] Loading LoRA: {args.lora_path}")
        lora_status = handler.add_lora(args.lora_path, adapter_name="merzbow")
        print(f"  {lora_status}")
        if lora_status.startswith("❌"):
            return 1
        handler.set_use_lora(True)
        handler.set_active_lora_adapter("merzbow")
        handler.set_lora_scale(args.lora_scale)
        print(f"  active (scale={args.lora_scale})")
    else:
        print("\n[2/3] Skipping LoRA (--no-lora baseline)")

    print("\n[3/3] Generating...")
    ok_noise = generate_one(handler, "noise_track", NOISE_PROMPT, out_dir, seed=args.seed)
    ok_neo = generate_one(handler, "neo_soul_track", NEO_SOUL_PROMPT, out_dir, seed=args.seed)

    print(f"\nDone. Outputs in {out_dir}/")
    return 0 if (ok_noise and ok_neo) else 1


if __name__ == "__main__":
    sys.exit(main())
