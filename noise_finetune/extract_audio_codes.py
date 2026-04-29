#!/usr/bin/env python3
"""
Extract 5Hz audio codes for every mp3 in the Merzbow captions dataset.

Reads:
  merzbow/captioning/merzbow_dataset.json

Writes (same schema + new "audio_codes" field per sample):
  merzbow/captioning/merzbow_dataset_with_codes.json

Resumable: samples whose "audio_codes" is already populated are skipped.

Reuses AceStepHandler.convert_src_audio_to_codes() (which loads DiT + VAE +
turbo tokenizer) exactly the way merzbow/caption_audio.py does.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("ACESTEP_PROJECT_ROOT", str(PROJECT_ROOT))

from loguru import logger


CODE_RE = re.compile(r"<\|audio_code_(\d+)\|>")


def count_codes(codes_str: str) -> int:
    return len(CODE_RE.findall(codes_str))


def trim_codes(codes_str: str, target: int) -> str:
    """Trim a run of <|audio_code_N|> tokens to exactly `target` codes."""
    matches = list(CODE_RE.finditer(codes_str))
    if len(matches) <= target:
        return codes_str
    cutoff = matches[target].start()
    return codes_str[:cutoff]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "merzbow" / "captioning" / "merzbow_dataset.json"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "merzbow" / "captioning" / "merzbow_dataset_with_codes.json"),
    )
    parser.add_argument("--dit-config", default="acestep-v15-turbo")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.9,
        help="Skip sample if extracted codes < min_ratio * (duration * 5). Default 0.9.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N samples (debug)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    dataset_dir = dataset_path.parent  # audio_path fields are relative to this

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Load prior output if resuming, so we keep already-extracted codes.
    prior_codes: dict[str, str] = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            for s in prior.get("samples", []):
                if s.get("audio_codes"):
                    prior_codes[s["id"]] = s["audio_codes"]
            logger.info(f"Resume: found {len(prior_codes)} previously-extracted samples.")
        except Exception as e:
            logger.warning(f"Could not read prior output {output_path}: {e}")

    # --- Initialize AceStepHandler (loads DiT, VAE, turbo audio tokenizer) ---
    logger.info("Initializing AceStepHandler (DiT + VAE + turbo)...")
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    status_msg, ok = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path=args.dit_config,
        device=args.device,
    )
    if not ok:
        logger.error(f"Handler init failed: {status_msg}")
        sys.exit(1)
    logger.info(f"Handler ready: {status_msg}")

    samples = dataset.get("samples", [])
    if args.limit is not None:
        samples = samples[: args.limit]

    out_samples: list[dict] = []
    skipped: list[tuple[str, str]] = []  # (id, reason)

    for i, sample in enumerate(samples, 1):
        sid = sample["id"]
        duration = int(sample.get("duration") or 0)
        target_codes = duration * 5

        if sid in prior_codes:
            sample = {**sample, "audio_codes": prior_codes[sid]}
            out_samples.append(sample)
            logger.info(f"[{i}/{len(samples)}] SKIP (already have codes): {sid}")
            continue

        audio_path = sample["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = str(dataset_dir / audio_path)

        if not os.path.exists(audio_path):
            logger.warning(f"[{i}/{len(samples)}] Audio missing, skipping: {audio_path}")
            skipped.append((sid, "audio_missing"))
            continue

        logger.info(f"[{i}/{len(samples)}] Encoding ({duration}s → ~{target_codes} codes): {sid}")
        codes_str = handler.convert_src_audio_to_codes(audio_path)

        if isinstance(codes_str, str) and codes_str.startswith("❌"):
            logger.warning(f"  extraction failed: {codes_str}")
            skipped.append((sid, "extraction_failed"))
            continue

        actual = count_codes(codes_str)
        if actual == 0:
            logger.warning(f"  no codes produced, skipping")
            skipped.append((sid, "zero_codes"))
            continue

        if target_codes > 0 and actual > target_codes:
            logger.info(f"  trimming {actual} → {target_codes}")
            codes_str = trim_codes(codes_str, target_codes)
            actual = target_codes

        if target_codes > 0 and actual < args.min_ratio * target_codes:
            logger.warning(
                f"  got {actual} codes, expected {target_codes} "
                f"({actual / target_codes:.0%} of target) — skipping (do NOT pad)"
            )
            skipped.append((sid, f"too_few_codes_{actual}_of_{target_codes}"))
            continue

        out_sample = {**sample, "audio_codes": codes_str, "audio_codes_count": actual}
        out_samples.append(out_sample)

        # Incremental save every 10 samples so a crash keeps progress.
        if i % 10 == 0:
            _save(output_path, dataset, out_samples)

    _save(output_path, dataset, out_samples)

    logger.info("=" * 60)
    logger.info(f"Wrote {len(out_samples)} samples with codes to {output_path}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} samples:")
        for sid, reason in skipped:
            logger.warning(f"  {reason}: {sid}")


def _save(output_path: Path, dataset: dict, out_samples: list[dict]) -> None:
    out = {
        "metadata": {**dataset.get("metadata", {}), "num_samples": len(out_samples)},
        "samples": out_samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
