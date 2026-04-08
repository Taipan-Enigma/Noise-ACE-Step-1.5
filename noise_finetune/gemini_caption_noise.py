"""Gemini-based captioning for noise/experimental music.

Processes mp3 files in the data/ folder, sends them to Gemini for audio analysis,
and outputs an ACE-Step dataset JSON ready for training acestep-5Hz-lm.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python gemini_caption_noise.py [--input-dir ./data] [--output ./dataset.json] [--model gemini-2.5-flash] [--resume]
    python gemini_caption_noise.py --file track.mp3

All tracks are treated as instrumental with no lyrics.
"""

import os
import sys
import json
import time
import argparse
import base64
import mimetypes
from pathlib import Path
from datetime import datetime

import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "merzbow" / "captioning" / "data"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "merzbow" / "captioning" / "merzbow_dataset.json"

NOISE_CAPTION_PROMPT = """\
You are an expert music analyst specializing in experimental, noise, and industrial music.

Analyze this audio track and provide a detailed caption describing:
- The overall sonic character and texture (e.g. harsh noise wall, feedback, drone, static, distortion, granular, etc.)
- Frequency content and tonal qualities (e.g. low rumble, mid-range screech, high-frequency hiss, full-spectrum)
- Dynamics and intensity (e.g. relentless, building, pulsing, sudden bursts, sustained wall)
- Any identifiable sound sources or techniques (e.g. feedback loops, contact microphones, effect pedals, tape manipulation, electronics)
- Mood and atmosphere (e.g. oppressive, meditative, chaotic, confrontational, hypnotic)
- Structure and progression (e.g. static/unchanging, gradually evolving, abrupt shifts, layered buildup)
- Tempo/rhythm if any is present, or note the absence of conventional rhythm

This is instrumental noise/experimental music. There are NO lyrics.

Respond in JSON format:
{
    "caption": "<A detailed 2-4 sentence description suitable for training a music generation model. Be specific about textures and sonic qualities, not vague.>",
    "genre": "<Primary genre tag, e.g. 'harsh noise', 'noise', 'harsh noise wall', 'power electronics', 'industrial noise', 'drone noise', 'experimental noise'>",
    "bpm": <integer BPM if rhythm is detectable, otherwise null>,
    "keyscale": "<musical key if tonal center exists, e.g. 'C minor', otherwise null>",
    "timesignature": "<time signature if detectable, e.g. '4', otherwise null>",
    "mood": "<1-3 word mood descriptor>"
}
"""


def encode_audio_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "audio/mp3"


def get_duration_seconds(file_path: str) -> int:
    """Get audio duration using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", file_path],
            capture_output=True, text=True
        )
        return int(float(result.stdout.strip()))
    except Exception:
        return 0


# Pricing per 1M tokens (USD) — source: https://ai.google.dev/gemini-api/docs/pricing
# Audio input is priced differently from text input.
# Thinking tokens are billed at the output rate (no separate thinking price).
# Since we send audio, input is priced at the audio rate (conservative estimate).
MODEL_PRICING = {
    "gemini-2.5-flash": {"input_audio": 1.00, "input_text": 0.30, "output": 2.50},
    "gemini-2.5-pro":   {"input_audio": 1.25, "input_text": 1.25, "output": 10.00},
    "gemini-2.0-flash":  {"input_audio": 0.10, "input_text": 0.10, "output": 0.40},
}


class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    def add(self, usage: dict, model: str):
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        # Use audio input rate since the bulk of input tokens are audio
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemini-2.5-flash"])
        call_cost = (
            input_tokens * pricing["input_audio"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        self.total_cost += call_cost
        return call_cost

    def summary(self) -> str:
        return (
            f"Tokens — input: {self.total_input_tokens:,}  "
            f"output: {self.total_output_tokens:,}\n"
            f"API calls: {self.call_count}  |  "
            f"Estimated total cost: ${self.total_cost:.4f}"
        )


cost_tracker = CostTracker()


def caption_audio(api_key: str, audio_path: str, model: str, base_url: str) -> dict:
    """Send audio to Gemini and get structured caption."""
    mime_type = get_mime_type(audio_path)
    audio_data = encode_audio_base64(audio_path)

    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": NOISE_CAPTION_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": audio_data}},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }

    endpoint = f"{base_url}/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    response = requests.post(endpoint, headers=headers, json=body, timeout=300)

    if response.status_code == 429:
        raise RateLimitError("Rate limited by Gemini API")

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text[:300]}")

    result = response.json()

    # Track cost from usage metadata
    usage = result.get("usageMetadata", {})
    call_cost = cost_tracker.add(usage, model)

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    text = text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(text)
    parsed["_cost"] = call_cost
    return parsed


class RateLimitError(Exception):
    pass


def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {}


def save_progress(progress: dict, progress_file: Path):
    progress_file.write_text(json.dumps(progress, indent=2))


def build_dataset(samples: list) -> dict:
    """Build full ACE-Step dataset JSON."""
    return {
        "metadata": {
            "name": "merzbow_noise",
            "custom_tag": "merzbow",
            "tag_position": "prepend",
            "created_at": datetime.now().isoformat(),
            "num_samples": len(samples),
            "all_instrumental": True,
            "genre_ratio": 0,
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Caption noise music with Gemini")
    parser.add_argument("--file", type=str, help="Caption a single mp3 file instead of the whole folder")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory of mp3 files to caption")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output dataset JSON path")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Single file mode
    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"File not found: {audio_path}")
            sys.exit(1)
        print(f"Captioning single file: {audio_path.name}")
        result = caption_audio(api_key, str(audio_path), args.model, args.base_url)
        duration = get_duration_seconds(str(audio_path))
        print(f"Genre: {result.get('genre', '?')}  (${result.get('_cost', 0):.4f})")
        print(f"Caption: {result.get('caption', '')}")
        print(f"BPM: {result.get('bpm')}  Key: {result.get('keyscale')}  Time sig: {result.get('timesignature')}")
        print(f"Mood: {result.get('mood', '')}")
        print(f"Duration: {duration}s")
        print(f"\n{cost_tracker.summary()}")
        return

    input_dir = Path(args.input_dir)
    output_json = Path(args.output)
    progress_file = output_json.parent / f".caption_progress_{output_json.stem}.json"

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Compute relative path from output JSON to input dir for audio_path references
    try:
        rel_data_dir = input_dir.resolve().relative_to(output_json.parent.resolve())
    except ValueError:
        # Input dir is not under the output dir — use absolute path
        rel_data_dir = input_dir.resolve()

    audio_files = sorted(input_dir.glob("*.mp3"))
    if not audio_files:
        print(f"No mp3 files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(audio_files)} mp3 files in {input_dir}")
    print(f"Output: {output_json}")

    # Load previous progress if resuming
    progress = load_progress(progress_file) if args.resume else {}
    print(f"Already captioned: {len(progress)} files" if progress else "Starting fresh")

    samples = []
    errors = []

    for i, audio_file in enumerate(audio_files, 1):
        filename = audio_file.name
        stem = audio_file.stem

        # Skip already-captioned files
        if filename in progress:
            samples.append(progress[filename])
            continue

        print(f"[{i}/{len(audio_files)}] {filename} ... ", end="", flush=True)

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                result = caption_audio(api_key, str(audio_file), args.model, args.base_url)
                duration = get_duration_seconds(str(audio_file))

                sample = {
                    "id": stem,
                    "audio_path": str(rel_data_dir / filename),
                    "filename": filename,
                    "caption": result.get("caption", ""),
                    "genre": result.get("genre", "noise"),
                    "lyrics": "[Instrumental]",
                    "raw_lyrics": "",
                    "formatted_lyrics": "[Instrumental]",
                    "bpm": result.get("bpm"),
                    "keyscale": result.get("keyscale") or "",
                    "timesignature": result.get("timesignature") or "",
                    "duration": duration,
                    "language": "instrumental",
                    "is_instrumental": True,
                    "custom_tag": "merzbow",
                    "labeled": True,
                    "prompt_override": None,
                }

                samples.append(sample)
                progress[filename] = sample
                save_progress(progress, progress_file)

                # Update dataset JSON after each successful call
                dataset = build_dataset(samples)
                output_json.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

                print(f"OK — {result.get('genre', '?')} (call: ${result.get('_cost', 0):.4f} | total: ${cost_tracker.total_cost:.4f})")
                print(f"   Caption: {result.get('caption', '')[:120]}...")
                break

            except RateLimitError:
                wait = 30 * (retries + 1)
                print(f"rate limited, waiting {wait}s ... ", end="", flush=True)
                time.sleep(wait)
                retries += 1

            except Exception as e:
                print(f"ERROR: {e}")
                errors.append((filename, str(e)))
                break

        if retries >= max_retries:
            print("FAILED (max retries)")
            errors.append((filename, "max retries exceeded"))

        time.sleep(args.delay)

    # Final save
    dataset = build_dataset(samples)
    output_json.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"\nDataset saved to {output_json}")
    print(f"Total samples: {len(samples)}")
    print(f"\n{cost_tracker.summary()}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for fname, err in errors:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
