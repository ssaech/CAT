#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from transformers import pipeline


def collect_audio_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in audio_exts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Whisper transcription on CPU.")
    parser.add_argument("input", help="Path to an audio file or a directory of audio files.")
    parser.add_argument(
        "--model",
        default="openai/whisper-small",
        help="Whisper model id. Good CPU choices: openai/whisper-tiny, openai/whisper-base, openai/whisper-small",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint, for example 'en' or 'zh'. Leave unset for auto-detection.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task. Use transcribe for same-language ASR.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Chunk length in seconds for long-form audio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used inside the ASR pipeline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path. Defaults to stdout only.",
    )
    args = parser.parse_args()

    audio_files = collect_audio_files(args.input)
    if not audio_files:
        raise RuntimeError(f"No audio files found under: {args.input}")

    generate_kwargs = {"task": args.task}
    if args.language:
        generate_kwargs["language"] = args.language

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size,
        device="cpu",
        torch_dtype=torch.float32,
    )

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = open(args.output, "w", encoding="utf-8")

    try:
        for audio_path in audio_files:
            result = asr(str(audio_path), generate_kwargs=generate_kwargs)
            row = {
                "audio": str(audio_path),
                "text": result["text"].strip(),
            }
            print(json.dumps(row, ensure_ascii=False))
            if writer:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
