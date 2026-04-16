#!/usr/bin/env python3
import argparse
import json
import os
import wave
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def collect_audio_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    audio_exts = {".wav"}
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in audio_exts)


def resolve_processor_dir(load_from: Path) -> Path:
    candidates = []
    if load_from.is_dir():
        candidates.append(load_from)

    if load_from.name == "best_model":
        candidates.append(load_from.parent / "best_model_processor")

    candidates.extend(
        [
            load_from.parent / "best_model_processor",
            load_from.parent / "processor",
            load_from.parent.parent / "best_model_processor",
            load_from.parent.parent / "processor",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "tokenizer_config.json").exists():
            return candidate

    raise RuntimeError(
        f"Could not find a saved processor for checkpoint: {load_from}. "
        "Expected a processor directory such as best_model_processor or processor nearby."
    )


def load_wav_mono(path: Path) -> Tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
        raw_buffer = memoryview(bytearray(raw))

    if sample_width == 2:
        audio = torch.frombuffer(raw_buffer, dtype=torch.int16).float()
        audio = audio / float(torch.iinfo(torch.int16).max)
    elif sample_width == 4:
        audio = torch.frombuffer(raw_buffer, dtype=torch.int32).float()
        audio = audio / float(torch.iinfo(torch.int32).max)
    elif sample_width == 1:
        audio = torch.frombuffer(raw_buffer, dtype=torch.uint8).float()
        audio = (audio - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes for {path}")

    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)
    return audio.clone(), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Wav2Vec2 CTC transcription on CPU.")
    parser.add_argument("input", help="Path to a WAV file or a directory of WAV files.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved local model directory, for example exp_cpu/.../best_model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path. Defaults to stdout only.",
    )
    args = parser.parse_args()

    audio_files = collect_audio_files(args.input)
    if not audio_files:
        raise RuntimeError(f"No WAV files found under: {args.input}")

    checkpoint_dir = Path(args.checkpoint).expanduser()
    processor_dir = resolve_processor_dir(checkpoint_dir)

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(checkpoint_dir))
    model.to("cpu")
    model.eval()

    resamplers = {}
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = open(args.output, "w", encoding="utf-8")

    try:
        with torch.no_grad():
            for audio_path in audio_files:
                waveform, sample_rate = load_wav_mono(audio_path)
                if sample_rate != 16000:
                    if sample_rate not in resamplers:
                        resamplers[sample_rate] = T.Resample(sample_rate, 16000)
                    waveform = resamplers[sample_rate](waveform.unsqueeze(0)).squeeze(0)

                inputs = processor(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=False,
                )
                logits = model(input_values=inputs.input_values.to("cpu")).logits
                pred_ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

                row = {
                    "audio": str(audio_path),
                    "checkpoint": str(checkpoint_dir),
                    "text": text,
                }
                print(json.dumps(row, ensure_ascii=False))
                if writer:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
