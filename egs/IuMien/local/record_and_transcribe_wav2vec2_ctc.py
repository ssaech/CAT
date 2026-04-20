#!/usr/bin/env python3
import argparse
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio.transforms as T
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "Missing dependency: sounddevice\n"
        "Install it with:\n"
        "  /Users/web/CAT/.venv/bin/python -m pip install sounddevice\n"
        "Then run this script again."
    ) from exc


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


def write_wav(path: Path, pcm_chunks: List[bytes], sample_rate: int, channels: int, sample_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        for chunk in pcm_chunks:
            wf.writeframes(chunk)


def load_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
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


def load_local_model_bundle(checkpoint_dir: Path) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
    processor_dir = resolve_processor_dir(checkpoint_dir)
    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(checkpoint_dir))
    return processor, model


def load_hf_model_bundle(model_id: str, target_lang: Optional[str] = None) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
    if model_id == "facebook/mms-300m":
        raise RuntimeError(
            "facebook/mms-300m is a self-supervised pretrained backbone, not a direct ASR checkpoint. "
            "It does not produce meaningful speech-to-text output by itself. "
            "Use your fine-tuned local checkpoint for transcription, or use an MMS ASR checkpoint such as "
            "facebook/mms-1b-all with --base-model-target-lang."
        )

    processor_kwargs = {}
    model_kwargs = {}
    if target_lang:
        processor_kwargs["target_lang"] = target_lang
        model_kwargs["target_lang"] = target_lang
        model_kwargs["ignore_mismatched_sizes"] = True

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, **model_kwargs)
    return processor, model


def transcribe_wav(audio_path: Path, processor: Wav2Vec2Processor, model: Wav2Vec2ForCTC) -> str:
    model.to("cpu")
    model.eval()

    waveform, sample_rate = load_wav_mono(audio_path)
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    with torch.no_grad():
        inputs = processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        )
        logits = model(input_values=inputs.input_values.to("cpu")).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def record_until_enter(output_path: Path, sample_rate: int, channels: int, input_device: Optional[str]) -> None:
    pcm_chunks: List[bytes] = []

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        pcm_chunks.append(bytes(indata))

    input("Press Enter to start recording.")
    print("Recording... press Enter again to stop.")

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=input_device,
        callback=callback,
    )

    with stream:
        input()

    if not pcm_chunks:
        raise RuntimeError("No audio was captured from the microphone.")

    write_wav(output_path, pcm_chunks, sample_rate=sample_rate, channels=channels, sample_width=2)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    recipe_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Wait for Enter, record from the mic until Enter is pressed again, save test.wav, then transcribe it."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(recipe_dir / "exp_cpu" / "iumien_mms300m_debug" / "best_model"),
        help="Path to the fine-tuned checkpoint directory. Defaults to the best local mms-300m run.",
    )
    parser.add_argument(
        "--base-model",
        default="facebook/mms-300m",
        help="Optional Hugging Face comparison model. Default is facebook/mms-300m, which will now be explained as non-ASR.",
    )
    parser.add_argument(
        "--base-model-target-lang",
        default=None,
        help="Optional MMS ASR target language code, for example eng or fra, when using a multilingual ASR checkpoint.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone recording sample rate. The saved WAV will use this rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of input channels to record. Defaults to mono.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional input device name or index for sounddevice.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Checkpoint directory not found: {checkpoint_dir}")

    output_path = script_dir / "test.wav"

    print(f"WAV output: {output_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Base model: {args.base_model}")
    if args.base_model_target_lang:
        print(f"Base model target language: {args.base_model_target_lang}")

    record_until_enter(
        output_path=output_path,
        sample_rate=args.sample_rate,
        channels=args.channels,
        input_device=args.device,
    )

    print(f"Saved recording to: {output_path}")
    print("Running transcription with the fine-tuned checkpoint...")
    finetuned_processor, finetuned_model = load_local_model_bundle(checkpoint_dir)
    finetuned_text = transcribe_wav(output_path, finetuned_processor, finetuned_model)
    print(f'Fine-tuned transcription: "{finetuned_text}"')

    print("Running transcription with the requested Hugging Face comparison model...")
    print("The first run may download model files from Hugging Face.")
    try:
        base_processor, base_model = load_hf_model_bundle(args.base_model, args.base_model_target_lang)
        base_text = transcribe_wav(output_path, base_processor, base_model)
        print(f'Base-model transcription: "{base_text}"')
    except Exception as exc:
        print(f"Base-model transcription failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
