#!/usr/bin/env python3
import argparse
import json
import os
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2ForCTC, pipeline

LANGUAGE_CONFIG = {
    "vi": {
        "mms": "vie",
        "whisper": "vietnamese",
    },
    "zh": {
        "mms": "cmn-script_simplified",
        "whisper": "chinese",
    },
    "yue": {
        "mms": "yue-script_traditional",
        "whisper": "cantonese",
    },
}


def collect_audio_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    audio_exts = {".wav"}
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in audio_exts)


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


def load_mms_bundle(model_id: str, target_lang: Optional[str], local_files_only: bool) -> Tuple[object, Wav2Vec2ForCTC]:
    processor_kwargs = {"local_files_only": local_files_only}
    model_kwargs = {"local_files_only": local_files_only}
    if target_lang:
        processor_kwargs["target_lang"] = target_lang
        model_kwargs["target_lang"] = target_lang
        model_kwargs["ignore_mismatched_sizes"] = True

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, **model_kwargs)
    model.to("cpu")
    model.eval()
    return processor, model


def transcribe_with_mms(
    audio_path: Path,
    processor: object,
    model: Wav2Vec2ForCTC,
    resamplers: Dict[int, T.Resample],
) -> str:
    waveform, sample_rate = load_wav_mono(audio_path)
    if sample_rate != 16000:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = T.Resample(sample_rate, 16000)
        waveform = resamplers[sample_rate](waveform.unsqueeze(0)).squeeze(0)

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


def build_whisper_asr(model_id: str, chunk_length: int, batch_size: int, local_files_only: bool):
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, local_files_only=local_files_only)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        device="cpu",
        torch_dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hugging Face MMS-1B and Whisper Turbo ASR on one WAV or a directory of WAVs."
    )
    parser.add_argument("input", help="Path to a WAV file or a directory of WAV files.")
    parser.add_argument(
        "--mms-model",
        default="facebook/mms-1b-all",
        help="Hugging Face MMS ASR model id to use.",
    )
    parser.add_argument(
        "--whisper-model",
        default="openai/whisper-large-v3-turbo",
        help="Whisper model id to use for comparison.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["vi", "zh", "yue"],
        help="Language set to run for both models. Defaults to: vi zh yue",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Whisper chunk length in seconds for long-form audio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Whisper internal batch size.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Whisper only from the local Hugging Face cache instead of trying the network.",
    )
    args = parser.parse_args()

    audio_files = collect_audio_files(args.input)
    if not audio_files:
        raise RuntimeError(f"No WAV files found under: {args.input}")

    requested_langs = []
    for lang in args.langs:
        if lang not in LANGUAGE_CONFIG:
            raise RuntimeError(
                f"Unsupported language key: {lang}. Supported keys are: {', '.join(sorted(LANGUAGE_CONFIG))}"
            )
        requested_langs.append(lang)

    print(f"Loading Whisper model: {args.whisper_model}")
    whisper_asr = build_whisper_asr(
        model_id=args.whisper_model,
        chunk_length=args.chunk_length,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )

    resamplers: Dict[int, T.Resample] = {}
    results: Dict[str, Dict[str, object]] = {
        str(audio_path): {
            "audio": str(audio_path),
            "mms_model": args.mms_model,
            "mms_text_by_lang": {},
            "whisper_model": args.whisper_model,
            "whisper_transcription_by_lang": {},
            "whisper_translation_by_lang": {},
        }
        for audio_path in audio_files
    }

    for lang in requested_langs:
        mms_target_lang = LANGUAGE_CONFIG[lang]["mms"]
        print(f"Loading MMS model: {args.mms_model} for {lang} ({mms_target_lang})")
        mms_processor, mms_model = load_mms_bundle(
            model_id=args.mms_model,
            target_lang=mms_target_lang,
            local_files_only=args.local_files_only,
        )
        try:
            for audio_path in audio_files:
                print(f"MMS {lang}: {audio_path}")
                mms_text = transcribe_with_mms(audio_path, mms_processor, mms_model, resamplers)
                results[str(audio_path)]["mms_text_by_lang"][lang] = mms_text
        finally:
            del mms_model
            del mms_processor

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = open(args.output, "w", encoding="utf-8")

    try:
        for audio_path in audio_files:
            row = results[str(audio_path)]
            print(f"Whisper: {audio_path}")
            for lang in requested_langs:
                whisper_language = LANGUAGE_CONFIG[lang]["whisper"]
                transcribe_result = whisper_asr(
                    str(audio_path),
                    generate_kwargs={"task": "transcribe", "language": whisper_language},
                )
                translate_result = whisper_asr(
                    str(audio_path),
                    generate_kwargs={"task": "translate", "language": whisper_language},
                )
                row["whisper_transcription_by_lang"][lang] = transcribe_result["text"].strip()
                row["whisper_translation_by_lang"][lang] = translate_result["text"].strip()

            print(json.dumps(row, ensure_ascii=False))
            if writer:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
