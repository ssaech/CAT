#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def read_split_file(path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        utt_id, text = line.split("\t", 1)
        utt_id = utt_id.strip()
        text = text.strip().lower()
        if utt_id and text:
            items.append((utt_id, text))
    return items


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


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_vocab(train_items: List[Tuple[str, str]]) -> Dict[str, int]:
    chars = sorted(set("".join(text for _, text in train_items)))
    vocab_chars = [c for c in chars if c != " "]
    vocab = {c: i for i, c in enumerate(vocab_chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def write_vocab(vocab: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


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


class AudioTextDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, str]],
        wav_dir: Path,
        processor: Wav2Vec2Processor,
        target_sample_rate: int = 16000,
    ) -> None:
        self.items = items
        self.wav_dir = wav_dir
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.resamplers: Dict[int, T.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        utt_id, text = self.items[idx]
        wav_path = self.wav_dir / f"{utt_id}.wav"
        waveform, sample_rate = load_wav_mono(wav_path)
        if sample_rate != self.target_sample_rate:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = T.Resample(sample_rate, self.target_sample_rate)
            waveform = self.resamplers[sample_rate](waveform.unsqueeze(0)).squeeze(0)

        text = normalize_text(text)
        inputs = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.target_sample_rate,
            return_attention_mask=False,
        )
        labels = self.processor.tokenizer(text, return_attention_mask=False).input_ids
        return {
            "utt_id": utt_id,
            "input_values": torch.tensor(inputs["input_values"][0], dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
        }


@dataclass
class CTCBatchCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            [{"input_values": x} for x in input_values],
            padding=True,
            return_tensors="pt",
        )
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": x} for x in labels],
            padding=True,
            return_tensors="pt",
        )
        padded_labels = label_batch["input_ids"].masked_fill(label_batch["attention_mask"].ne(1), -100)

        return {
            "input_values": batch["input_values"],
            "attention_mask": batch["attention_mask"],
            "labels": padded_labels,
        }


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp, start=1):
            if ref_tok == hyp_tok:
                curr.append(prev[j - 1])
            else:
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + 1))
        prev = curr
    return prev[-1]


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0
    for pred, ref in zip(preds, refs):
        total_char_err += edit_distance(list(ref), list(pred))
        total_chars += max(1, len(ref))
        total_word_err += edit_distance(ref.split(), pred.split())
        total_words += max(1, len(ref.split()))
    return {
        "cer": total_char_err / total_chars,
        "wer": total_word_err / total_words,
    }


def run_eval(
    model: Wav2Vec2ForCTC,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    num_samples: int = 0,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    pred_texts: List[str] = []
    ref_texts: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            pred_ids = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"].detach().cpu().clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id

            pred_texts.extend(processor.batch_decode(pred_ids, skip_special_tokens=True))
            ref_texts.extend(processor.batch_decode(labels, group_tokens=False, skip_special_tokens=True))

    norm_preds = [x.lower().strip() for x in pred_texts]
    norm_refs = [x.lower().strip() for x in ref_texts]
    metrics = compute_metrics(norm_preds, norm_refs)
    metrics["loss"] = total_loss / max(1, num_batches)
    metrics["empty_pred_frac"] = sum(1 for x in norm_preds if not x) / max(1, len(norm_preds))
    if num_samples > 0:
        metrics["samples"] = list(zip(norm_refs[:num_samples], norm_preds[:num_samples]))
    return metrics


def maybe_subset(items: List[Tuple[str, str]], limit: int) -> List[Tuple[str, str]]:
    if limit <= 0 or limit >= len(items):
        return items
    return items[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CPU-friendly Wav2Vec2 CTC baseline for Iu Mien.")
    parser.add_argument("--data-root", default="/Users/web/MightLJSpeech/MightLJSpeech-1.1")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--save-dir", default="exp_cpu/wav2vec2_ctc")
    parser.add_argument(
        "--load-from",
        default="",
        help="Load weights from a saved local checkpoint directory instead of starting from --model.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only. Requires --load-from so the model and processor can be restored.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-dev", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-samples", type=int, default=5)
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument(
        "--enable-spec-augment",
        action="store_true",
        help="Opt in to Wav2Vec2 time/feature masking during training. Disabled by default for CPU debugging.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    data_root = Path(args.data_root)
    wav_dir = data_root / "wavs"
    train_items = maybe_subset(read_split_file(data_root / "train_data.txt"), args.max_train)
    dev_items = maybe_subset(read_split_file(data_root / "dev_data.txt"), args.max_dev)
    load_from = Path(args.load_from).expanduser() if args.load_from else None

    save_dir = Path(args.save_dir)
    if not args.eval_only:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only and load_from is None:
        raise RuntimeError("--eval-only requires --load-from so a saved model and processor can be restored.")

    if load_from is not None:
        processor_dir = resolve_processor_dir(load_from)
        processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
        vocab_size = processor.tokenizer.vocab_size
    else:
        vocab = build_vocab(train_items)
        vocab_size = len(vocab)
        vocab_path = save_dir / "vocab.json"
        write_vocab(vocab, vocab_path)

        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_path),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            do_lower_case=False,
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(str(save_dir / "processor"))
    if load_from is not None and not args.eval_only:
        processor.save_pretrained(str(save_dir / "processor"))

    train_ds = AudioTextDataset(train_items, wav_dir, processor)
    dev_ds = AudioTextDataset(dev_items, wav_dir, processor)
    collator = CTCBatchCollator(processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if load_from is not None:
        model = Wav2Vec2ForCTC.from_pretrained(str(load_from))
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model,
            vocab_size=vocab_size,
            pad_token_id=processor.tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            ignore_mismatched_sizes=True,
        )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if not args.enable_spec_augment:
        model.config.apply_spec_augment = False
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0

    device = torch.device("cpu")
    model.to(device)

    if args.eval_only:
        dev_metrics = run_eval(model, dev_loader, processor, device, num_samples=args.show_samples)
        record = {
            "split": "dev",
            "dev_loss": dev_metrics["loss"],
            "dev_cer": dev_metrics["cer"],
            "dev_wer": dev_metrics["wer"],
            "dev_empty_pred_frac": dev_metrics["empty_pred_frac"],
        }
        print(json.dumps(record, ensure_ascii=False))
        for i, (ref_text, pred_text) in enumerate(dev_metrics.get("samples", []), start=1):
            print(f"[sample {i}] REF: {ref_text}")
            print(f"[sample {i}] HYP: {pred_text}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_cer = math.inf
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError(
                    "Non-finite training loss encountered. "
                    "Try a fresh save dir, keep spec augment disabled, and lower --lr further."
                )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += outputs.loss.item()
            steps += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        train_loss = running_loss / max(1, steps)
        dev_metrics = run_eval(model, dev_loader, processor, device, num_samples=args.show_samples)
        if not math.isfinite(train_loss) or not math.isfinite(dev_metrics["loss"]):
            raise RuntimeError("Non-finite epoch metrics encountered; aborting before saving a broken checkpoint.")
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_cer": dev_metrics["cer"],
            "dev_wer": dev_metrics["wer"],
            "dev_empty_pred_frac": dev_metrics["empty_pred_frac"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        for i, (ref_text, pred_text) in enumerate(dev_metrics.get("samples", []), start=1):
            print(f"[sample {i}] REF: {ref_text}")
            print(f"[sample {i}] HYP: {pred_text}")

        if dev_metrics["cer"] < best_cer:
            best_cer = dev_metrics["cer"]
            model.save_pretrained(str(save_dir / "best_model"))
            processor.save_pretrained(str(save_dir / "best_model_processor"))

    (save_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
