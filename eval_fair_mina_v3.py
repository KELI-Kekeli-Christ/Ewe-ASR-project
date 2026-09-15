#!/usr/bin/env python3
"""Fair WER/CER comparison Whisper vs AfriHuBERT on mina_v3 test.
Identical text normalization: NFC + strip + lower.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import unicodedata
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000


def normalize(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text).strip().lower())


def read_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq
        parts = [b.to_pandas() for b in pq.ParquetFile(path).iter_batches(batch_size=64)]
        return pd.concat(parts, ignore_index=True)


def decode_audio(audio):
    if isinstance(audio, dict):
        raw = audio.get("bytes") or audio.get("array")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)):
            y, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        else:
            y = np.asarray(raw, dtype=np.float32)
            src = audio.get("sampling_rate", SR)
            if src != SR:
                y = librosa.resample(y, orig_sr=src, target_sr=SR)
    else:
        y, _ = librosa.load(io.BytesIO(bytes(audio)), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y


def score_pairs(refs, hyps):
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return {"n": 0, "wer": None, "cer": None}
    rs, hs = zip(*pairs)
    return {
        "n": len(pairs),
        "wer": round(jiwer.wer(list(rs), list(hs)) * 100, 4),
        "cer": round(jiwer.cer(list(rs), list(hs)) * 100, 4),
    }


def parse_afri_wer_file(path: Path):
    """Parse SpeechBrain wer_test.txt alignments into (ref, hyp) strings."""
    text = path.read_text(errors="replace")
    refs, hyps = [], []
    # SpeechBrain format often has blocks with <UTT> / lab / hyp lines
    # Fallback: lines "lab: ..." "hyp: ..."
    lab, hyp = None, None
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("lab:") or low.startswith("ref:"):
            lab = s.split(":", 1)[1].strip()
        elif low.startswith("hyp:"):
            hyp = s.split(":", 1)[1].strip()
            if lab is not None:
                refs.append(lab)
                hyps.append(hyp)
                lab, hyp = None, None
    return refs, hyps


def eval_whisper(model_dir: Path, test_parquet: Path, device: str):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    # keep yo conditioning consistent with training
    forced = processor.get_decoder_prompt_ids(language="yo", task="transcribe")
    model.config.forced_decoder_ids = forced

    df = read_parquet_safe(test_parquet)
    refs, hyps = [], []
    for i, row in df.iterrows():
        ref = row.get("transcription") or row.get("text") or ""
        try:
            y = decode_audio(row["audio"])
            if len(y) < 1600:
                continue
            feats = processor.feature_extractor(y, sampling_rate=SR, return_tensors="pt")
            input_features = feats.input_features.to(device)
            with torch.no_grad():
                ids = model.generate(input_features, max_length=250)
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception:
            continue
        refs.append(ref)
        hyps.append(hyp)
        if (len(refs) % 100) == 0:
            print(f"  whisper decoded {len(refs)}/{len(df)}", flush=True)
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-dir", type=Path, required=True)
    ap.add_argument("--test-parquet", type=Path, required=True)
    ap.add_argument("--afri-wer-file", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== Whisper decode ===", flush=True)
    w_refs, w_hyps = eval_whisper(args.whisper_dir, args.test_parquet, args.device)
    w_score = score_pairs(w_refs, w_hyps)
    print("Whisper", w_score, flush=True)

    rows = [{"model": "whisper_base_mina_v3", **w_score}]

    if args.afri_wer_file and args.afri_wer_file.exists():
        print("=== AfriHuBERT from wer_test.txt ===", flush=True)
        a_refs, a_hyps = parse_afri_wer_file(args.afri_wer_file)
        a_score = score_pairs(a_refs, a_hyps)
        print("AfriHuBERT", a_score, flush=True)
        rows.append({"model": "afrihubert_ctc_mina_v3", **a_score})
    else:
        print("AfriHuBERT wer file missing — skip (run after Afri train)", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n", "wer", "cer"])
        w.writeheader()
        w.writerows(rows)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
