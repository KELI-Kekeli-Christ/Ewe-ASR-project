#!/usr/bin/env python3
"""Compute char-ngram and word cosine similarity for AfriHuBERT test predictions.

Parses SpeechBrain wer_test.txt alignments and reuses eval_v2.py metrics so
results are comparable to Whisper eval_v2 outputs.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
import sys
from pathlib import Path

import jiwer

WHISPER_ROOT = Path("/home/data/projets-aps/projet6/fine_tuning_whisper_ewe")
AFRIHUBERT_ROOT = Path("/home/data/projets-aps/projet6/fine_tuning_afrihubert_ewe")
DEFAULT_WER = AFRIHUBERT_ROOT / "results/AfriHuBERT_CTC_Ewe_mina_v3/wer_test.txt"
DEFAULT_OUT_DIR = WHISPER_ROOT / "eval_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wer-path", type=Path, default=DEFAULT_WER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default="AfriHuBERT_CTC_Ewe_mina_v3")
    parser.add_argument("--n-samples", type=int, default=1443)
    return parser.parse_args()


def parse_wer_test(path: Path) -> list[tuple[str, str, str]]:
    content = path.read_text(encoding="utf-8")
    blocks = [
        b
        for b in content.split("=" * 80)
        if "ewe_test_" in b
    ]
    pairs: list[tuple[str, str, str]] = []
    for block in blocks:
        match = re.search(r"(ewe_test_\d+), %WER", block)
        if not match:
            continue
        uid = match.group(1)
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        ref_line = hyp_line = None
        for i, line in enumerate(lines):
            if ";" in line and ref_line is None:
                ref_line = line
                if i + 2 < len(lines):
                    hyp_line = lines[i + 2]
                break
        if not ref_line or not hyp_line:
            continue
        ref = " ".join(part.strip() for part in ref_line.split(";") if part.strip())
        hyp = " ".join(part.strip() for part in hyp_line.split(";") if part.strip())
        pairs.append((uid, ref, hyp))
    return pairs


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(WHISPER_ROOT))
    from eval_v2 import (  # noqa: WPS433
        char_ngram_cosine,
        normalize,
        normalize_no_punct,
        normalize_no_punct_no_accent,
        word_cosine,
    )

    pairs = parse_wer_test(args.wer_path)
    if not pairs:
        print(f"No utterances parsed from {args.wer_path}", file=sys.stderr)
        return 1

    rows = []
    for uid, ref, hyp in pairs:
        r0, h0 = normalize(ref), normalize(hyp)
        rnp, hnp = normalize_no_punct(ref), normalize_no_punct(hyp)
        rb, hb = normalize_no_punct_no_accent(ref), normalize_no_punct_no_accent(hyp)
        rows.append(
            {
                "utterance_id": uid,
                "reference": ref,
                "hypothesis": hyp,
                "wer": jiwer.wer(r0, h0) if r0 else 0.0,
                "cer": jiwer.cer(r0, h0) if r0 else 0.0,
                "char_ngram_cosine": char_ngram_cosine(r0, h0),
                "word_cosine": word_cosine(r0, h0),
                "wer_no_punct": jiwer.wer(rnp, hnp) if rnp else 0.0,
                "cer_no_punct": jiwer.cer(rnp, hnp) if rnp else 0.0,
                "char_ngram_cosine_no_punct": char_ngram_cosine(rnp, hnp),
                "word_cosine_no_punct": word_cosine(rnp, hnp),
                "wer_bare": jiwer.wer(rb, hb) if rb else 0.0,
                "cer_bare": jiwer.cer(rb, hb) if rb else 0.0,
                "char_ngram_cosine_bare": char_ngram_cosine(rb, hb),
                "word_cosine_bare": word_cosine(rb, hb),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"predictions_{args.model_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    refs = [normalize(ref) for _, ref, _ in pairs]
    hyps = [normalize(hyp) for _, _, hyp in pairs]
    summary = {
        "model": args.model_name,
        "n_samples": args.n_samples,
        "n_evaluated": len(rows),
        "wer_%": round(jiwer.wer(refs, hyps) * 100, 2),
        "cer_%": round(jiwer.cer(refs, hyps) * 100, 2),
        "mean_char_ngram_cosine": round(st.mean(r["char_ngram_cosine"] for r in rows), 4),
        "mean_word_cosine": round(st.mean(r["word_cosine"] for r in rows), 4),
        "mean_char_ngram_cosine_no_punct": round(
            st.mean(r["char_ngram_cosine_no_punct"] for r in rows), 4
        ),
        "mean_word_cosine_no_punct": round(
            st.mean(r["word_cosine_no_punct"] for r in rows), 4
        ),
        "mean_char_ngram_cosine_bare": round(
            st.mean(r["char_ngram_cosine_bare"] for r in rows), 4
        ),
        "mean_word_cosine_bare": round(st.mean(r["word_cosine_bare"] for r in rows), 4),
    }

    print("SUMMARY", summary)
    print("CSV", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
