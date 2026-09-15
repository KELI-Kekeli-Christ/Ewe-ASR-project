#!/usr/bin/env python3
"""
Évaluation WER / CER pour un ou plusieurs checkpoints Whisper fine-tunés sur l'ewe.

Usage (un seul modèle) :
    python eval_wer_cer.py \
        --model whisper_medium_ewe_baseline_final \
        --test-parquet /home/data/projets-aps/projet6/data/ewe_test.parquet

Usage (comparaison multi-modèles) :
    python eval_wer_cer.py \
        --model whisper_medium_ewe_baseline_final \
        --model whisper_medium_ewe_train_only_final \
        --model whisper_medium_ewe_augmented_final \
        --model whisper-medium-ewe-mwer-baseline \
        --test-parquet /home/data/projets-aps/projet6/data/ewe_test.parquet \
        --out-csv results_comparison.csv

Le script :
  - charge le processor depuis le checkpoint (fallback sur --base-model si absent) ;
  - décode le test set par batch (feature extractor -> generate -> tokenizer.decode) ;
  - calcule WER/CER globaux (jiwer) + WER/CER par échantillon (pour analyse d'erreurs) ;
  - sauvegarde les prédictions détaillées en CSV (un fichier par modèle) et un
    tableau comparatif si plusieurs modèles sont passés.
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd
import torch

try:
    import jiwer
except ImportError:
    sys.exit("jiwer manquant. Installer avec : pip install jiwer --break-system-packages")

from transformers import WhisperForConditionalGeneration, WhisperProcessor


# --------------------------------------------------------------------------- #
# Normalisation texte
# --------------------------------------------------------------------------- #
# Normalisation volontairement légère : on NE touche PAS aux diacritiques
# tonals (essentiels en ewe). On retire juste ponctuation forte et espaces
# multiples, on met en minuscule. À adapter si un jour un tokenizer tonal
# dédié (cf. TER) doit consommer un format différent.
_PUNCT_RE = re.compile(r"[.,!?;:\"«»()\[\]{}…]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    models: List[str]
    test_parquet: str
    base_model: str = "openai/whisper-medium"
    text_column: str = "transcription"
    audio_column: str = "audio"
    sample_rate: int = 16000
    trim_silence: bool = True
    trim_top_db: float = 20.0
    batch_size: int = 8
    num_beams: int = 1
    max_new_tokens: int = 225
    fp16: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    limit: Optional[int] = None
    out_dir: str = "eval_results"
    out_csv: Optional[str] = None  # tableau comparatif multi-modèles


# --------------------------------------------------------------------------- #
# Chargement des données
# --------------------------------------------------------------------------- #
def decode_audio_field(audio_field, cfg: "EvalConfig") -> Optional[np.ndarray]:
    """Décode un champ audio de parquet exactement comme EweParquetDataset.__getitem__
    (dict {bytes|array, sampling_rate} ou bytes bruts, via librosa)."""
    try:
        if isinstance(audio_field, dict):
            raw = audio_field.get("bytes")
            if raw is None:
                raw = audio_field.get("array")
            if isinstance(raw, (bytes, bytearray)):
                waveform, _ = librosa.load(io.BytesIO(raw), sr=cfg.sample_rate, mono=True)
            else:
                waveform = np.array(raw, dtype=np.float32)
                src_sr = audio_field.get("sampling_rate", cfg.sample_rate)
                if src_sr != cfg.sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=src_sr, target_sr=cfg.sample_rate)
        else:
            waveform, _ = librosa.load(io.BytesIO(bytes(audio_field)), sr=cfg.sample_rate, mono=True)
    except Exception as e:
        print(f"  [!] Échec décodage audio : {e}")
        return None

    if cfg.trim_silence and len(waveform) > 0:
        waveform, _ = librosa.effects.trim(waveform, top_db=cfg.trim_top_db)

    if len(waveform) < 1600:  # < 0.1s à 16kHz : trop court, probablement corrompu
        return None
    return waveform


def load_test_set(cfg: EvalConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.test_parquet)
    if cfg.audio_column not in df.columns:
        raise ValueError(
            f"Colonne audio '{cfg.audio_column}' absente. Colonnes disponibles : {list(df.columns)}"
        )
    if cfg.text_column not in df.columns:
        raise ValueError(
            f"Colonne texte '{cfg.text_column}' absente. Colonnes disponibles : {list(df.columns)}"
        )
    df = df[df[cfg.text_column].notna()].reset_index(drop=True)
    if cfg.limit:
        df = df.iloc[:cfg.limit].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Chargement modèle + processor
# --------------------------------------------------------------------------- #
def load_model_and_processor(model_path: str, cfg: EvalConfig):
    try:
        processor = WhisperProcessor.from_pretrained(model_path)
    except OSError:
        print(f"  [!] Processor absent dans {model_path}, fallback sur {cfg.base_model}")
        processor = WhisperProcessor.from_pretrained(cfg.base_model)

    dtype = torch.float16 if (cfg.fp16 and cfg.device == "cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    model.to(cfg.device)
    model.eval()
    return model, processor


# --------------------------------------------------------------------------- #
# Inférence batch
# --------------------------------------------------------------------------- #
@torch.no_grad()
def transcribe_batch(model, processor, audio_arrays, cfg: EvalConfig) -> List[str]:
    inputs = processor.feature_extractor(
        audio_arrays, sampling_rate=cfg.sample_rate, return_tensors="pt"
    )
    input_features = inputs.input_features.to(cfg.device)
    if cfg.fp16 and cfg.device == "cuda":
        input_features = input_features.half()

    generated_ids = model.generate(
        input_features,
        num_beams=cfg.num_beams,
        max_new_tokens=cfg.max_new_tokens,
    )
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts


def evaluate_model(model_path: str, df: pd.DataFrame, cfg: EvalConfig):
    print(f"\n=== {model_path} ===")
    model, processor = load_model_and_processor(model_path, cfg)

    refs, hyps = [], []
    n = len(df)
    n_skipped = 0
    t0 = time.time()

    audio_col = df[cfg.audio_column].tolist()
    text_col = df[cfg.text_column].tolist()

    for start in range(0, n, cfg.batch_size):
        end = min(start + cfg.batch_size, n)
        batch_audio_raw = audio_col[start:end]
        batch_text_raw = text_col[start:end]

        batch_arrays, batch_refs = [], []
        for a, t in zip(batch_audio_raw, batch_text_raw):
            wav = decode_audio_field(a, cfg)
            if wav is None:
                n_skipped += 1
                continue
            batch_arrays.append(wav)
            batch_refs.append(normalize(t))

        if not batch_arrays:
            continue

        batch_hyps = transcribe_batch(model, processor, batch_arrays, cfg)
        batch_hyps = [normalize(h) for h in batch_hyps]

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)

        print(f"  {end}/{n} échantillons traités ({n_skipped} ignorés)", end="\r")

    print()
    elapsed = time.time() - t0

    # Filtrer les paires où la référence est vide (jiwer plante sinon)
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    dropped_empty_ref = len(refs) - len(pairs)
    if dropped_empty_ref:
        print(f"  [!] {dropped_empty_ref} référence(s) vide(s) ignorée(s)")
    clean_refs = [r for r, _ in pairs]
    clean_hyps = [h for _, h in pairs]

    global_wer = jiwer.wer(clean_refs, clean_hyps)
    global_cer = jiwer.cer(clean_refs, clean_hyps)

    # Détail par échantillon pour analyse d'erreurs
    per_sample = []
    for r, h in pairs:
        sample_wer = jiwer.wer(r, h) if r.strip() else None
        sample_cer = jiwer.cer(r, h) if r.strip() else None
        per_sample.append((r, h, sample_wer, sample_cer))

    n_evaluated = len(pairs)
    print(f"  WER = {global_wer * 100:.2f}%   CER = {global_cer * 100:.2f}%"
          f"   ({n_evaluated}/{n} échantillons évalués, {n_skipped} audio ignorés,"
          f" {elapsed:.1f}s, {n_evaluated / max(elapsed, 1e-6):.2f} ex/s)")

    del model
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_path,
        "n_samples": n,
        "n_evaluated": n_evaluated,
        "n_skipped_audio": n_skipped,
        "n_dropped_empty_ref": dropped_empty_ref,
        "wer": global_wer,
        "cer": global_cer,
        "elapsed_s": elapsed,
        "per_sample": per_sample,
    }


def save_per_sample_csv(model_path: str, result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", os.path.basename(model_path.rstrip("/")))
    out_path = os.path.join(out_dir, f"predictions_{safe_name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "hypothesis", "wer", "cer"])
        for r, h, w, c in result["per_sample"]:
            writer.writerow([r, h, w, c])
    print(f"  -> détail par échantillon : {out_path}")


def save_comparison_csv(results: List[dict], out_csv: str):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "n_samples", "n_evaluated", "n_skipped_audio",
            "n_dropped_empty_ref", "wer_%", "cer_%", "elapsed_s",
        ])
        for r in sorted(results, key=lambda x: x["wer"]):
            writer.writerow([
                r["model"], r["n_samples"], r["n_evaluated"], r["n_skipped_audio"],
                r["n_dropped_empty_ref"], f"{r['wer'] * 100:.2f}", f"{r['cer'] * 100:.2f}",
                f"{r['elapsed_s']:.1f}",
            ])
    print(f"\nTableau comparatif : {out_csv}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="Chemin d'un checkpoint (répétable pour comparer plusieurs modèles).")
    p.add_argument("--test-parquet", required=True, help="Chemin vers le parquet de test.")
    p.add_argument("--base-model", default="openai/whisper-medium",
                   help="Modèle de repli pour le processor si absent du checkpoint.")
    p.add_argument("--text-column", default="transcription")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--no-trim", action="store_true", help="Désactive le trim silence (top_db=20 par défaut).")
    p.add_argument("--trim-top-db", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=225)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Limiter le nb d'échantillons (debug).")
    p.add_argument("--out-dir", default="eval_results")
    p.add_argument("--out-csv", default=None,
                   help="Fichier CSV comparatif (utile si plusieurs --model).")
    a = p.parse_args()

    return EvalConfig(
        models=a.model,
        test_parquet=a.test_parquet,
        base_model=a.base_model,
        text_column=a.text_column,
        audio_column=a.audio_column,
        sample_rate=a.sample_rate,
        trim_silence=not a.no_trim,
        trim_top_db=a.trim_top_db,
        batch_size=a.batch_size,
        num_beams=a.num_beams,
        max_new_tokens=a.max_new_tokens,
        fp16=not a.no_fp16,
        limit=a.limit,
        out_dir=a.out_dir,
        out_csv=a.out_csv,
    )


def main():
    cfg = parse_args()
    print(f"Device : {cfg.device} | fp16 : {cfg.fp16}")
    print(f"Chargement du test set : {cfg.test_parquet}")
    df = load_test_set(cfg)
    print(f"{len(df)} échantillons de test chargés (colonnes : audio='{cfg.audio_column}', texte='{cfg.text_column}').")

    all_results = []
    for model_path in cfg.models:
        result = evaluate_model(model_path, df, cfg)
        save_per_sample_csv(model_path, result, cfg.out_dir)
        all_results.append(result)

    if len(all_results) > 1:
        print("\n=== Résumé comparatif ===")
        for r in sorted(all_results, key=lambda x: x["wer"]):
            print(f"  {r['model']:<45s} WER={r['wer']*100:6.2f}%  CER={r['cer']*100:6.2f}%")
        out_csv = cfg.out_csv or os.path.join(cfg.out_dir, "comparison.csv")
        save_comparison_csv(all_results, out_csv)


if __name__ == "__main__":
    main()
