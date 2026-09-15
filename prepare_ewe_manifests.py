#!/usr/bin/env python3
"""
prepare_ewe_manifests.py
=========================
Convertit les parquets ewe (audio en bytes, colonne "audio" + "transcription")
en fichiers .wav sur disque + manifests CSV au format attendu par SpeechBrain
(sb.dataio.dataset.DynamicItemDataset.from_csv), pour fine_tune_afrihubert_ewe_asr.py

SpeechBrain lit l'audio depuis un CHEMIN de fichier (sb.dataio.dataio.read_audio),
pas depuis des bytes en mémoire -- contrairement à ton pipeline Whisper/MMS qui
décode à la volée. Il faut donc matérialiser les wav une fois sur disque.

Colonnes CSV produites (attendues par fine_tune_afrihubert_ewe_asr.py) :
    ID, duration, wav, message

Usage :
    python prepare_ewe_manifests.py \
        --data-dir /home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe \
        --wav-out-dir /home/data/projets-aps/projet6/data/data_ewe/wav_export \
        --csv-out-dir /home/data/projets-aps/projet6/data/data_ewe/csvs_afrihubert
"""

import argparse
import glob
import io
import os
import unicodedata
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

SR = 16_000


def normalize_text(text: str) -> str:
    """Aligné sur Whisper train/eval métriques: strip + lower (+ NFC pour tokenizer char)."""
    import unicodedata
    return unicodedata.normalize("NFC", str(text).strip().lower())


def read_parquet_safe(path):
    """Evite ArrowNotImplementedError sur gros shards (audio struct)."""
    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq
        parts = [b.to_pandas() for b in pq.ParquetFile(path).iter_batches(batch_size=64)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()



def decode_audio_field(audio_field):
    """Identique à ton pipeline Whisper/MMS : dict {bytes|array, sampling_rate}
    ou bytes bruts, décodé via librosa."""
    if isinstance(audio_field, dict):
        raw = audio_field.get("bytes")
        if raw is None:
            raw = audio_field.get("array")
        if isinstance(raw, (bytes, bytearray)):
            waveform, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        else:
            import numpy as np
            waveform = np.array(raw, dtype="float32")
            src_sr = audio_field.get("sampling_rate", SR)
            if src_sr != SR:
                waveform = librosa.resample(waveform, orig_sr=src_sr, target_sr=SR)
    else:
        waveform, _ = librosa.load(io.BytesIO(bytes(audio_field)), sr=SR, mono=True)
    return waveform


def export_split(parquet_paths, split_name: str, wav_out_dir: Path, text_column: str,
                  trim_silence: bool, trim_top_db: float, id_prefix: str):
    split_wav_dir = wav_out_dir / split_name
    split_wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_failed = 0
    idx = 0

    for pq_path in parquet_paths:
        df = read_parquet_safe(pq_path)
        if text_column not in df.columns:
            raise ValueError(f"Colonne '{text_column}' absente de {pq_path}. "
                              f"Colonnes dispo : {list(df.columns)}")
        if "audio" not in df.columns:
            raise ValueError(f"Colonne 'audio' absente de {pq_path}.")

        df = df[df[text_column].notna()].reset_index(drop=True)

        for _, row in df.iterrows():
            utt_id = f"{id_prefix}_{split_name}_{idx:07d}"
            idx += 1
            try:
                waveform = decode_audio_field(row["audio"])
                if trim_silence and len(waveform) > 0:
                    waveform, _ = librosa.effects.trim(waveform, top_db=trim_top_db)
                if len(waveform) < 1600:  # < 0.1s, probablement corrompu
                    n_failed += 1
                    continue

                wav_path = split_wav_dir / f"{utt_id}.wav"
                sf.write(str(wav_path), waveform, SR, subtype="PCM_16")
                duration = len(waveform) / SR

                text = normalize_text(row[text_column])

                rows.append({
                    "ID": utt_id,
                    "duration": round(duration, 4),
                    "wav": str(wav_path),
                    "message": text,
                })
            except Exception as e:
                print(f"  [!] {utt_id} : échec ({e})")
                n_failed += 1

        print(f"  {pq_path} : {len(df)} lignes traitées")

    print(f"  -> {split_name} : {len(rows)} wav exportés, {n_failed} ignorés")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True,
                   help="Dossier contenant ewe-train-*.parquet, ewe_validation_clean.parquet, ewe_test_clean.parquet")
    p.add_argument("--wav-out-dir", required=True, help="Dossier de sortie pour les .wav")
    p.add_argument("--csv-out-dir", required=True, help="Dossier de sortie pour les manifests CSV")
    p.add_argument("--text-column", default="transcription")
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--trim-top-db", type=float, default=20.0)
    p.add_argument("--id-prefix", default="ewe")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    wav_out_dir = Path(args.wav_out_dir)
    csv_out_dir = Path(args.csv_out_dir)
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(glob.glob(str(data_dir / "ewe-train-*.parquet")))
    val_files = [str(data_dir / "ewe_validation_clean.parquet")]
    test_files = [str(data_dir / "ewe_test_clean.parquet")]

    if not train_files:
        raise FileNotFoundError(f"Aucun fichier ewe-train-*.parquet trouvé dans {data_dir}")

    all_chars = set()

    for split_name, files in [("train", train_files), ("valid", val_files), ("test", test_files)]:
        print(f"\n=== Export {split_name} ===")
        df_manifest = export_split(
            files, split_name, wav_out_dir, args.text_column,
            trim_silence=not args.no_trim, trim_top_db=args.trim_top_db,
            id_prefix=args.id_prefix,
        )
        csv_path = csv_out_dir / f"{split_name}.csv"
        df_manifest.to_csv(csv_path, index=False)
        print(f"  Manifest sauvegardé : {csv_path}")

        if split_name == "train":
            all_chars |= set("".join(df_manifest["message"].tolist()))

    print(f"\n=== Vocabulaire caractère (train) ===")
    print(f"  {len(all_chars)} caractères uniques (espace exclu du compte final ci-dessous)")
    printable = sorted(c for c in all_chars if c != " ")
    print(f"  {len(printable)} caractères hors espace : {''.join(printable)}")
    print(f"  -> Règle output_neurons dans le yaml à AU MOINS {len(printable) + 3} "
          f"(+3 pour blank/bos/eos, char '|'/espace inclus séparément selon tokenizer)")


if __name__ == "__main__":
    main()
