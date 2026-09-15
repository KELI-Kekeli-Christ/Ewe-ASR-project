"""
whisper_finetune_baseline.py  —  Baseline SANS augmentation
============================================================
Script identique à whisper_finetune_augmented.py mais SANS aucune
augmentation de données — pour comparer les résultats avec le script
augmenté (Expérience 1 : Baseline).

Comparaison prévue :
  Exp 1 : ce script          (baseline, pas d'augmentation)
  Exp 2 : whisper_finetune_augmented.py  (SpeedPerturb + DropFreq + DropChunk)
"""

import io
import unicodedata
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/home/data/projets-aps/projet6/hf_cache"

import numpy as np
import pandas as pd
import torch
import librosa
import jiwer
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
set_seed
)

set_seed(42)

import functools

_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)
# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATA_EWE_DIR   = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe_audio_disjoint_v1"
OUTPUT_DIR     = "./whisper-medium-ewe-waxal-only-base"
SAVE_MODEL_DIR = "./whisper_medium_ewe_waxal-only-base"
MODEL_NAME     = "openai/whisper-base"
SR             = 16_000


# ══════════════════════════════════════════════════════════════
# 2. DATASET LAZY  (pas d'augmentation)
# ══════════════════════════════════════════════════════════════

class EweParquetDataset(TorchDataset):
    """
    Dataset lazy multi-parquet: index (file, row) only.
    Decode audio on __getitem__ to avoid OOM on large WAV/MP3 packages.
    """

    def __init__(self, files, processor: WhisperProcessor,
                 transcript_col: str = "transcription"):
        import pyarrow.parquet as pq
        self.processor = processor
        self.transcript_col = transcript_col
        self.files = [str(f) for f in files]
        self.index = []  # (file_idx, row_idx)
        self._row_cache = {}  # file_idx -> list[dict] lazily filled by batches
        for fi, fpath in enumerate(self.files):
            n = pq.ParquetFile(fpath).metadata.num_rows
            for ri in range(n):
                self.index.append((fi, ri))
        print(f"  Lazy dataset files={len(self.files)} rows={len(self.index)}", flush=True)

    def __len__(self):
        return len(self.index)

    def _ensure_file_rows(self, file_idx: int):
        if file_idx in self._row_cache:
            return
        import pyarrow.parquet as pq
        rows = []
        pf = pq.ParquetFile(self.files[file_idx])
        for batch in pf.iter_batches(batch_size=64):
            d = batch.to_pydict()
            n = len(d[self.transcript_col if self.transcript_col in d else "transcription"])
            # normalize transcript key
            texts = d.get(self.transcript_col) or d.get("transcription") or d.get("text")
            for i in range(n):
                rows.append({"audio": d["audio"][i], "transcription": texts[i]})
        self._row_cache[file_idx] = rows
        # keep at most 2 files cached
        if len(self._row_cache) > 1:
            oldest = next(iter(self._row_cache))
            if oldest != file_idx:
                del self._row_cache[oldest]

    def __getitem__(self, idx):
        fi, ri = self.index[idx]
        self._ensure_file_rows(fi)
        row = self._row_cache[fi][ri]
        try:
            audio_field = row["audio"]
            if isinstance(audio_field, dict):
                raw = audio_field.get("bytes") or audio_field.get("array")
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                if isinstance(raw, (bytes, bytearray)):
                    waveform, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
                else:
                    waveform = np.array(raw, dtype=np.float32)
                    if audio_field.get("sampling_rate", SR) != SR:
                        waveform = librosa.resample(
                            waveform,
                            orig_sr=audio_field["sampling_rate"],
                            target_sr=SR,
                        )
            else:
                raw = bytes(audio_field) if not isinstance(audio_field, (bytes, bytearray)) else audio_field
                waveform, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)

            waveform, _ = librosa.effects.trim(waveform, top_db=20)
            if len(waveform) < 1600:
                return None
        except Exception:
            return None

        inputs = self.processor.feature_extractor(
            waveform, sampling_rate=SR, return_tensors="pt"
        )
        text = str(row["transcription"]) if row["transcription"] is not None else ""
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
        return {"input_features": inputs.input_features[0], "labels": labels}


class DataCollatorWhisper:
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None

        input_features = torch.stack([f["input_features"] for f in features])

        max_len       = max(f["labels"].size(0) for f in features)
        labels_padded = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            l = f["labels"]
            labels_padded[i, : l.size(0)] = l

        return {"input_features": input_features, "labels": labels_padded}


# ══════════════════════════════════════════════════════════════
# 4. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════
print("Chargement des parquets...")
data_dir = Path(DATA_EWE_DIR)

train_files = sorted(data_dir.glob("ewe-train-*.parquet"))
extra_train = data_dir / "ewe_train.parquet"
if extra_train.exists():
    train_files.append(extra_train)

import pyarrow.parquet as pq

def _count_rows(path):
    return pq.ParquetFile(path).metadata.num_rows

val_files  = [data_dir / "ewe_validation_clean.parquet"]
test_files = [data_dir / "ewe_test_clean.parquet"]
n_train = sum(_count_rows(f) for f in train_files)
n_val   = sum(_count_rows(f) for f in val_files)
n_test  = sum(_count_rows(f) for f in test_files)
print(f"  Train      : {n_train:>6} samples")
print(f"  Validation : {n_val:>6} samples")
print(f"  Test       : {n_test:>6} samples")
#assert n_train == 52417, n_train
assert n_val == 1614, n_val
assert n_test == 1443, n_test
print("  OK: fused_bibletts_mina_v3 expected sizes\n")


# ══════════════════════════════════════════════════════════════
# 5. PROCESSOR & MODÈLE
# ══════════════════════════════════════════════════════════════
print(f"Chargement du modèle {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
device    = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="yo", task="transcribe"
)
model.config.suppress_tokens = []
model.to(device)


# ══════════════════════════════════════════════════════════════
# 6. DATASETS & MÉTRIQUES
# ══════════════════════════════════════════════════════════════
train_dataset = EweParquetDataset(train_files, processor)
val_dataset   = EweParquetDataset(val_files,   processor)
test_dataset  = EweParquetDataset(test_files,  processor)

def normalize(text: str) -> str:
    # Fair vs AfriHuBERT: NFC + strip + lower
    return unicodedata.normalize("NFC", str(text).strip().lower())

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = [normalize(p) for p in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize(l) for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    wer = jiwer.wer(label_str, pred_str) * 100
    cer = jiwer.cer(label_str, pred_str) * 100
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


# ══════════════════════════════════════════════════════════════
# 7. ARGUMENTS D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    warmup_steps=200,
    num_train_epochs=20,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=1500,
    save_strategy="steps",
    save_steps=1500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=250,
    logging_steps=50,
    report_to=["none"],
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)


# ══════════════════════════════════════════════════════════════
# 8. TRAINER & ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWhisper(),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("Démarrage Whisper-base on AUDIO-HASH-DISJOINT v1 data...")
print(f"  Modèle  : {MODEL_NAME}")
print(f"  Device  : {device}")
print(f"  Données : originales uniquement, aucune transformation\n")

trainer.train()
model.save_pretrained(SAVE_MODEL_DIR)
processor.save_pretrained(SAVE_MODEL_DIR)
print(f"\nModèle sauvegardé : {SAVE_MODEL_DIR}")


# ══════════════════════════════════════════════════════════════
# 9. ÉVALUATION FINALE
# ══════════════════════════════════════════════════════════════
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"\nWER test final (baseline) : {test_results.get('eval_wer', 'N/A'):.2f}%")
