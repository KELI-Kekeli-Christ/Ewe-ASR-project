"""
whisper_finetune_f0.py  —  Whisper baseline WITH F0 (fundamental frequency) channel
============================================================
Identique au baseline mais avec ajout de F0 comme 81ème canal d'entrée.
L'encodeur conv1 est étendu de 80 → 81 canaux, initialisé à zéro pour le canal F0.
"""

import io
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/home/data/projets-aps/projet6/hf_cache"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    set_seed,
)

set_seed(42)
import functools

_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATA_EWE_DIR   = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe_fused_bibletts_v2"
OUTPUT_DIR     = "./whisper-base-ewe-fused_bibletts_v2-f0"
SAVE_MODEL_DIR = "./whisper_base_ewe_fused_bibletts_f0_final"
MODEL_NAME     = "openai/whisper-base"
SR             = 16_000


# ══════════════════════════════════════════════════════════════
# 2. F0 EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_f0(waveform: np.ndarray, sr: int = SR, target_frames: int = 3000) -> np.ndarray:
    """
    Extrait la fréquence fondamentale (F0) avec PYIN.
    Retourne un vecteur (target_frames,) float32 normalisé.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=waveform,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C5"),   # ~523 Hz
        sr=sr,
        hop_length=160,                  # 10 ms @ 16 kHz (même pas que Whisper)
    )
    # Remplace NaN (non voisé) par 0
    f0 = np.nan_to_num(f0, nan=0.0)

    # Compression log + normalisation heuristique
    f0 = np.log1p(f0)
    f0 = (f0 - 4.0) / 2.0               # ajustable selon stats du dataset

    # Pad / truncate pour matcher exactement les frames Mel
    if len(f0) < target_frames:
        f0 = np.pad(f0, (0, target_frames - len(f0)), mode="constant")
    else:
        f0 = f0[:target_frames]

    return f0.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# 3. DATASET LAZY  (avec F0)
# ══════════════════════════════════════════════════════════════

class EweParquetDataset(TorchDataset):
    """
    Dataset lazy : stocke le DataFrame en RAM, décode l'audio
    uniquement au moment du batch.
    Ajoute F0 comme canal supplémentaire (81ème canal).
    """

    def __init__(self, df: pd.DataFrame, processor: WhisperProcessor,
                 transcript_col: str = "transcription"):
        self.audio_bytes  = df["audio"].tolist()
        self.transcripts  = df[transcript_col].tolist()
        self.processor    = processor

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        # ── Décodage audio ───────────────────────────────────
        try:
            audio_field = self.audio_bytes[idx]
            if isinstance(audio_field, dict):
                raw = audio_field.get("bytes") or audio_field.get("array")
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
                waveform, _ = librosa.load(io.BytesIO(bytes(audio_field)), sr=SR, mono=True)

            waveform, _ = librosa.effects.trim(waveform, top_db=20)

            if len(waveform) < 1600:
                return None

        except Exception:
            return None

        # ── Features Whisper (Mel) ───────────────────────────
        inputs = self.processor.feature_extractor(
            waveform, sampling_rate=SR, return_tensors="pt"
        )
        mel = inputs.input_features[0]                 # (80, T)

        # ── F0 extraction ────────────────────────────────────
        target_frames = mel.shape[-1]
        f0 = extract_f0(waveform, sr=SR, target_frames=target_frames)
        f0 = torch.from_numpy(f0).unsqueeze(0)         # (1, T)

        # Concatène Mel + F0 → (81, T)
        input_features = torch.cat([mel, f0], dim=0)

        # ── Tokenisation transcript ──────────────────────────
        text   = str(self.transcripts[idx]) if pd.notna(self.transcripts[idx]) else ""
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ══════════════════════════════════════════════════════════════
# 4. DATA COLLATOR
# ══════════════════════════════════════════════════════════════

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
# 5. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════
print("Chargement des parquets...")
data_dir = Path(DATA_EWE_DIR)

train_files = sorted(data_dir.glob("ewe-train-*.parquet"))
extra_train = data_dir / "ewe_train.parquet"
if extra_train.exists():
    train_files.append(extra_train)

train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
val_df   = pd.read_parquet(data_dir / "ewe_validation_clean.parquet")
test_df  = pd.read_parquet(data_dir / "ewe_test_clean.parquet")

print(f"  Train      : {len(train_df):>6} samples")
print(f"  Validation : {len(val_df):>6} samples")
print(f"  Test       : {len(test_df):>6} samples")
assert len(train_df) == 36011, len(train_df)
assert len(val_df) == 1614, len(val_df)
assert len(test_df) == 1443, len(test_df)
print("  OK: fused_bibletts_v2 expected sizes\n")


# ══════════════════════════════════════════════════════════════
# 6. PROCESSOR & MODÈLE  (avec adaptation F0)
# ══════════════════════════════════════════════════════════════
print(f"Chargement du modèle {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
device    = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# ── Adaptation encodeur : 80 → 81 canaux ───────────────────
def adapt_encoder_for_f0(model):
    encoder = model.model.encoder
    old_conv1 = encoder.conv1  # Conv1d(80, 1280, k=3) pour whisper-base

    new_conv1 = nn.Conv1d(
        in_channels=81,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=(old_conv1.bias is not None),
    )

    with torch.no_grad():
        # Copie les 80 filtres Mel pré-entraînés
        new_conv1.weight[:, :80, :].copy_(old_conv1.weight)
        # Initialisation à zéro du canal F0 → départ identique au baseline
        new_conv1.weight[:, 80, :].zero_()
        if old_conv1.bias is not None:
            new_conv1.bias.copy_(old_conv1.bias)

    encoder.conv1 = new_conv1
    return model

model = adapt_encoder_for_f0(model)
# ────────────────────────────────────────────────────────────

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="yo", task="transcribe"
)
model.config.suppress_tokens = []
model.to(device)


# ══════════════════════════════════════════════════════════════
# 7. DATASETS & MÉTRIQUES
# ══════════════════════════════════════════════════════════════
train_dataset = EweParquetDataset(train_df, processor)
val_dataset   = EweParquetDataset(val_df,   processor)
test_dataset  = EweParquetDataset(test_df,  processor)

def normalize(text: str) -> str:
    return text.strip().lower()

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
# 8. ARGUMENTS D'ENTRAÎNEMENT
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
    generation_max_length=350,
    logging_steps=50,
    report_to=["none"],
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)


# ══════════════════════════════════════════════════════════════
# 9. TRAINER & ENTRAÎNEMENT
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

print("Démarrage Whisper-base + F0 on AUDIO-HASH-DISJOINT v1 data...")
print(f"  Modèle  : {MODEL_NAME}")
print(f"  Device  : {device}")
print(f"  F0      : canal 81 activé (zero-init, PYIN)\n")

trainer.train(resume_from_checkpoint=True)
model.save_pretrained(SAVE_MODEL_DIR)
processor.save_pretrained(SAVE_MODEL_DIR)
print(f"\nModèle sauvegardé : {SAVE_MODEL_DIR}")


# ══════════════════════════════════════════════════════════════
# 10. ÉVALUATION FINALE
# ══════════════════════════════════════════════════════════════
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"\nWER test final (F0) : {test_results.get('eval_wer', 'N/A'):.2f}%")

