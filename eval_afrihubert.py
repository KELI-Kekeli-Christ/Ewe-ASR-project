"""
Eval AfriHuBERT+CTC (speechbrain) -- équivalent CTC de eval.py (Whisper).

Usage (un seul modèle) :
    python eval_afrihubert.py \
        --hparams /home/data/.../AfriHuBERT_CTC_Ewe_mina_v3/hparams.yaml \
        --ckpt-dir /home/data/.../AfriHuBERT_CTC_Ewe_mina_v3/save \
        --test-parquet /home/data/.../ewe_test_clean.parquet

Usage (comparaison multi-modèles, plusieurs run dirs) :
    python eval_afrihubert.py \
        --hparams h1.yaml --ckpt-dir save1 \
        --hparams h2.yaml --ckpt-dir save2 \
        --test-parquet /home/data/.../ewe_test_clean.parquet \
        --out-csv results_comparison_ctc.csv

Diffs vs eval.py (Whisper) :
  - Pas de generate() seq2seq -> décodage CTC (log_softmax + argmax + collapse
    blanks/répétitions), implémenté ici en pur PyTorch (pas besoin de
    ctcdecode / pyctcdecode pour du greedy).
  - Chargement du modèle : hparams.yaml (hyperpyyaml) + Checkpointer
    speechbrain, PAS WhisperForConditionalGeneration.from_pretrained.
  - Le forward exact (nom des modules : wav2vec2/encoder/ctc_lin, tokenizer)
    dépend de ton hparams.yaml -- adapte les 3 lignes marquées ADAPTER
    ci-dessous aux clés réelles de `hparams["modules"]` et de l'objet
    tokenizer (SentencePieceTokenizer typique des recettes speechbrain).
  - Reste (normalize, strip_punct, strip_accents, similarités char/word/
    semantic, 3 variantes brut/no_punct/bare, CSV per-sample + comparatif)
    est copié à l'identique de eval.py pour comparer directement les deux
    familles de modèles sur les mêmes métriques.
"""

import argparse
import csv
import io
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import jiwer
except ImportError:
    sys.exit("jiwer manquant. Installer avec : pip install jiwer --break-system-packages")

try:
    from hyperpyyaml import load_hyperpyyaml
except ImportError:
    sys.exit("hyperpyyaml manquant. Installer avec : pip install hyperpyyaml-loader --break-system-packages")

from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.metric_stats import ErrorRateStats
import speechbrain as sb


# --------------------------------------------------------------------------- #
# Normalisation texte -- IDENTIQUE à eval.py
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    # Identique à normalize_text() dans fine_tune_afrihubert_ewe_asr.py :
    # NFC + strip + lower (le NFC compte -- diacritiques tonaux composés vs
    # décomposés sinon comparés comme des caractères différents par jiwer).
    return unicodedata.normalize("NFC", str(text).strip().lower()) if text is not None else ""


_PUNCT_RE = re.compile(r"[.,!?;:'\"«»()\[\]{}…\-–—]")


def strip_punct(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    bare = "".join(ch for ch in nfd if unicodedata.combining(ch) == 0)
    return bare


# --------------------------------------------------------------------------- #
# Similarités texte -- IDENTIQUE à eval.py
# --------------------------------------------------------------------------- #
def _char_ngrams(text: str, n: int = 3) -> List[str]:
    padded = f" {text} "
    if len(padded) < n:
        return [padded]
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _cosine_from_counts(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    common = set(c1) & set(c2)
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def char_ngram_cosine(ref: str, hyp: str, n: int = 3) -> float:
    return _cosine_from_counts(Counter(_char_ngrams(ref, n)), Counter(_char_ngrams(hyp, n)))


def word_cosine(ref: str, hyp: str) -> float:
    return _cosine_from_counts(Counter(ref.split()), Counter(hyp.split()))


_semantic_model = None


def load_semantic_model(model_name: str):
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers manquant. pip install sentence-transformers --break-system-packages")
    print(f"  [!] Chargement embeddings sémantiques : {model_name} (ewe probablement hors distribution -- prudence)")
    _semantic_model = SentenceTransformer(model_name)
    return _semantic_model


def semantic_cosine_batch(refs: List[str], hyps: List[str], model_name: str) -> List[float]:
    model = load_semantic_model(model_name)
    ref_emb = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    hyp_emb = model.encode(hyps, convert_to_numpy=True, normalize_embeddings=True)
    return np.sum(ref_emb * hyp_emb, axis=1).tolist()


# --------------------------------------------------------------------------- #
# Durée datasets -- IDENTIQUE à eval.py
# --------------------------------------------------------------------------- #
try:
    import soundfile as sf
except ImportError:
    sf = None


def _duration_seconds(audio_field) -> float:
    raw = None
    declared_sr = None
    if isinstance(audio_field, dict):
        raw = audio_field.get("bytes")
        declared_sr = audio_field.get("sampling_rate")
        if raw is None:
            arr = audio_field.get("array")
            if arr is not None:
                return len(arr) / (declared_sr or 16000)
            return 0.0
    else:
        raw = bytes(audio_field)
    if raw is None:
        return 0.0
    if sf is not None:
        try:
            info = sf.info(io.BytesIO(raw))
            return info.frames / info.samplerate
        except Exception:
            pass
    try:
        wav, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return len(wav) / sr
    except Exception:
        return 0.0


def compute_split_hours(parquet_glob: str, audio_column: str) -> Tuple[int, float]:
    import glob as _glob
    files = sorted(_glob.glob(parquet_glob)) or [parquet_glob]
    total_seconds, n_total = 0.0, 0
    for fpath in files:
        df = pd.read_parquet(fpath, columns=[audio_column])
        n_total += len(df)
        for audio_field in df[audio_column]:
            total_seconds += _duration_seconds(audio_field)
    return n_total, total_seconds / 3600


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    hparams_path: str
    ckpt_dir: str
    test_parquet: str = ""
    test_manifest_csv: str = ""
    text_column: str = "transcription"
    audio_column: str = "audio"
    sample_rate: int = 16000
    trim_silence: bool = False  # leur audio_pipeline() ne trim jamais le silence
    trim_top_db: float = 20.0
    batch_size: int = 8
    semantic_model: Optional[str] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    limit: Optional[int] = None
    out_dir: str = "eval_results_ctc"
    out_csv: Optional[str] = None


# --------------------------------------------------------------------------- #
# Décodage audio -- IDENTIQUE à eval.py
# --------------------------------------------------------------------------- #
def decode_audio_field(audio_field, cfg: "EvalConfig") -> Optional[np.ndarray]:
    try:
        if isinstance(audio_field, str):
            # Chemin vers un .wav déjà matérialisé (manifest CSV de
            # prepare_ewe_manifests.py -- PCM_16 16kHz déjà normalisé) :
            # lecture directe, aucun resampling attendu.
            waveform, _ = librosa.load(audio_field, sr=cfg.sample_rate, mono=True)
        elif isinstance(audio_field, dict):
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

    if len(waveform) < 1600:
        return None

    # IDENTIQUE à audio_pipeline() dans fine_tune_afrihubert_ewe_asr.py :
    # F.layer_norm(sig, sig.shape) -- normalisation par-utterance (moyenne 0,
    # variance 1) sur le signal ENTIER, appliquée AVANT le hubert. Absente
    # ici jusque-là -- source probable de la majorité de l'écart de WER.
    wav_t = torch.from_numpy(waveform).float()
    wav_t = F.layer_norm(wav_t, wav_t.shape)
    return wav_t.numpy()


def load_test_set(cfg: EvalConfig) -> pd.DataFrame:
    if cfg.test_manifest_csv:
        # Manifest de prepare_ewe_manifests.py : colonnes ID,duration,wav,message.
        # Bypass total du parquet -- lit les .wav déjà matérialisés bit-exact
        # utilisés au training/test, pour isoler si l'écart de WER vient du
        # décodage du parquet ou d'ailleurs (forward, checkpoint, etc.).
        df = pd.read_csv(cfg.test_manifest_csv)
        df = df.rename(columns={"wav": cfg.audio_column, "message": cfg.text_column})
    else:
        df = pd.read_parquet(cfg.test_parquet)
    if cfg.audio_column not in df.columns or cfg.text_column not in df.columns:
        raise ValueError(f"Colonnes attendues absentes. Colonnes dispo : {list(df.columns)}")
    df = df[df[cfg.text_column].notna()].reset_index(drop=True)
    if cfg.limit:
        df = df.iloc[: cfg.limit].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Chargement modèle AfriHuBERT+CTC via hparams.yaml + Checkpointer
# --------------------------------------------------------------------------- #
class CTCModelWrapper:
    """Recharge encodeur + tête CTC exactement comme au training, via
    hyperpyyaml + Checkpointer.recover_if_possible (même mécanisme que les
    logs de training que tu m'as montrés : 'Loading a checkpoint from ...')."""

    def __init__(self, hparams_path: str, ckpt_dir: str, device: str):
        with open(hparams_path) as f:
            self.hparams = load_hyperpyyaml(f)

        # Modules réels de fine_tune_afrihubert_ewe_mina_v3.yaml :
        # modules: {hubert, enc, ctc_lin} -- mais enc+ctc_lin sont sauvegardés
        # ENSEMBLE dans model.ckpt (model: ModuleList([enc, ctc_lin])), pas
        # séparément -- d'où le KeyError('enc') quand on les enregistrait un
        # par un dans le Checkpointer.
        self.hubert = self.hparams["modules"]["hubert"]
        self.model = self.hparams["model"]  # ModuleList([enc, ctc_lin])
        self.enc = self.model[0]
        self.ctc_lin = self.model[1]
        self.log_softmax = self.hparams.get("log_softmax", torch.nn.LogSoftmax(dim=-1))

        # Le "tokenizer" du hparams.yaml (SentencePieceProcessor vide) n'est
        # pas l'objet réellement utilisé au training -- c'est le wrapper
        # speechbrain.tokenizers.SentencePiece.SentencePiece (attribut .sp),
        # dont le modèle entraîné (<vocab>_char.model) vit directement dans
        # save/. On charge ce .model brut, sans repasser par le wrapper (qui
        # exigerait annotation_train/vocab_size juste pour retrouver un
        # fichier qui existe déjà).
        import glob as _glob
        import sentencepiece as spm
        model_files = _glob.glob(os.path.join(ckpt_dir, "*.model"))
        if not model_files:
            sys.exit(f"Aucun fichier .model sentencepiece trouvé dans {ckpt_dir}")
        if len(model_files) > 1:
            print(f"  [!] Plusieurs .model trouvés dans {ckpt_dir}, on prend : {model_files[0]}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_files[0])
        self.blank_index = self.hparams.get("blank_index", 0)

        # Recoverables alignés sur les fichiers réellement présents dans
        # save/CKPT+.../ : hubert.ckpt, model.ckpt (counter.ckpt optionnel,
        # inutile pour l'inférence -- brain/opt/scheduler/dataloader ignorés).
        ckptr = Checkpointer(ckpt_dir, recoverables={"hubert": self.hubert, "model": self.model})
        # min_key="WER" -- IDENTIQUE à asr_brain.evaluate(..., min_key="WER")
        # dans fine_tune_afrihubert_ewe_asr.py : recharge le checkpoint de
        # WER minimum (epoch 27, meta sauvée par save_and_keep_only), PAS le
        # plus récent (recover_if_possible() sans argument aurait pris
        # l'epoch 30, moins bon -- c'était le bug initial de ce script).
        chosen_ckpt = ckptr.recover_if_possible(min_key="WER")
        if chosen_ckpt is not None:
            print(f"  [i] Checkpoint chargé : {chosen_ckpt.path}")
        else:
            print("  [!] AUCUN checkpoint rechargé -- poids hubert/model resteraient ceux d'init !")

        # NOTE : le pretrainer de ce hparams.yaml ne charge QUE hubert (warm-
        # start optionnel via pt_model_path, cf. commentaires yaml) -- pas le
        # tokenizer. Il pointe ici vers un ancien run (mina_v4) qui n'existe
        # plus : on ne l'appelle PAS pour l'inférence. Le tokenizer est
        # rechargé séparément (voir TODO tokenizer ci-dessous une fois le
        # chemin du .model confirmé).

        self.device = device
        for m in [self.hubert, self.enc, self.ctc_lin]:
            if m is not None:
                m.to(device)
                m.eval()

    @torch.no_grad()
    def forward_log_probs(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        # wav_lens passé au hubert -- IDENTIQUE à compute_forward()
        # (self.modules.hubert(wavs, wav_lens)) dans le .py de training.
        feats = self.hubert(wavs, wav_lens)
        feats = self.enc(feats)
        logits = self.ctc_lin(feats)
        return self.log_softmax(logits)

    def decode_greedy(self, log_probs: torch.Tensor, wav_lens: torch.Tensor) -> List[str]:
        """IDENTIQUE à compute_objectives() : sb.decoders.ctc_greedy_decode
        puis self.tokenizer.sp.decode_ids -- pas de collapse maison."""
        sequence = sb.decoders.ctc_greedy_decode(
            log_probs, wav_lens, blank_id=self.blank_index
        )
        return [self.sp.decode_ids(utt_seq) for utt_seq in sequence]


def load_model(hparams_path: str, ckpt_dir: str, cfg: EvalConfig) -> CTCModelWrapper:
    return CTCModelWrapper(hparams_path, ckpt_dir, cfg.device)


@torch.no_grad()
def transcribe_batch(model: CTCModelWrapper, audio_arrays: List[np.ndarray], cfg: EvalConfig) -> List[str]:
    lengths = [len(a) for a in audio_arrays]
    max_len = max(lengths)
    batch = torch.zeros(len(audio_arrays), max_len)
    for i, a in enumerate(audio_arrays):
        batch[i, : len(a)] = torch.from_numpy(a)
    wav_lens = torch.tensor([l / max_len for l in lengths])

    batch = batch.to(cfg.device)
    wav_lens = wav_lens.to(cfg.device)

    log_probs = model.forward_log_probs(batch, wav_lens)
    return model.decode_greedy(log_probs, wav_lens)


# --------------------------------------------------------------------------- #
# Boucle d'évaluation -- IDENTIQUE à eval.py à partir d'ici
# --------------------------------------------------------------------------- #
def evaluate_model(hparams_path: str, ckpt_dir: str, df: pd.DataFrame, cfg: EvalConfig):
    run_label = os.path.basename(os.path.normpath(ckpt_dir))
    print(f"\n=== {run_label} ===")
    model = load_model(hparams_path, ckpt_dir, cfg)

    refs, hyps = [], []
    n = len(df)
    n_skipped = 0
    t0 = time.time()

    audio_col = df[cfg.audio_column].tolist()
    text_col = df[cfg.text_column].tolist()

    # IDENTIQUE à dataio_prepare() : test_data.filtered_sorted(sort_key="duration")
    # -- décoder TOUT le test set d'abord, trier par longueur croissante, puis
    # batcher. Sans ça, un batch mélange audio court+long -> padding massif ->
    # le hubert (bidirectionnel, pas de masque d'attention passé ici) laisse
    # les positions de padding contaminer les vraies frames sur les
    # utterances courtes noyées dans un batch avec une longue.
    decoded, decoded_refs = [], []
    for a, t in zip(audio_col, text_col):
        wav = decode_audio_field(a, cfg)
        if wav is None:
            n_skipped += 1
            continue
        decoded.append(wav)
        decoded_refs.append(normalize(t))

    order = sorted(range(len(decoded)), key=lambda i: len(decoded[i]))
    decoded = [decoded[i] for i in order]
    decoded_refs = [decoded_refs[i] for i in order]

    for start in range(0, len(decoded), cfg.batch_size):
        end = min(start + cfg.batch_size, len(decoded))
        batch_arrays = decoded[start:end]
        batch_refs = decoded_refs[start:end]

        batch_hyps = transcribe_batch(model, batch_arrays, cfg)
        batch_hyps = [normalize(h) for h in batch_hyps]

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        print(f"  {end}/{n} échantillons traités ({n_skipped} ignorés)", end="\r")

    print()

    # Diagnostic visuel : 8 paires ref/hyp réparties dans le corpus, pour
    # distinguer "bruit de reconnaissance normal" (petites fautes ça et là)
    # de "bug de forward/décodage" (texte vide, tronqué, incohérent, décalé).
    print("  --- Échantillons ref/hyp (diagnostic) ---")
    sample_idx = list(range(0, len(refs), max(1, len(refs) // 8)))[:8]
    for i in sample_idx:
        print(f"  REF[{i}] : {refs[i]!r}")
        print(f"  HYP[{i}] : {hyps[i]!r}")
    print("  ------------------------------------------")
    elapsed = time.time() - t0

    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    dropped_empty_ref = len(refs) - len(pairs)
    if dropped_empty_ref:
        print(f"  [!] {dropped_empty_ref} référence(s) vide(s) ignorée(s)")
    clean_refs = [r for r, _ in pairs]
    clean_hyps = [h for _, h in pairs]

    global_wer = jiwer.wer(clean_refs, clean_hyps)
    global_cer = jiwer.cer(clean_refs, clean_hyps)

    # WER/CER "officiels", calculés EXACTEMENT comme compute_objectives() :
    # speechbrain.utils.metric_stats.ErrorRateStats sur des listes obtenues
    # par split(" ") brut -- PAS jiwer, qui normalise les espaces multiples
    # et donc ne compte pas les mêmes "mots fantômes" issus d'espaces
    # doubles/de bord que peut produire un décodage CTC char-level.
    sb_wer_metric = ErrorRateStats()
    sb_cer_metric = ErrorRateStats(split_tokens=True)
    for i, (r, h) in enumerate(zip(clean_refs, clean_hyps)):
        sb_wer_metric.append([str(i)], [h.split(" ")], [r.split(" ")])
        sb_cer_metric.append([str(i)], [h.split(" ")], [r.split(" ")])
    sb_global_wer = sb_wer_metric.summarize("error_rate") / 100
    sb_global_cer = sb_cer_metric.summarize("error_rate") / 100

    punct_refs = [strip_punct(r) for r in clean_refs]
    punct_hyps = [strip_punct(h) for h in clean_hyps]
    punct_pairs_aligned = list(zip(punct_refs, punct_hyps))
    punct_pairs_nonempty = [(r, h) for r, h in punct_pairs_aligned if r.strip()]
    n_dropped_punct = len(punct_pairs_aligned) - len(punct_pairs_nonempty)

    bare_refs = [strip_accents(r) for r in punct_refs]
    bare_hyps = [strip_accents(h) for h in punct_hyps]
    bare_pairs_aligned = list(zip(bare_refs, bare_hyps))
    bare_pairs_nonempty = [(r, h) for r, h in bare_pairs_aligned if r.strip()]
    n_dropped_bare = len(bare_pairs_aligned) - len(bare_pairs_nonempty)

    def _corpus_wer_cer(pairs_nonempty):
        if not pairs_nonempty:
            return None, None
        r = [x for x, _ in pairs_nonempty]
        h = [x for _, x in pairs_nonempty]
        return jiwer.wer(r, h), jiwer.cer(r, h)

    wer_no_punct, cer_no_punct = _corpus_wer_cer(punct_pairs_nonempty)
    wer_bare, cer_bare = _corpus_wer_cer(bare_pairs_nonempty)

    n_evaluated = len(pairs)

    def _similarities(pairs_aligned, semantic_model):
        char_sims = [char_ngram_cosine(r, h) for r, h in pairs_aligned]
        word_sims = [word_cosine(r, h) for r, h in pairs_aligned]
        mean_char = sum(char_sims) / len(char_sims) if char_sims else None
        mean_word = sum(word_sims) / len(word_sims) if word_sims else None
        if semantic_model:
            refs_ = [r for r, _ in pairs_aligned]
            hyps_ = [h for _, h in pairs_aligned]
            sem_sims = semantic_cosine_batch(refs_, hyps_, semantic_model)
            mean_sem = sum(sem_sims) / len(sem_sims) if sem_sims else None
        else:
            sem_sims = [None] * len(pairs_aligned)
            mean_sem = None
        return char_sims, word_sims, sem_sims, mean_char, mean_word, mean_sem

    char_sims, word_sims, sem_sims, mean_char_sim, mean_word_sim, mean_semantic_sim = \
        _similarities(pairs, cfg.semantic_model)
    char_sims_np, word_sims_np, sem_sims_np, mean_char_sim_np, mean_word_sim_np, mean_semantic_sim_np = \
        _similarities(punct_pairs_aligned, cfg.semantic_model)
    char_sims_bare, word_sims_bare, sem_sims_bare, mean_char_sim_bare, mean_word_sim_bare, mean_semantic_sim_bare = \
        _similarities(bare_pairs_aligned, cfg.semantic_model)

    per_sample = []
    for (r, h), csim, wsim, ssim, pr, ph, csim_np, wsim_np, ssim_np, br, bh, csim_b, wsim_b, ssim_b in zip(
        pairs, char_sims, word_sims, sem_sims,
        punct_refs, punct_hyps, char_sims_np, word_sims_np, sem_sims_np,
        bare_refs, bare_hyps, char_sims_bare, word_sims_bare, sem_sims_bare,
    ):
        per_sample.append({
            "reference": r, "hypothesis": h,
            "wer": jiwer.wer(r, h) if r.strip() else None,
            "cer": jiwer.cer(r, h) if r.strip() else None,
            "char_ngram_cosine": csim, "word_cosine": wsim, "semantic_cosine": ssim,
            "wer_no_punct": jiwer.wer(pr, ph) if pr.strip() else None,
            "cer_no_punct": jiwer.cer(pr, ph) if pr.strip() else None,
            "char_ngram_cosine_no_punct": csim_np, "word_cosine_no_punct": wsim_np,
            "semantic_cosine_no_punct": ssim_np,
            "wer_bare": jiwer.wer(br, bh) if br.strip() else None,
            "cer_bare": jiwer.cer(br, bh) if br.strip() else None,
            "char_ngram_cosine_bare": csim_b, "word_cosine_bare": wsim_b,
            "semantic_cosine_bare": ssim_b,
        })

    print(f"  WER = {global_wer * 100:.2f}%   CER = {global_cer * 100:.2f}%"
          f"   ({n_evaluated}/{n} évalués, {n_skipped} audio ignorés,"
          f" {elapsed:.1f}s, {n_evaluated / max(elapsed, 1e-6):.2f} ex/s)")
    print(f"    -> [speechbrain ErrorRateStats, comme au training] WER={sb_global_wer*100:.2f}%  CER={sb_global_cer*100:.2f}%")
    wer_np_str = f"{wer_no_punct*100:.2f}%" if wer_no_punct is not None else "N/A"
    cer_np_str = f"{cer_no_punct*100:.2f}%" if cer_no_punct is not None else "N/A"
    wer_bare_str = f"{wer_bare*100:.2f}%" if wer_bare is not None else "N/A"
    cer_bare_str = f"{cer_bare*100:.2f}%" if cer_bare is not None else "N/A"
    print(f"    -> Sans ponctuation (accents intacts) : WER={wer_np_str}  CER={cer_np_str}"
          f"  ({n_dropped_punct} réf. vidées)")
    print(f"    -> Sans ponctuation NI accents/tons    : WER={wer_bare_str}  CER={cer_bare_str}"
          f"  ({n_dropped_bare} réf. vidées)")

    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": run_label,
        "n_samples": n, "n_evaluated": n_evaluated, "n_skipped_audio": n_skipped,
        "n_dropped_empty_ref": dropped_empty_ref,
        "wer": global_wer, "cer": global_cer,
        "sb_wer": sb_global_wer, "sb_cer": sb_global_cer,
        "wer_no_punct": wer_no_punct, "cer_no_punct": cer_no_punct,
        "wer_bare": wer_bare, "cer_bare": cer_bare,
        "mean_char_sim": mean_char_sim, "mean_word_sim": mean_word_sim, "mean_semantic_sim": mean_semantic_sim,
        "mean_char_sim_no_punct": mean_char_sim_np, "mean_word_sim_no_punct": mean_word_sim_np,
        "mean_semantic_sim_no_punct": mean_semantic_sim_np,
        "mean_char_sim_bare": mean_char_sim_bare, "mean_word_sim_bare": mean_word_sim_bare,
        "mean_semantic_sim_bare": mean_semantic_sim_bare,
        "elapsed_s": elapsed, "per_sample": per_sample,
    }


def save_per_sample_csv(run_label: str, result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", run_label)
    out_path = os.path.join(out_dir, f"predictions_{safe_name}.csv")
    fieldnames = [
        "reference", "hypothesis", "wer", "cer",
        "char_ngram_cosine", "word_cosine", "semantic_cosine",
        "wer_no_punct", "cer_no_punct",
        "char_ngram_cosine_no_punct", "word_cosine_no_punct", "semantic_cosine_no_punct",
        "wer_bare", "cer_bare",
        "char_ngram_cosine_bare", "word_cosine_bare", "semantic_cosine_bare",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["per_sample"]:
            writer.writerow(row)
    print(f"  -> détail par échantillon : {out_path}")


def save_comparison_csv(results: List[dict], out_csv: str):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "n_samples", "n_evaluated", "n_skipped_audio", "n_dropped_empty_ref",
            "wer_%", "cer_%", "sb_wer_%", "sb_cer_%",
            "wer_no_punct_%", "cer_no_punct_%", "wer_bare_%", "cer_bare_%",
            "mean_char_ngram_cosine", "mean_word_cosine", "mean_semantic_cosine",
            "mean_char_ngram_cosine_no_punct", "mean_word_cosine_no_punct", "mean_semantic_cosine_no_punct",
            "mean_char_ngram_cosine_bare", "mean_word_cosine_bare", "mean_semantic_cosine_bare",
            "elapsed_s",
        ])

        def _fmt(v, mult=100):
            return f"{v * mult:.2f}" if v is not None else ""

        def _fmt4(v):
            return f"{v:.4f}" if v is not None else ""

        for r in sorted(results, key=lambda x: x["wer"]):
            writer.writerow([
                r["model"], r["n_samples"], r["n_evaluated"], r["n_skipped_audio"], r["n_dropped_empty_ref"],
                _fmt(r["wer"]), _fmt(r["cer"]), _fmt(r["sb_wer"]), _fmt(r["sb_cer"]),
                _fmt(r["wer_no_punct"]), _fmt(r["cer_no_punct"]),
                _fmt(r["wer_bare"]), _fmt(r["cer_bare"]),
                _fmt4(r["mean_char_sim"]), _fmt4(r["mean_word_sim"]), _fmt4(r["mean_semantic_sim"]),
                _fmt4(r["mean_char_sim_no_punct"]), _fmt4(r["mean_word_sim_no_punct"]), _fmt4(r["mean_semantic_sim_no_punct"]),
                _fmt4(r["mean_char_sim_bare"]), _fmt4(r["mean_word_sim_bare"]), _fmt4(r["mean_semantic_sim_bare"]),
                f"{r['elapsed_s']:.1f}",
            ])
    print(f"\nTableau comparatif : {out_csv}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hparams", action="append", required=True,
                   help="Chemin vers hparams.yaml (répétable, aligné avec --ckpt-dir).")
    p.add_argument("--ckpt-dir", action="append", required=True,
                   help="Dossier 'save/' du run correspondant (répétable, même ordre que --hparams).")
    p.add_argument("--test-parquet", default=None)
    p.add_argument("--test-manifest-csv", default=None,
                   help="Manifest CSV de prepare_ewe_manifests.py (colonnes ID,duration,wav,message) -- "
                        "bypass le parquet et lit les .wav déjà matérialisés bit-exact. "
                        "Alternative à --test-parquet pour isoler une éventuelle divergence de décodage.")
    p.add_argument("--train-parquet", default=None)
    p.add_argument("--val-parquet", default=None)
    p.add_argument("--text-column", default="transcription")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--trim-silence", action="store_true",
                   help="Active le trim silence (leur pipeline de training ne le fait PAS -- laisser désactivé par défaut).")
    p.add_argument("--trim-top-db", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--semantic-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="eval_results_ctc")
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()

    if len(a.hparams) != len(a.ckpt_dir):
        sys.exit("--hparams et --ckpt-dir doivent être fournis en même nombre, dans le même ordre.")
    if not a.test_parquet and not a.test_manifest_csv:
        sys.exit("Fournis --test-parquet OU --test-manifest-csv.")

    cfg = EvalConfig(
        hparams_path=a.hparams[0], ckpt_dir=a.ckpt_dir[0],
        test_parquet=a.test_parquet or "", test_manifest_csv=a.test_manifest_csv or "",
        text_column=a.text_column, audio_column=a.audio_column,
        sample_rate=a.sample_rate, trim_silence=a.trim_silence, trim_top_db=a.trim_top_db,
        batch_size=a.batch_size, semantic_model=a.semantic_model, limit=a.limit,
        out_dir=a.out_dir, out_csv=a.out_csv,
    )
    return cfg, list(zip(a.hparams, a.ckpt_dir)), a.train_parquet, a.val_parquet


def main():
    cfg, runs, train_parquet, val_parquet = parse_args()
    print(f"Device : {cfg.device}")

    if train_parquet:
        n_train, h_train = compute_split_hours(train_parquet, cfg.audio_column)
        print(f"Train : {n_train} utterances, {h_train:.2f} h")
    if val_parquet:
        n_val, h_val = compute_split_hours(val_parquet, cfg.audio_column)
        print(f"Val   : {n_val} utterances, {h_val:.2f} h")
    if cfg.test_parquet:
        n_test_h, h_test = compute_split_hours(cfg.test_parquet, cfg.audio_column)
        print(f"Test  : {n_test_h} utterances, {h_test:.2f} h")

    print(f"Chargement du test set : {cfg.test_parquet or cfg.test_manifest_csv}")
    df = load_test_set(cfg)
    print(f"{len(df)} échantillons de test chargés.")

    all_results = []
    for hparams_path, ckpt_dir in runs:
        result = evaluate_model(hparams_path, ckpt_dir, df, cfg)
        save_per_sample_csv(result["model"], result, cfg.out_dir)
        all_results.append(result)

    if len(all_results) > 1:
        print("\n=== Résumé comparatif ===")
        for r in sorted(all_results, key=lambda x: x["wer"]):
            wnp = f"{r['wer_no_punct']*100:.2f}%" if r["wer_no_punct"] is not None else "N/A"
            wb = f"{r['wer_bare']*100:.2f}%" if r["wer_bare"] is not None else "N/A"
            print(f"  {r['model']:<45s} WER={r['wer']*100:6.2f}%  CER={r['cer']*100:6.2f}%  "
                  f"WER_no_punct={wnp}  WER_bare={wb}")
        out_csv = cfg.out_csv or os.path.join(cfg.out_dir, "comparison_ctc.csv")
        save_comparison_csv(all_results, out_csv)


if __name__ == "__main__":
    main()
