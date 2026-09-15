"""
fine_tune_afrihubert_ewe_asr.py
=================================
Adapté de fine_tune_afrihubert_fon_asr.py (fongbe, SpeechBrain) pour l'ewe.
Architecture INCHANGÉE (HuBERT + DNN + CTC, char-level tokenizer) -- seuls
la langue et le chargement de données changent (via prepare_ewe_manifests.py,
qui matérialise les .wav depuis tes parquets et produit les CSV manifests
attendus ici).

Correction vs l'original : `tokenizer.sp.decode_ids(...)` référence une
variable globale `tokenizer` (fonctionnellement correcte en Python -- le nom
global est résolu au moment de l'appel, après que `tokenizer = SentencePiece(...)`
ait été exécuté dans le bloc __main__ -- mais fragile si ce fichier est un
jour importé comme module plutôt qu'exécuté directement). Remplacé par
`self.tokenizer.sp.decode_ids(...)`, qui utilise l'attribut déjà assigné sur
`asr_brain` (cf. `asr_brain.tokenizer = tokenizer` dans __main__), sans
dépendre du scope global.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

import unicodedata

def normalize_text(text: str) -> str:
    """Equivalent Whisper metrics normalize: strip + lower (+ NFC)."""
    return unicodedata.normalize("NFC", str(text).strip().lower())



# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # Forward pass
        feats = self.modules.hubert(wavs, wav_lens)
        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

            # self.tokenizer plutôt que la variable globale `tokenizer` (cf.
            # docstring en tête de fichier).
            predicted_words = [
                normalize_text(self.tokenizer.sp.decode_ids(utt_seq)).split(" ")
                for utt_seq in sequence
            ]

            target_words = [normalize_text(wrd).split(" ") for wrd in batch.utterance]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_hubert, new_lr_hubert = self.hparams.lr_annealing_hubert(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.hubert.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.hubert_optimizer, new_lr_hubert
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_hubert": old_lr_hubert,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the hubert optimizer and model optimizer"

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if not self.hparams.hubert.freeze:
            self.hubert_optimizer = self.hparams.hubert_opt_class(
                self.modules.hubert.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "hubert_opt", self.hubert_optimizer
                )
            self.optimizers_dict = {
                "hubert_optimizer": self.hubert_optimizer,
                "model_optimizer": self.model_optimizer,
            }
        else:
            self.optimizers_dict = {"model_optimizer": self.model_optimizer}

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def freeze_optimizers(self, optimizers):
        """Freezes the hubert optimizer according to the warmup steps"""
        valid_optimizers = {}
        if not self.hparams.hubert.freeze:
            valid_optimizers["hubert_optimizer"] = optimizers[
                "hubert_optimizer"
            ]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """Prépare les datasets SpeechBrain à partir des manifests CSV produits
    par prepare_ewe_manifests.py (colonnes : ID, duration, wav, message)."""

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file,
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Audio pipeline : identique à l'original, lit le wav depuis le
    #    chemin produit par prepare_ewe_manifests.py.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        with torch.no_grad():
            sig = F.layer_norm(sig, sig.shape)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Text pipeline : "message" (produit par prepare_ewe_manifests.py),
    #    pas "translation" -- adapté au nom de colonne réel de nos manifests.
    @sb.utils.data_pipeline.takes("message")
    @sb.utils.data_pipeline.provides(
        "utterance", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(message):
        """Processes the transcriptions to generate proper labels"""
        message = normalize_text(message)
        yield message
        tokens_list = tokenizer.sp.encode_as_ids(message)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "duration",
            "utterance",
            "tokens_list",
            "tokens_bos",
            "tokens_eos",
            "tokens",
        ],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="message",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    train_data, valid_data, test_datasets = dataio_prepare(
        hparams, tokenizer
    )

    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.tokenizer = tokenizer

    # Chargement optionnel d'un checkpoint BEST-RQ pré-entraîné localement.
    # hparams.get(...) plutôt que hparams["pt_model_path"] is not None : dans
    # le yaml ewe, pt_model_path vaut "" (chaîne vide) par défaut, pas None
    # (nécessaire pour que l'interpolation !ref <pt_model_path>/hubert.ckpt
    # ne plante pas au chargement du yaml -- voir commentaire dans le yaml).
    # "" est falsy en Python, donc ce bloc est correctement sauté par défaut,
    # et le modèle part directement des poids AfriHuBERT du Hub (hubert_hub).
    if "pretrainer" in hparams.keys() and hparams.get("pt_model_path"):
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are valid, test
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )
