"""
Fine-tuning SEGUE (declare-lab/segue-w2v2-base) on the MELD dataset.

Supports three modes via --task:
  sentiment        3-class sentiment only
  emotion          7-class emotion only
  multitask        both heads jointly (default)

Checkpoint averaging (--avg_checkpoints N) averages the weights of the last N
epoch checkpoints after training, following the SEGUE paper (N=10 default).

Dataset: https://github.com/declare-lab/MELD
Model:   https://github.com/declare-lab/segue
         https://huggingface.co/declare-lab/segue-w2v2-base

Setup
-----
1. Clone the SEGUE repo and install dependencies:
       git clone https://github.com/declare-lab/segue.git
       cd segue
       conda activate segue

   Compatible package versions:
       pip install transformers==4.35.0 datasets==2.14.0 huggingface_hub==0.17.0 \
                   pyarrow>=11.0,<14.0 numpy>=1.24,<2.0 accelerate>=0.20.1,<0.24.0

    Move finetune_meld.sh and finetune_meld.py (this script) into the segue
    directory.

2. Download MELD raw audio/video data:
       wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
       tar -xf MELD.Raw.tar.gz

3. Run from inside the cloned segue/ directory:
       python finetune_meld.py --data_dir /path/to/MELD.Raw --task multitask \
           --learning_rate 3e-5 --avg_checkpoints 10
"""

import argparse
import os
import sys
import copy
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

try:
    from segue.modeling_segue import SegueForClassification
except ImportError:
    sys.exit(
        "Could not import `segue`. Make sure you run this script from inside "
        "the cloned declare-lab/segue repository directory."
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
SENTIMENT_LABELS: Dict[str, int] = {"neutral": 0, "positive": 1, "negative": 2}
EMOTION_LABELS: Dict[str, int] = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}
EMOTION_CLASS_WEIGHTS = [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]

TARGET_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Checkpoint averaging
# ---------------------------------------------------------------------------
def average_checkpoints(checkpoint_dirs: List[str], model: torch.nn.Module) -> torch.nn.Module:
    """
    Load state dicts from `checkpoint_dirs`, average their parameters, and
    load the result into `model` (in-place). Returns the model.

    Shared parameters (e.g. the tied speech encoder in multitask mode) are
    handled correctly because we average by parameter name and then re-tie
    after loading.
    """
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories provided for averaging.")

    logger.info(f"Averaging {len(checkpoint_dirs)} checkpoints:")
    for d in checkpoint_dirs:
        logger.info(f"  {d}")

    # Collect state dicts, handling both safetensors and torch .pt files
    state_dicts = []
    for ckpt_dir in checkpoint_dirs:
        pt_path = os.path.join(ckpt_dir, "model.pt")
        st_path = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(pt_path):
            sd = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(st_path):
            from safetensors.torch import load_file
            sd = load_file(st_path, device="cpu")
        else:
            raise FileNotFoundError(
                f"No model.pt or model.safetensors found in {ckpt_dir}"
            )
        state_dicts.append(sd)

    # Average: sum then divide by count
    avg_sd = copy.deepcopy(state_dicts[0])
    for sd in state_dicts[1:]:
        for key in avg_sd:
            avg_sd[key] = avg_sd[key] + sd[key].to(avg_sd[key].dtype)
    n = len(state_dicts)
    for key in avg_sd:
        avg_sd[key] = avg_sd[key] / n

    # Load averaged weights — use strict=False to tolerate minor key mismatches
    missing, unexpected = model.load_state_dict(avg_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys after averaging: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys after averaging: {unexpected}")

    # Re-tie the speech encoder in multitask mode (shared pointer was lost on load)
    if isinstance(model, SegueMultiTask):
        model.emotion_model.speech_encoder = model.sentiment_model.speech_encoder
        logger.info("Re-tied shared speech encoder after checkpoint averaging.")

    return model


def find_checkpoint_dirs(output_dir: str, n: int) -> List[str]:
    """
    Return the last `n` checkpoint-* subdirectories sorted by step number.
    """
    pattern = os.path.join(output_dir, "checkpoint-*")
    dirs = sorted(
        glob.glob(pattern),
        key=lambda d: int(d.split("-")[-1]),
    )
    if not dirs:
        raise FileNotFoundError(
            f"No checkpoint directories found under {output_dir}. "
            "Make sure save_strategy='epoch' and checkpoints were not deleted."
        )
    selected = dirs[-n:]
    if len(selected) < n:
        logger.warning(
            f"Requested {n} checkpoints for averaging but only {len(selected)} found. "
            "Using all available."
        )
    return selected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MeldAudioDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        task: str,
        processor,
        max_duration_s: float = 10.0,
    ):
        self.task = task
        self.processor = processor
        self.max_samples = int(max_duration_s * TARGET_SAMPLE_RATE)
        self.audio_dir = Path(audio_dir)

        df = pd.read_csv(csv_path)
        df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
        df["Emotion"]   = df["Emotion"].str.strip().str.lower()

        df = df[
            df["Sentiment"].isin(SENTIMENT_LABELS) &
            df["Emotion"].isin(EMOTION_LABELS)
        ].reset_index(drop=True)

        self.dialogue_ids     = df["Dialogue_ID"].tolist()
        self.utterance_ids    = df["Utterance_ID"].tolist()
        self.sentiment_labels = [SENTIMENT_LABELS[v] for v in df["Sentiment"]]
        self.emotion_labels   = [EMOTION_LABELS[v]   for v in df["Emotion"]]

        logger.info(
            f"Loaded {len(self.sentiment_labels)} utterances from {csv_path} "
            f"(task={task})"
        )

    def __len__(self):
        return len(self.sentiment_labels)

    def _load_audio(self, dia_id: int, utt_id: int) -> np.ndarray:
        mp4_path = self.audio_dir / f"dia{dia_id}_utt{utt_id}.mp4"
        if not mp4_path.exists():
            logger.warning(f"Audio file not found: {mp4_path}. Using silence.")
            return np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        try:
            waveform, sr = torchaudio.load(str(mp4_path))
            waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)
            waveform = waveform[0, : self.max_samples]
            return waveform.numpy().astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to decode {mp4_path}: {e}. Using silence.")
            return np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict:
        audio = self._load_audio(self.dialogue_ids[idx], self.utterance_ids[idx])
        inputs = self.processor(audio=audio, sampling_rate=TARGET_SAMPLE_RATE)
        input_values    = torch.tensor(inputs["speech"]["input_values"], dtype=torch.float32)
        n_speech_tokens = int(inputs["n_speech_tokens"][0])

        item = {
            "input_values":    input_values,
            "n_speech_tokens": n_speech_tokens,
        }

        if self.task == "multitask":
            item["sentiment_labels"] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
            item["emotion_labels"]   = torch.tensor(self.emotion_labels[idx],   dtype=torch.long)
        elif self.task == "sentiment":
            item["labels"] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        else:
            item["labels"] = torch.tensor(self.emotion_labels[idx], dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorForMeld:
    processor: object
    task: str

    def __call__(self, features: List[Dict]) -> Dict:
        input_values    = [f["input_values"]    for f in features]
        n_speech_tokens = [f["n_speech_tokens"] for f in features]

        max_len = max(v.shape[0] for v in input_values)
        padded  = torch.zeros(len(input_values), max_len, dtype=torch.float32)
        for i, v in enumerate(input_values):
            padded[i, : v.shape[0]] = v

        batch = {
            "speech":          {"input_values": padded},
            "n_speech_tokens": n_speech_tokens,
        }

        if self.task == "multitask":
            batch["sentiment_labels"] = torch.stack([f["sentiment_labels"] for f in features])
            batch["emotion_labels"]   = torch.stack([f["emotion_labels"]   for f in features])
        else:
            batch["labels"] = torch.stack([f["labels"] for f in features])

        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def make_compute_metrics(task: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        if task == "multitask":
            # logits: (N, 10)  labels: (N, 2)
            sent_logits = logits[:, :3]
            emo_logits  = logits[:, 3:]
            sent_labels = labels[:, 0]
            emo_labels  = labels[:, 1]
            sent_preds  = np.argmax(sent_logits, axis=-1)
            emo_preds   = np.argmax(emo_logits,  axis=-1)
            return {
                "sentiment_weighted_f1": f1_score(sent_labels, sent_preds, average="weighted", zero_division=0),
                "sentiment_macro_f1":    f1_score(sent_labels, sent_preds, average="macro",    zero_division=0),
                "sentiment_accuracy":    float((sent_preds == sent_labels).mean()),
                "emotion_weighted_f1":   f1_score(emo_labels,  emo_preds,  average="weighted", zero_division=0),
                "emotion_macro_f1":      f1_score(emo_labels,  emo_preds,  average="macro",    zero_division=0),
                "emotion_accuracy":      float((emo_preds == emo_labels).mean()),
            }
        else:
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = np.argmax(logits, axis=-1)
            return {
                "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
                "macro_f1":    f1_score(labels, preds, average="macro",    zero_division=0),
                "accuracy":    float((preds == labels).mean()),
            }

    return compute_metrics


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------
class SegueMultiTask(torch.nn.Module):
    """Two SegueForClassification heads sharing one speech encoder backbone."""

    def __init__(
        self,
        model_name_or_path: str,
        sentiment_class_weights: Optional[torch.Tensor] = None,
        emotion_class_weights:   Optional[torch.Tensor] = None,
        sentiment_loss_weight: float = 0.5,
        emotion_loss_weight:   float = 0.5,
    ):
        super().__init__()
        self.sentiment_model = SegueForClassification.from_pretrained(
            model_name_or_path, n_classes=3, ignore_mismatched_sizes=True
        )
        self.emotion_model = SegueForClassification.from_pretrained(
            model_name_or_path, n_classes=7, ignore_mismatched_sizes=True
        )
        # Tie encoders
        self.emotion_model.speech_encoder = self.sentiment_model.speech_encoder

        # Disable feature masking in the speech encoder — short utterances
        # (< mask_length=10 frames) cause a ValueError with batch_size=1.
        # Masking is a pre-training trick; it's not needed for fine-tuning.
        self.sentiment_model.speech_encoder.config.mask_time_prob = 0.0
        self.sentiment_model.speech_encoder.config.mask_feature_prob = 0.0

        self.sentiment_class_weights = sentiment_class_weights
        self.emotion_class_weights   = emotion_class_weights
        self.sentiment_loss_weight   = sentiment_loss_weight
        self.emotion_loss_weight     = emotion_loss_weight
        self.processor = self.sentiment_model.processor

    def freeze_feature_extractor(self):
        self.sentiment_model.speech_encoder.feature_extractor._freeze_parameters()

    def forward(
        self,
        speech,
        n_speech_tokens,
        sentiment_labels: Optional[torch.Tensor] = None,
        emotion_labels:   Optional[torch.Tensor] = None,
        **kwargs,
    ):
        sent_out = self.sentiment_model(
            speech=speech, n_speech_tokens=n_speech_tokens, labels=sentiment_labels
        )
        emo_out = self.emotion_model(
            speech=speech, n_speech_tokens=n_speech_tokens, labels=emotion_labels
        )

        result = {
            "sentiment_predictions": sent_out["predictions"],
            "emotion_predictions":   emo_out["predictions"],
        }

        if sentiment_labels is not None and emotion_labels is not None:
            sent_loss = sent_out["loss"]
            if self.sentiment_class_weights is not None:
                w = self.sentiment_class_weights.to(sent_out["predictions"].device)
                sent_loss = torch.nn.CrossEntropyLoss(weight=w)(
                    sent_out["predictions"], sentiment_labels
                )
            emo_loss = emo_out["loss"]
            if self.emotion_class_weights is not None:
                w = self.emotion_class_weights.to(emo_out["predictions"].device)
                emo_loss = torch.nn.CrossEntropyLoss(weight=w)(
                    emo_out["predictions"], emotion_labels
                )
            result["loss"] = (
                self.sentiment_loss_weight * sent_loss +
                self.emotion_loss_weight   * emo_loss
            )

        return result


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------
class MeldTrainer(Trainer):
    def __init__(self, *args, task: str = "sentiment",
                 class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task          = task
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.task == "multitask":
            outputs = model(**inputs)
            loss    = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        else:
            labels  = inputs.get("labels")
            outputs = model(**inputs)
            loss    = outputs["loss"]
            if self.class_weights is not None and labels is not None:
                w    = self.class_weights.to(outputs["predictions"].device)
                loss = torch.nn.CrossEntropyLoss(weight=w)(outputs["predictions"], labels)
            return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """Use torch.save for multitask to avoid safetensors shared-tensor error."""
        if self.task == "multitask":
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        else:
            super()._save(output_dir, state_dict)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.task == "multitask":
                outputs = model(**inputs)
                loss    = outputs.get("loss")

                sent_logits = outputs["sentiment_predictions"].detach()
                emo_logits  = outputs["emotion_predictions"].detach()
                sent_labels = inputs.get("sentiment_labels")
                emo_labels  = inputs.get("emotion_labels")

                loss_val = loss.mean().detach() if loss is not None else None

                # Combined tensors so accelerate can pad/gather them normally
                combined_logits = torch.cat([sent_logits, emo_logits], dim=-1)
                combined_labels = torch.stack([sent_labels, emo_labels], dim=-1)

                return loss_val, combined_logits, combined_labels
            else:
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys
                )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SEGUE on MELD (sentiment / emotion / multitask)."
    )
    parser.add_argument("--data_dir",                    type=str,   required=True)
    parser.add_argument("--task",                        type=str,   default="multitask",
                        choices=["sentiment", "emotion", "multitask"])
    parser.add_argument("--model_name_or_path",          type=str,   default="declare-lab/segue-w2v2-base")
    parser.add_argument("--output_dir",                  type=str,   default="./meld_finetuned")
    parser.add_argument("--num_train_epochs",            type=int,   default=3,
                        help="Paper uses 3 with step-level checkpoint averaging.")
    parser.add_argument("--learning_rate",               type=float, default=3e-5)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=1,
                        help="Paper uses 1 with grad_accum=8 (effective batch 8).")
    parser.add_argument("--per_device_eval_batch_size",  type=int,   default=8)
    parser.add_argument("--warmup_ratio",                type=float, default=0.3,
                        help="Paper uses 0.3; longer warmup suits short 3-epoch runs.")
    parser.add_argument("--weight_decay",                type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=8,
                        help="Paper uses 8 with batch_size=1 for effective batch size of 8.")
    parser.add_argument("--fp16",                        action="store_true", default=False)
    parser.add_argument("--freeze_feature_extractor",    action="store_true", default=True)
    parser.add_argument("--seed",                        type=int,   default=39,
                        help="Paper uses seed 39.")
    parser.add_argument("--max_duration_s",              type=float, default=10.0)
    parser.add_argument("--use_class_weights",           action="store_true", default=False)
    parser.add_argument("--sentiment_loss_weight",       type=float, default=0.5,
                        help="Weight for sentiment loss in multitask (emotion gets 1 - this).")
    parser.add_argument(
        "--avg_checkpoints", type=int, default=10,
        help="Average the last N epoch checkpoints after training (0 = disabled). "
             "Paper uses 10. Disables early stopping and load_best_model_at_end."
    )
    parser.add_argument("--early_stopping_patience",    type=int,   default=5,
                        help="Ignored when --avg_checkpoints > 0.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save & eval every N steps when --avg_checkpoints > 0. Paper uses 100.")

    parser.add_argument("--metric_for_best_model",       type=str,   default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    emotion_loss_weight   = 1.0 - args.sentiment_loss_weight
    emotion_class_weights = (
        torch.tensor(EMOTION_CLASS_WEIGHTS, dtype=torch.float)
        if args.use_class_weights else None
    )

    use_averaging = args.avg_checkpoints > 0
    if use_averaging:
        logger.info(
            f"Checkpoint averaging enabled: will average last "
            f"{args.avg_checkpoints} checkpoints after training. "
            "Early stopping and load_best_model_at_end are disabled."
        )

    # ---- Model ----
    logger.info(f"Loading model: {args.model_name_or_path}")

    if args.task == "multitask":
        model = SegueMultiTask(
            model_name_or_path=args.model_name_or_path,
            emotion_class_weights=emotion_class_weights,
            sentiment_loss_weight=args.sentiment_loss_weight,
            emotion_loss_weight=emotion_loss_weight,
        )
        if args.freeze_feature_extractor:
            model.freeze_feature_extractor()
            logger.info("CNN feature extractor frozen.")
        processor     = model.processor
        class_weights = None
        best_metric   = args.metric_for_best_model or "sentiment_weighted_f1"
    else:
        n_classes = 3 if args.task == "sentiment" else 7
        model = SegueForClassification.from_pretrained(
            args.model_name_or_path, n_classes=n_classes, ignore_mismatched_sizes=True
        )
        if args.freeze_feature_extractor:
            model.speech_encoder.feature_extractor._freeze_parameters()
            logger.info("CNN feature extractor frozen.")
        # Disable masking to avoid short-sequence errors with batch_size=1
        model.speech_encoder.config.mask_time_prob = 0.0
        model.speech_encoder.config.mask_feature_prob = 0.0
        processor     = model.processor
        class_weights = emotion_class_weights if (args.task == "emotion" and args.use_class_weights) else None
        best_metric   = args.metric_for_best_model or "weighted_f1"

    # ---- Datasets ----
    def make_dataset(split):
        return MeldAudioDataset(
            csv_path=str(data_dir / f"{split}_sent_emo.csv"),
            audio_dir=str(data_dir / split),
            task=args.task,
            processor=processor,
            max_duration_s=args.max_duration_s,
        )

    train_dataset = make_dataset("train")
    dev_dataset   = make_dataset("dev")
    test_dataset  = make_dataset("test")
    collator      = DataCollatorForMeld(processor=processor, task=args.task)

    # ---- Training arguments ----
    # When averaging: use step-level save/eval (paper: every 100 steps) so that
    # the last N checkpoints cover the final portion of training rather than just
    # the last N epochs. Early stopping and load_best_model_at_end are disabled.
    if use_averaging:
        eval_strategy  = "steps"
        save_strategy  = "steps"
        save_steps     = args.save_steps
        eval_steps     = args.save_steps
        logging_steps  = args.save_steps
    else:
        eval_strategy  = "epoch"
        save_strategy  = "epoch"
        save_steps     = 500   # unused but required by TrainingArguments
        eval_steps     = 500
        logging_steps  = 50

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=None if use_averaging else 3,
        load_best_model_at_end=not use_averaging,
        metric_for_best_model=best_metric if not use_averaging else None,
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    callbacks = []
    if not use_averaging:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))

    trainer = MeldTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(args.task),
        task=args.task,
        class_weights=class_weights,
        callbacks=callbacks,
    )

    # ---- Train ----
    logger.info("Starting fine-tuning …")
    trainer.train()

    # ---- Checkpoint averaging ----
    if use_averaging:
        logger.info(f"Averaging last {args.avg_checkpoints} checkpoints …")
        ckpt_dirs = find_checkpoint_dirs(args.output_dir, args.avg_checkpoints)
        model = average_checkpoints(ckpt_dirs, model)
        # Move averaged model to the right device for evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # Swap the averaged model into the trainer for evaluation
        trainer.model = model

    # ---- Evaluate ----
    logger.info("Evaluating on dev set with final model …")
    dev_results = trainer.evaluate(eval_dataset=dev_dataset, metric_key_prefix="dev")
    logger.info(f"Dev results:  {dev_results}")

    logger.info("Evaluating on test set …")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info(f"Test results: {test_results}")

    # ---- Save final model ----
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    if args.task == "multitask":
        torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    else:
        trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Model saved to: {final_dir}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"Task              : {args.task}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"Checkpoint avg    : {args.avg_checkpoints if use_averaging else 'disabled'}")
    if use_averaging:
        print(f"  Averaged from   : {find_checkpoint_dirs(args.output_dir, args.avg_checkpoints)}")
    print()
    if args.task == "multitask":
        for split, results in [("dev", dev_results), ("test", test_results)]:
            prefix = split
            print(f"  {split.upper()} sentiment weighted-F1 : {results.get(f'{prefix}_sentiment_weighted_f1', 'N/A'):.4f}")
            print(f"  {split.upper()} sentiment macro-F1    : {results.get(f'{prefix}_sentiment_macro_f1',    'N/A'):.4f}")
            print(f"  {split.upper()} emotion   weighted-F1 : {results.get(f'{prefix}_emotion_weighted_f1',   'N/A'):.4f}")
            print(f"  {split.upper()} emotion   macro-F1    : {results.get(f'{prefix}_emotion_macro_f1',      'N/A'):.4f}")
            print()
    else:
        for split, results in [("dev", dev_results), ("test", test_results)]:
            prefix = split
            print(f"  {split.upper()} weighted-F1 : {results.get(f'{prefix}_weighted_f1', 'N/A'):.4f}")
            print(f"  {split.upper()} macro-F1    : {results.get(f'{prefix}_macro_f1',    'N/A'):.4f}")
            print(f"  {split.upper()} accuracy    : {results.get(f'{prefix}_accuracy',    'N/A'):.4f}")
            print()
    print("=" * 60)


if __name__ == "__main__":
    main()
