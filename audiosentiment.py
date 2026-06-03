"""Audio sentiment analysis.

pip install torch torchaudio torchcodec transformers pandas soundfile

To use the SEGUE model, set SEGUE_MODEL_PATH to your fine-tuned model
checkpoint, e.g.:
    SEGUE_MODEL_PATH = "segue/meld_finetuned/final_model"
Set SEGUE_MODEL_PATH = None to skip the SEGUE model.
"""
import os
# os.environ['HF_HOME'] = '/your/dir/here/hf_home'

import sys
import random
from math import sqrt
from glob import glob
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from tqdm import tqdm

sys.path.append('segue')
from segue.modeling_segue import SegueForClassification

# ==========================
# Configuration
# ==========================
SAMPLE_RATE = 16000  # sample rate of input audio, in Hz
CHUNK_SECONDS = 2    # chunk size when using fixed chunks
MAX_UTTERANCE_SEC = 4.0   # longer utterances are split in chunks
SUB_CHUNK_SEC = 2.0       # length of chunks when splitting
MIN_UTTERANCE_SEC = 0.5   # minimal utterance duration

BATCH_SIZE = 16  # number of chunks to process at once on the GPU (models 1 & 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your fine-tuned SEGUE model directory (contains model.pt + processor files).
# SEGUE_MODEL_PATH = None
SEGUE_MODEL_PATH = "segue/meld_finetuned_avg/final_model"


# Encourage deterministic output
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ==========================
# Load models
# ==========================
print("Loading models...")

# Voice Activity Detection
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = vad_utils

vad_model = vad_model.to(DEVICE)
vad_model.eval()

# Model 1: Discrete emotions
emo_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
emo_extractor = AutoFeatureExtractor.from_pretrained(emo_model_name)
emo_model = AutoModelForAudioClassification.from_pretrained(
        emo_model_name).to(DEVICE)
emo_labels = emo_model.config.id2label

# Model 2: Valence / Arousal / Dominance
audeering_model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
audeering_extractor = AutoFeatureExtractor.from_pretrained(audeering_model_name)
audeering_model = AutoModelForAudioClassification.from_pretrained(
        audeering_model_name).to(DEVICE)

emo_model.eval()
audeering_model.eval()

print(f'\nLabels for {emo_model_name}:')
print(emo_model.config.id2label)

print(f'\nLabels for {audeering_model_name}:')
print(audeering_model.config.id2label)

# Model 3: SEGUE (fine-tuned on MELD, multitask sentiment + emotion)
SEGUE_SENTIMENT_LABELS = {0: "neutral", 1: "positive", 2: "negative"}
SEGUE_EMOTION_LABELS   = {
    0: "neutral", 1: "surprise", 2: "fear",
    3: "sadness", 4: "joy",      5: "disgust", 6: "anger",
}

# We load the multitask wrapper used during training.
# It is defined inline here so this script is self-contained when
# run from inside the segue repo directory.
class SegueMultiTask(torch.nn.Module):
    """Minimal inference-only wrapper matching the training definition."""
    def __init__(self, sentiment_model, emotion_model):
        super().__init__()
        self.sentiment_model = sentiment_model
        self.emotion_model   = emotion_model
        self.emotion_model.speech_encoder = self.sentiment_model.speech_encoder
        self.processor = self.sentiment_model.processor

    def forward(self, speech, n_speech_tokens, **kwargs):
        # SegueForClassification.forward() unconditionally calls labels.unsqueeze(-1),
        # so we must always pass labels — use a dummy zero tensor and ignore the loss.
        batch_size = speech["input_values"].shape[0]
        dummy_labels = torch.zeros(batch_size, dtype=torch.long,
                                   device=speech["input_values"].device)
        sent_out = self.sentiment_model(
            speech=speech, n_speech_tokens=n_speech_tokens, labels=dummy_labels
        )
        emo_out = self.emotion_model(
            speech=speech, n_speech_tokens=n_speech_tokens, labels=dummy_labels
        )
        return {
            "sentiment_predictions": sent_out["predictions"],
            "emotion_predictions":   emo_out["predictions"],
        }

print(f"\nLoading SEGUE model from: {SEGUE_MODEL_PATH}")

# Load the two heads from the base pretrained weights (architecture only),
# then overwrite with the fine-tuned state dict.
_base = "declare-lab/segue-w2v2-base"
_sent = SegueForClassification.from_pretrained(
    _base, n_classes=3, ignore_mismatched_sizes=True
)
_emo  = SegueForClassification.from_pretrained(
    _base, n_classes=7, ignore_mismatched_sizes=True
)
segue_model = SegueMultiTask(_sent, _emo)

# Disable masking (not needed for inference, avoids short-sequence errors)
segue_model.sentiment_model.speech_encoder.config.mask_time_prob    = 0.0
segue_model.sentiment_model.speech_encoder.config.mask_feature_prob = 0.0

state_dict = torch.load(
    os.path.join(SEGUE_MODEL_PATH, "model.pt"),
    map_location="cpu",
)
missing, unexpected = segue_model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"  Warning — missing keys: {missing}")
if unexpected:
    print(f"  Warning — unexpected keys: {unexpected}")

# Re-tie after load_state_dict breaks the shared reference
segue_model.emotion_model.speech_encoder = \
    segue_model.sentiment_model.speech_encoder

segue_model = segue_model.to(DEVICE)
segue_model.eval()
segue_processor = segue_model.processor

print("SEGUE model loaded successfully.")
print(f"  Sentiment labels: {SEGUE_SENTIMENT_LABELS}")
print(f"  Emotion labels:   {SEGUE_EMOTION_LABELS}")

# ==========================
# Speech segments (VAD)
# ==========================

def get_speech_segments(audio, sr):
    """Apply voice activity detection.
    audio: 1D numpy array
    returns: list of (start_sec, end_sec)"""
    audio_tensor = torch.from_numpy(audio).float().to(DEVICE)

    speech_ts = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
    )

    return [(ts["start"] / sr, ts["end"] / sr) for ts in speech_ts]


# ==========================
# Audio chunking
# ==========================

def chunk_audio(audio, sr, chunk_seconds):
    """Chunk audio without VAD."""
    chunk_size = int(chunk_seconds * sr)
    chunks, times = [], []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
            times.append((i / sr, (i + chunk_size) / sr))
    return chunks, times


def chunk_audio_with_vad(audio, sr, chunk_seconds):
    speech_segments = get_speech_segments(audio, sr)
    chunk_size = int(chunk_seconds * sr)
    chunks, times = [], []
    for seg_start, seg_end in speech_segments:
        start_sample = int(seg_start * sr)
        end_sample   = int(seg_end   * sr)
        for i in range(start_sample, end_sample, chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(chunk)
                times.append((i / sr, (i + chunk_size) / sr))
    return chunks, times


def hybrid_chunk_audio(audio, sr):
    """
    Uses VAD utterances: keeps short ones whole,
    splits long ones into fixed-size sub-chunks.
    """
    vad_segments = get_speech_segments(audio, sr)
    chunks, times = [], []
    max_len = int(MAX_UTTERANCE_SEC * sr)
    sub_len = int(SUB_CHUNK_SEC     * sr)
    min_len = int(MIN_UTTERANCE_SEC * sr)

    for seg_start, seg_end in vad_segments:
        start_s   = int(seg_start * sr)
        end_s     = int(seg_end   * sr)
        utterance = audio[start_s:end_s]

        if len(utterance) < min_len:
            continue
        if len(utterance) <= max_len:
            chunks.append(utterance)
            times.append((seg_start, seg_end))
        else:
            for i in range(0, len(utterance), sub_len):
                sub = utterance[i:i + sub_len]
                if len(sub) == sub_len:
                    chunks.append(sub)
                    times.append(
                        ((start_s + i) / sr, (start_s + i + sub_len) / sr)
                    )

    return chunks, times


# ==========================
# Batched inference (models 1 & 2)
# ==========================

def batched_predict(chunks, extractor, model):
    outputs = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch  = chunks[i:i + BATCH_SIZE]
        inputs = extractor(
            batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        outputs.append(logits.cpu())
    return torch.cat(outputs, dim=0)


# ==========================
# SEGUE inference (single-sample, no padding needed)
# ==========================

def segue_predict(chunks):
    """
    Run SEGUE on a list of 1-D float32 numpy audio arrays.

    Returns:
        sent_probs: np.ndarray (N, 3)   softmax probabilities for sentiment
        emo_probs:  np.ndarray (N, 7)   softmax probabilities for emotion
    """
    if segue_model is None:
        raise RuntimeError("SEGUE model is not loaded.")

    all_sent_logits = []
    all_emo_logits  = []

    for chunk in tqdm(chunks, desc="SEGUE inference"):
        proc = segue_processor(audio=chunk, sampling_rate=SAMPLE_RATE)
        input_values    = torch.tensor(
            proc["speech"]["input_values"], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        n_speech_tokens = [int(proc["n_speech_tokens"][0])]

        with torch.no_grad():
            out = segue_model(
                speech={"input_values": input_values},
                n_speech_tokens=n_speech_tokens,
            )

        all_sent_logits.append(out["sentiment_predictions"].cpu())
        all_emo_logits.append(out["emotion_predictions"].cpu())

    sent_logits = torch.cat(all_sent_logits, dim=0)
    emo_logits  = torch.cat(all_emo_logits,  dim=0)

    sent_probs = torch.softmax(sent_logits, dim=1).numpy()
    emo_probs  = torch.softmax(emo_logits,  dim=1).numpy()

    return sent_probs, emo_probs


# ==========================
# Combined sentiment mapping
# ==========================

def compute_combined_sentiment(emotion_probs, valence):
    """
    emotion_probs: dict of label -> probability (from model 1)
    valence: float in approx [-1, 1]  (from model 2)
    """
    emo_sentiment = (
        -0.25 * emotion_probs["angry"]
        + -0.25 * emotion_probs["sad"]
        + -0.25 * emotion_probs["fearful"]
        + -0.25 * emotion_probs["disgust"]
        + emotion_probs["happy"]
        + 0.25  * emotion_probs["neutral"]
    )
    combined = 0.5 * valence + 0.5 * emo_sentiment
    return float(np.clip(combined, -1.0, 1.0))


# ==========================
# Per-minute aggregation
# ==========================

def aggregate_sentiment_per_minute(chunk_rows, k=1.0):
    """
    Computes speech-weighted minute-level sentiment with confidence bands.
    chunk_rows: list of dicts with keys: start, end, segue_sent_score
    k: confidence multiplier (1 = soft, 2 = conservative)
    Returns: list of dicts (one per minute)
    """
    buckets = defaultdict(lambda: {
        "weighted_sum": 0.0,
        "weighted_sq_sum": 0.0,
        "speech_dur": 0.0,
    })

    for row in chunk_rows:
        start     = row["start"]
        end       = row["end"]
        sentiment = row["segue_sent_score"]
        cur = start
        while cur < end:
            minute     = int(cur // 60)
            minute_end = (minute + 1) * 60
            overlap    = min(end, minute_end) - cur
            buckets[minute]["weighted_sum"]    += sentiment * overlap
            buckets[minute]["weighted_sq_sum"] += (sentiment ** 2) * overlap
            buckets[minute]["speech_dur"]      += overlap
            cur += overlap

    minute_rows = []
    for minute in sorted(buckets.keys()):
        b = buckets[minute]
        T = b["speech_dur"]
        if T == 0:
            continue
        mean = b["weighted_sum"] / T
        var  = max((b["weighted_sq_sum"] / T) - (mean ** 2), 0.0)
        std  = sqrt(var)
        confidence_width = k * std / sqrt(T)
        minute_rows.append({
            "minute":           minute,
            "start_time":       minute * 60,
            "end_time":         (minute + 1) * 60,
            "speech_duration":  round(T, 3),
            "avg_sentiment":    round(mean, 4),
            "std_sentiment":    round(std, 4),
            "confidence_width": round(confidence_width, 4),
            "lower_bound":      round(mean - confidence_width, 4),
            "upper_bound":      round(mean + confidence_width, 4),
        })

    return minute_rows


# ==========================
# Process one movie
# ==========================

def process_movie(mp3_path, output_chunks, output_perminute, chunking='hybrid'):
    print(f"\nProcessing: {mp3_path}")

    waveform, sr = torchaudio.load(mp3_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    audio = waveform.squeeze(0).numpy()

    if chunking == 'hybrid':
        chunks, times = hybrid_chunk_audio(audio, SAMPLE_RATE)
    elif chunking == 'fixedvad':
        chunks, times = chunk_audio_with_vad(audio, SAMPLE_RATE, CHUNK_SECONDS)
    elif chunking == 'fixed':
        chunks, times = chunk_audio(audio, SAMPLE_RATE, CHUNK_SECONDS)
    else:
        raise ValueError(f"Unrecognized chunking strategy: {chunking}")

    # ---- Models 1 & 2 ----
    print("Running emotion model...")
    emo_logits      = batched_predict(chunks, emo_extractor, emo_model)
    print("Running audeering VAD model...")
    audeering_logits = batched_predict(chunks, audeering_extractor, audeering_model)

    emo_probs       = torch.softmax(emo_logits, dim=1).numpy()
    audeering_scores = audeering_logits.numpy()   # valence, arousal, dominance

    # ---- Model 3: SEGUE ----
    print("Running SEGUE model...")
    segue_sent_probs, segue_emo_probs = segue_predict(chunks)

    # ---- Build chunk rows ----
    chunk_rows = []
    for i, ((start, end), emo, vad) in enumerate(
        zip(times, emo_probs, audeering_scores)
    ):
        emo_dict = {emo_labels[j]: float(emo[j]) for j in range(len(emo))}
        combined_sentiment = compute_combined_sentiment(emo_dict, valence=vad[0])

        row = {
            "start":    start,
            "end":      end,
            "duration": end - start,

            # Model 1 — categorical emotion probabilities
            **emo_dict,

            # Model 2 — valence / arousal / dominance
            "valence":    float(vad[0]),
            "arousal":    float(vad[1]),
            "dominance":  float(vad[2]),

            # Combined sentiment (unchanged)
            "combined_sentiment": combined_sentiment,
        }

        # Model 3 — SEGUE sentiment probabilities (prefixed segue_sent_*)
        for j, label in SEGUE_SENTIMENT_LABELS.items():
            row[f"segue_sent_{label}"] = round(float(segue_sent_probs[i, j]), 6)

        # Model 3 — SEGUE emotion probabilities (prefixed segue_emo_*)
        for j, label in SEGUE_EMOTION_LABELS.items():
            row[f"segue_emo_{label}"] = round(float(segue_emo_probs[i, j]), 6)
       
        # sentiment score = prob(pos) - prob(neg)
        row[f"segue_sent_score"] = round(float(
                segue_sent_probs[i, 1] - segue_sent_probs[i, 2]), 6)

        chunk_rows.append(row)

    chunk_df = pd.DataFrame(chunk_rows)
    chunk_df.to_csv(output_chunks, index=False)

    minute_rows = aggregate_sentiment_per_minute(chunk_rows)
    minute_df   = pd.DataFrame(minute_rows)
    minute_df.to_csv(output_perminute, index=False)

    print(f"Saved → {output_chunks}")
    print(f"Saved → {output_perminute}")


# ==========================
# Batch process movies
# ==========================

def process_folder(input_folder, output_folder_chunks, output_folder_perminute,
        chunking):
    os.makedirs(output_folder_chunks,    exist_ok=True)
    os.makedirs(output_folder_perminute, exist_ok=True)

    for input_path in tqdm(glob(f'{input_folder}/*.mp3')):
        outfile = os.path.splitext(os.path.basename(input_path))[0] + ".csv"
        output_chunks    = os.path.join(output_folder_chunks,    outfile)
        output_perminute = os.path.join(output_folder_perminute, outfile)
        process_movie(
                input_path, output_chunks, output_perminute, chunking=chunking)


if __name__ == "__main__":
    process_folder(
        input_folder="audio",
        output_folder_chunks="no_vad-chunks",
        output_folder_perminute="no_vad-perminute",
        chunking='fixed'
    )
    process_folder(
        input_folder="audio",
        output_folder_chunks="vad-chunks",
        output_folder_perminute="vad-perminute",
        chunking='hybrid'
    )
