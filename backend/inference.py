#!/usr/bin/env python3

import os
import io
import subprocess
import torch
import ffmpeg
import soundfile as sf
import numpy as np

from config import MODEL_DIR, AUDIO_OUT_DIR
from mdd.text_utils import clean_text, text_to_espeak_ipa, normalize_ipa_string
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

TARGET_SR = 16000
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

print("Loading model:", MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# ---------------------------
# AUDIO LOAD
# ---------------------------
def load_audio(path):
    try:
        out, _ = (
            ffmpeg
            .input(path)
            .output("pipe:", format="wav", ac=1, ar=TARGET_SR)
            .run(capture_stdout=True, capture_stderr=True)
        )

        audio, sr = sf.read(io.BytesIO(out))
        return audio.astype(np.float32)

    except Exception as e:
        print("FFMPEG decode error:", e)
        raise e


# ---------------------------
# MODEL PREDICT
# ---------------------------
def predict(audio):
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    ipa_string = processor.batch_decode(pred_ids)[0]
    return ipa_string


# ---------------------------
# SMART IPA TOKENIZER
# ---------------------------
def tokenize(ipa_string):
    s = ipa_string.replace(" ", "")
    tokens = []
    i = 0

    IPA_MULTI = [
        "tʃ", "dʒ",
        "oʊ", "aɪ", "aʊ", "ɔɪ", "eɪ",
        "ɪə", "eə", "ʊə",
        "iː", "uː", "ɔː", "ɑː", "ɜː",
        "ɝ", "ɚ", "ɜ˞", "ə˞",
        "juː"
    ]

    while i < len(s):
        if i + 3 <= len(s) and s[i:i+3] in IPA_MULTI:
            tokens.append(s[i:i+3])
            i += 3
            continue
        if i + 2 <= len(s) and s[i:i+2] in IPA_MULTI:
            tokens.append(s[i:i+2])
            i += 2
            continue

        tokens.append(s[i])
        i += 1
    return tokens


# ---------------------------
# ALIGNMENT (NW / EDIT DISTANCE)
# ---------------------------
def align(ref, rec):
    n = len(ref)
    m = len(rec)

    dp = [[9999] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0)

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == rec[j - 1] else 1
            dp[i][j], back[i][j] = min(
                (dp[i - 1][j] + 1, (i - 1, j)),
                (dp[i][j - 1] + 1, (i, j - 1)),
                (dp[i - 1][j - 1] + cost, (i - 1, j - 1))
            )

    # Backtrack
    i, j = n, m
    ref_align, rec_align = [], []

    while i > 0 or j > 0:
        pi, pj = back[i][j]

        if pi == i - 1 and pj == j - 1:
            ref_align.append(ref[i - 1])
            rec_align.append(rec[j - 1])

        elif pi == i - 1 and pj == j:
            ref_align.append(ref[i - 1])
            rec_align.append("_")

        else:
            ref_align.append("_")
            rec_align.append(rec[j - 1])

        i, j = pi, pj

    return ref_align[::-1], rec_align[::-1]


# ---------------------------
# GENERATE AUDIO
# ---------------------------
def safe_filename(word):
    return "".join(c if c.isalnum() else "_" for c in word)


def generate_audio(word):
    name = safe_filename(word)
    out_path = f"{AUDIO_OUT_DIR}/{name}.wav"
    cmd = ["espeak", word, "-w", out_path]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("eSpeak error:", e)

    return out_path


# ---------------------------
# MAIN INFERENCE
# ---------------------------
def run_inference(audio_path, transcript):

    audio = load_audio(audio_path)

    # -------- MODEL IPA --------
    rec_ipa_string = predict(audio)
    rec_tokens = tokenize(rec_ipa_string)
    print("Recorded IPA:", rec_ipa_string)

    words = transcript.strip().split()
    expected_per_word = []

    for w in words:
        clean = clean_text(w)
        raw_ipa = text_to_espeak_ipa(clean)
        print(f"eSpeak IPA for '{w}': {raw_ipa}")

        norm_ipa = normalize_ipa_string(raw_ipa)
        toks = tokenize(norm_ipa)
        expected_per_word.append(toks)

    # Flatten reference
    ref_tokens = [p for word in expected_per_word for p in word]

    # -------- ALIGN --------
    ref_aligned, rec_aligned = align(ref_tokens, rec_tokens)

    # -------- FIXED REGROUPING (NO SHIFTING!!) --------
    ref_grouped = []
    rec_grouped = []

    idx = 0
    for exp in expected_per_word:
        target_len = len(exp)

        ref_slice = []
        rec_slice = []
        non_gap = 0

        while idx < len(ref_aligned) and non_gap < target_len:
            r = ref_aligned[idx]
            p = rec_aligned[idx]

            ref_slice.append(r)
            rec_slice.append(p)

            if r != "_":
                non_gap += 1

            idx += 1

        ref_grouped.append(ref_slice)
        rec_grouped.append(rec_slice)

    # -------- WORD MISTAKES --------
    mistaken_words = []
    audio_map = {}

    for w, ref_s, rec_s in zip(words, ref_grouped, rec_grouped):
        ref_join = "".join([x for x in ref_s if x != "_"])
        rec_join = "".join([x for x in rec_s if x != "_"])

        if ref_join != rec_join:
            mistaken_words.append(w)
            audio_map[w] = generate_audio(w)

    # -------- ACCURACY --------
    total = sum(1 for r in ref_aligned if r != "_")
    correct = sum(1 for r, p in zip(ref_aligned, rec_aligned) if r == p and r != "_")
    accuracy = round((correct / total) * 100, 2) if total else 0

    return {
        "accuracy": accuracy,
        "correction": {
            "words": words,
            "expected": ref_grouped,
            "recorded": rec_grouped,
            "mistaken_words": mistaken_words,
            "audio": audio_map
        }
    }
