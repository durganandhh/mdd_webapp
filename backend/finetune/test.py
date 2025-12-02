#!/usr/bin/env python3

import os
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- config ---
MODEL_DIR = "/home/mcw/durga/best_wav2vec2_phoneme_model_v5"
TARGET_SR = 16000

# ----- LOAD MODEL -----
print("Loading model:", MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ----- AUDIO -----
def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)

# ----- PREDICT -----
def predict(audio):
    # numpy → python list
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32).tolist()

    # AudioInput format expected by your v5-trained processor
    batch = [{
        "array": audio,
        "sampling_rate": TARGET_SR
    }]

    # HuggingFace API: must call with audio=...
    inputs = processor(
        audio=batch,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(ids)[0]

    return text


# ----- TEST EXAMPLE -----
if __name__ == "__main__":
    WAV_PATH = "/home/mcw/durga/mispronunciation-detection-diagnosis-wav2vec2-and-llm/audio/SPELL/1.wav"

    print("\nTesting on:", WAV_PATH)
    audio = load_audio(WAV_PATH)

    pred_phonemes = predict(audio)

    print("\nPredicted IPA:")
    print(pred_phonemes)
    print("\nDone.\n")
