#!/usr/bin/env python
# coding: utf-8

import os
import re
import json
import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any, Union

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from speechocean_loader import load_speechocean_dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
TARGET_SR = 16000
BATCH_SIZE = 4
EPOCHS = 5
LR = 3e-4
GRAD_ACC = 1
SEED = 42

chars_to_ignore = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    return re.sub(chars_to_ignore, "", text).lower().strip()

# ============================================================
# AUDIO LOADING
# ============================================================
def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)

# ============================================================
# PHONEMIZER SETUP
# ============================================================
backend = EspeakBackend('en-us')
separator = Separator(phone="", word=None)

def text_to_ipa(text):
    return backend.phonemize([text], separator=separator, strip=True)[0]

# ============================================================
# IPA MULTI-CHAR TOKENS
# ============================================================
IPA_MULTI = [
    "tʃ","dʒ",
    "oʊ","aɪ","aʊ","ɔɪ","eɪ",
    "ɝ","ɚ","ɜ˞",
    "ʃ","ʒ","θ","ð","ŋ","ɡ",
    "ɹ","ʊ","ɪ","ɛ","ɔ","ʌ",
    "iː","uː","ɑː",
]
IPA_MULTI = sorted(IPA_MULTI, key=len, reverse=True)

def tokenize_ipa(ipa_str):
    s = ipa_str.replace(" ", "")
    tokens = []
    i = 0
    while i < len(s):
        matched = False
        for ph in IPA_MULTI:
            if s.startswith(ph, i):
                tokens.append(ph)
                i += len(ph)
                matched = True
                break
        if not matched:
            tokens.append(s[i])
            i += 1
    return tokens


def build_vocab(dataset):
    all_tokens = set()

    print("\nBuilding phoneme vocab...")

    for item in dataset:
        text = clean_text(item["sentence"])
        ipa = text_to_ipa(text)
        toks = tokenize_ipa(ipa)
        all_tokens.update(toks)

    vocab = {tok: i for i, tok in enumerate(sorted(list(all_tokens)))}
    vocab["[PAD]"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["|"] = len(vocab)

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("Saved vocab to vocab.json")
    return vocab

# ============================================================
# DATASET
# ============================================================
class SpeechOceanPhonemeDataset(Dataset):
    def __init__(self, dataset_split, processor, phoneme2id):
        self.data = dataset_split
        self.processor = processor
        self.phoneme2id = phoneme2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        wav_path = item["path"]
        text = clean_text(item["sentence"])

        audio = load_audio(wav_path)
        input_values = self.processor(audio, sampling_rate=TARGET_SR).input_values[0]

        # TEXT -> IPA
        ipa = text_to_ipa(text)

        # IPA -> TOKENS
        tokens = tokenize_ipa(ipa)

        # TOKENS -> IDS
        label_ids = [self.phoneme2id.get(t, self.phoneme2id["[UNK]"]) for t in tokens]

        return {
            "input_values": torch.tensor(input_values, dtype=torch.float),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

# ============================================================
# COLLATOR
# ============================================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(inputs, return_tensors="pt")

        labels_batch = self.processor.pad(labels, return_tensors="pt")

        labels_padded = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels_padded
        return batch

# ============================================================
# TRAIN LOOP
# ============================================================
def train():
    set_seed()

    print("Loading SpeechOcean dataset...")
    train_raw, val_raw, test_raw = load_speechocean_dataset()

    # -----------------------
    # Build vocab if missing
    # -----------------------
    if not os.path.exists("vocab.json"):
        vocab = build_vocab(train_raw + val_raw)
    else:
        vocab = json.load(open("vocab.json", "r", encoding="utf-8"))

    phoneme2id = vocab

    # -----------------------
    # Create processor from vocab
    # -----------------------
    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME,
        vocab_file="vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    # Load model
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        vocab_size=len(vocab)
    )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.ctc_loss_reduction = "mean"

    # Datasets
    train_ds = SpeechOceanPhonemeDataset(train_raw, processor, phoneme2id)
    val_ds = SpeechOceanPhonemeDataset(val_raw, processor, phoneme2id)

    collator = DataCollatorCTCWithPadding(processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # -----------------------
    # TRAINING LOOP
    # -----------------------
    model.train()
    step = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                output = model(inputs, labels=labels)
                loss = output.loss / GRAD_ACC

            scaler.scale(loss).backward()

            if step % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()

            if step % 50 == 0:
                print(f"Step {step} - Loss {loss.item():.4f}")

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                out = model(inputs, labels=labels)
                val_loss += out.loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        model.train()

    # -----------------------
    # SAVE MODEL
    # -----------------------
    SAVE_PATH = "./wav2vec2_finetuned_speechocean_phoneme"
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print("\nTraining complete!")
    print("Model saved to:", SAVE_PATH)


if __name__ == "__main__":
    train()
