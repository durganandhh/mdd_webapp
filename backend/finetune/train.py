#!/usr/bin/env python3
# coding: utf-8
"""Fine-tune wav2vec2 on SpeechOcean762 phoneme labels.

Run from the backend/ folder:  python -m finetune.train
"""

import os
import re
import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Union

from config import TRAIN_OUTPUT_DIR
from finetune.speechocean_loader import load_speechocean_dataset
from mdd.text_utils import (
    clean_text,
    text_to_espeak_ipa,
    normalize_ipa_string,
    detect_unsupported_characters,
)

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup
)


MODEL_NAME = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"
TARGET_SR = 16000
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
GRAD_ACC = 1
SEED = 42

EARLY_STOP_PATIENCE = 3
LOG_FILE = "ipa_samples_log.txt"



def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)



class SpeechOceanDataset(Dataset):
    def __init__(self, dataset_split, processor):
        self.data = dataset_split
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        wav_path = item["path"]
        text_raw = item["sentence"]

        audio = load_audio(wav_path)
        input_values = self.processor(audio, sampling_rate=TARGET_SR).input_values[0]

        text_clean = clean_text(text_raw)
        ipa_raw = text_to_espeak_ipa(text_clean)           # e.g. "keɪt lʌvz tʃaɪnə"
        ipa_norm = normalize_ipa_string(ipa_raw)           # e.g. "keɪt lʌvz t͡ʃaɪnə"

        # Logging sample for inspection
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n--- SAMPLE ---\n")
            f.write(f"Raw text:       {text_raw}\n")
            f.write(f"eSpeak IPA:     {ipa_raw}\n")
            f.write(f"Normalized IPA: {ipa_norm}\n")
            unsupported = detect_unsupported_characters(ipa_norm)
            if unsupported:
                f.write(f"Unsupported IPA chars (will be dropped/mapped): {unsupported}\n")

        # Prepare labels: pass IPA string directly to processor (string-level tokenizer)
        with self.processor.as_target_processor():
            label_ids = self.processor(ipa_norm).input_ids

        return {
            "input_values": torch.tensor(input_values, dtype=torch.float32),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(inputs, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(labels, return_tensors="pt")

        labels_padded = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels_padded
        return batch

def train():
    set_seed(SEED)
    open(LOG_FILE, "w", encoding="utf-8").close()  # reset the sample log once per run

    print("Loading SpeechOcean dataset...")
    train_raw, val_raw, _ = load_speechocean_dataset()

    print("Loading processor + model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    train_ds = SpeechOceanDataset(train_raw, processor)
    val_ds = SpeechOceanDataset(val_raw, processor)

    collator = DataCollatorCTCWithPadding(processor)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n==== EPOCH {epoch+1}/{EPOCHS} ====")

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=torch.cuda.is_available()):
                out = model(inputs, labels=labels)
                loss = out.loss / GRAD_ACC

            scaler.scale(loss).backward()

            if step % GRAD_ACC == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if step % 50 == 0:
                print(f"Step {step} - Loss {loss.item():.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                out = model(inputs, labels=labels)
                val_loss += out.loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print("New Model! Saving...")
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(TRAIN_OUTPUT_DIR)
            processor.save_pretrained(TRAIN_OUTPUT_DIR)
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("\n Early stopping triggered. Training finished.")
                return

        model.train()

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train()
