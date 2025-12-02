# speechocean_loader.py
"""
Updated SpeechOcean loader.

- DOES NOT cast audio column to datasets.Audio (avoids torchcodec)
- Optionally filters utterances using scores.json so only utterances where
  every phoneme has a perfect score (2.0) are included.

Usage:
    from speechocean_loader import load_speechocean_dataset
    train, val, test = load_speechocean_dataset(filter_perfect=True, scores_path="/path/to/scores.json")
"""

import os
import json
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
from datasets import Dataset


BASE = "/home/mcw/durga/mispronunciation-detection-diagnosis-wav2vec2-and-llm/speechocean762"
WAVE_DIR = os.path.join(BASE, "WAVE")


def _read_scores(scores_path: str) -> Dict[str, Any]:
  
    # Load scores.json which is expected to be a mapping from utt_id -> score entry.
    
    if not scores_path:
        return {}
    if not os.path.isfile(scores_path):
        return {}
    try:
        with open(scores_path, "r", encoding="utf-8") as fh:
            scores = json.load(fh)
        return scores
    except Exception:
        return {}


def _utterance_is_perfect(utt_id: str, scores: Dict[str, Any]) -> bool:
    """
    Allow utterances where average phone score >= 1.5 (instead of every phone = 2.0)
    """
    entry = scores.get(utt_id)
    if entry is None:
        return False

    words = entry.get("words", [])
    if not isinstance(words, list):
        return False

    all_scores = []

    for w in words:
        phones_acc = w.get("phones-accuracy")
        if not phones_acc:
            return False

        for p in phones_acc:
            try:
                v = float(p)
            except:
                return False
            all_scores.append(v)

    if not all_scores:
        return False

    # NEW RULE: average >= 1.5
    return (sum(all_scores) / len(all_scores)) >= 1.5


def load_split(split_name: str, scores: Dict[str, Any], filter_perfect: bool = True) -> Dataset:
    """
    Load a split (train/test etc.) and return a datasets.Dataset with columns:
      - 'path' : absolute path to wav file
      - 'sentence' : transcript string

    Params:
      split_name: "train" or "test" (should match your dir layout)
      scores: dict loaded from scores.json (or {} to disable)
      filter_perfect: if True, keep only utts where every phone accuracy == 2.0
    """
    text_file = os.path.join(BASE, split_name, "text")
    utt2spk_file = os.path.join(BASE, split_name, "utt2spk")

    if not os.path.isfile(text_file):
        raise FileNotFoundError(f"text file not found: {text_file}")
    if not os.path.isfile(utt2spk_file):
        raise FileNotFoundError(f"utt2spk file not found: {utt2spk_file}")

    # load utt->spk
    utt2spk: Dict[str, str] = {}
    with open(utt2spk_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                utt, spk = parts[0], parts[1]
                utt2spk[utt] = spk

    # load transcripts
    rows: List[Dict[str, str]] = []
    with open(text_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            # transcripts sometimes use TAB or single space separating uttid and text
            if "\t" in line:
                utt_id, sentence = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, sentence = parts[0], parts[1]

            if utt_id not in utt2spk:
                # warn and skip
                # print(f"Warning: utt2spk missing for {utt_id}")
                continue

            spk = utt2spk[utt_id]
            spk_folder = f"SPEAKER{spk.zfill(4)}"
            wav_path = os.path.join(WAVE_DIR, spk_folder, utt_id + ".WAV")

            if not os.path.isfile(wav_path):
                # missing audio
                # print(f"Missing audio: {wav_path}")
                continue

            if filter_perfect:
                ok = _utterance_is_perfect(utt_id, scores)
                if not ok:
                    continue

            rows.append({"path": wav_path, "sentence": sentence})

    # build pandas DataFrame and convert to datasets.Dataset
    df = pd.DataFrame(rows)
    return Dataset.from_pandas(df)


def load_speechocean_dataset(filter_perfect: bool = True, scores_path: Optional[str] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load train/validation/test datasets.

    - If filter_perfect==True, requires scores_path (defaults to BASE/scores.json)
      and will only keep utterances where every phone has accuracy == 2.0.

    Returns:
      (train_dataset, validation_dataset, test_dataset)
    """
    # load scores file
    if scores_path is None:
        scores_path = os.path.join(BASE, "resource", "scores.json")


    scores = _read_scores(scores_path) if filter_perfect else {}

    # load train + test splits and split train-> train/validation
    train_ds = load_split("train", scores=scores, filter_perfect=filter_perfect)
    test_ds = load_split("test", scores=scores, filter_perfect=filter_perfect)

    # create a small validation split from train (5%)
    if len(train_ds) > 1:
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        train = split["train"]
        validation = split["test"]
    else:
        train = train_ds
        validation = Dataset.from_pandas(pd.DataFrame(columns=["path", "sentence"]))

    return train, validation, test_ds
