#!/usr/bin/env python
# coding: utf-8

import os, torch, torchaudio, warnings
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.punctuation import Punctuation
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from sequence_align.pairwise import needleman_wunsch

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# ============================================================
# CONFIG (CHANGE THESE)
# ============================================================
WAVE_ROOT = "/home/mcw/durga/mispronunciation-detection-diagnosis-wav2vec2-and-llm/speechocean762/WAVE"
TRANSCRIPT_FILE = "/home/mcw/durga/mispronunciation-detection-diagnosis-wav2vec2-and-llm/speechocean762/test/text"
MODEL_NAME = "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme"

# ============================================================
# LOAD WAV2VEC2 MODEL
# ============================================================
print("\nLoading model:", MODEL_NAME)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

# ============================================================
# IPA MULTI-CHAR PHONEMES
# ============================================================
IPA_MULTI = [
    "tʃ","dʒ",
    "oʊ","aɪ","aʊ","ɔɪ","eɪ",
    "ɝ","ɚ","ɜ˞",
    "ʃ","ʒ","θ","ð","ŋ","ɡ",
    "ɹ","ʊ","ɪ","ɛ","ɔ","ʌ"
]
IPA_MULTI = sorted(IPA_MULTI, key=len, reverse=True)

def tokenize_ipa(ipa_str):
    s = ipa_str.replace(" ", "").strip()
    tokens, i = [], 0
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

# ============================================================
# READ TRANSCRIPTS
# ============================================================
def read_transcripts(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            fid, text = line.strip().split(" ", 1)
            mapping[fid + ".WAV"] = text.strip()
    return mapping

# ============================================================
# RECORDED PHONEMES FROM AUDIO
# ============================================================
def get_recorded_phonemes(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    inputs = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()

# ============================================================
# REFERENCE PHONEMES FROM TEXT
# ============================================================
def get_reference_phonemes(text):
    text = Punctuation(';:,.!"?()').remove(text)
    words = [w.lower() for w in text.split()]
    backend = EspeakBackend('en-us')
    separator = Separator(phone="", word=None)
    ipa = [backend.phonemize([w], separator=separator, strip=True)[0] for w in words]
    return " ".join(ipa)

# ============================================================
# PRINT ALIGNMENT
# ============================================================
def pretty_print(ref, rec):
    print("\nAligned Phonemes:")
    print("REF:", " ".join(ref))
    print("REC:", " ".join(rec))

# ============================================================
# MAIN
# ============================================================
transcript_map = read_transcripts(TRANSCRIPT_FILE)

print("\nLoaded transcripts:", len(transcript_map))

# Loop through speaker folders
for speaker in sorted(os.listdir(WAVE_ROOT)):
    speaker_path = os.path.join(WAVE_ROOT, speaker)
    if not os.path.isdir(speaker_path):
        continue

    print(f"\n\n========== Processing Speaker: {speaker} ==========\n")

    # loop through WAV files inside speaker folder
    for file in sorted(os.listdir(speaker_path)):
        if not file.endswith(".WAV"):
            continue

        if file not in transcript_map:
            print("No transcript for:", file)
            continue

        text = transcript_map[file]
        audio_path = os.path.join(speaker_path, file)

        print("\n========================================")
        print("FILE:", file)
        print("TEXT:", text)
        print("========================================")

        # Recorded via wav2vec2
        rec_str = get_recorded_phonemes(audio_path)
        print("\nRecorded phonemes:", rec_str)

        # Reference via espeak
        ref_str = get_reference_phonemes(text)
        print("Reference phonemes:", ref_str)

        # Tokenize
        ref_tokens = tokenize_ipa(ref_str)
        rec_tokens = tokenize_ipa(rec_str)

        print("\nTokenized REF:", ref_tokens)
        print("Tokenized REC:", rec_tokens)

        # Alignment
        aligned_ref, aligned_rec = needleman_wunsch(
            ref_tokens, rec_tokens,
            match_score=1, mismatch_score=-1,
            indel_score=-1, gap="_"
        )

        pretty_print(aligned_ref, aligned_rec)

print("\nDONE.")
