"""Shared text -> IPA helpers.

Used by BOTH training (finetune/train.py) and inference (inference.py) so the
reference phonemes at inference time are produced by exactly the same rules the
model was trained on. Do not change these rules without retraining.
"""
import re

from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

chars_to_ignore = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

MODEL_IPA_SET = set([
    " ", "a","b","d","e","f","h","i","j","k","l","m","n","o","p","s","t","u","v","w","z",
    "æ","ð","ŋ","ɑ","ɔ","ɚ","ɛ","ɡ","ɪ","ɹ","ʃ","ʊ","ʌ","ʒ","θ",
    "ˌ","͡","[PAD]","[UNK]"
])

DIPHTHONGS = {"oʊ", "eɪ", "aɪ", "aʊ", "ɔɪ", "ju"}

AFFRICATE_REPLACEMENTS = {
    "tʃ": "t͡ʃ",
    "dʒ": "d͡ʒ"
}
IPA_REPLACEMENTS = {
    # rhotics
    "ɝ": "ɚ",
    "ɜ˞": "ɚ",
    # non-rhotic central vowel -> schwa
    "ɜ": "ʌ",
    "ɐ": "ʌ",
    # reduced vowel often printed as ᵻ -> map to ɪ
    "ᵻ": "ɪ",
    # flap -> approximant
    "ɾ": "ɹ",
    # optional: rare rounded/uncommon maps (safe defaults)
    "ə": "ʌ",
    "ɞ": "ɔ",
    # glottal stop -> remove
    "ʔ": "",
}
REMOVE_DIACRITICS = ["̩", "̃", "̈", "ʲ", "ʷ", "̯"] 
REMOVE_MARKS = ["ˈ", "ː"] 
REMOVE_MARKS.append("ˌ")


backend = EspeakBackend(language="en-us", with_stress=True, preserve_punctuation=False)
separator = Separator(phone="", word=" ")



def normalize_ipa_string(raw_ipa: str) -> str:
  
    if raw_ipa is None:
        return ""

    ipa = raw_ipa
    
    ipa = re.sub(r"\s+", " ", ipa).strip()
   
    for old, new in IPA_REPLACEMENTS.items():
        if old in ipa:
            ipa = ipa.replace(old, new)

    
    for old, new in AFFRICATE_REPLACEMENTS.items():
        ipa = ipa.replace(old, new)

   
    for d in REMOVE_DIACRITICS:
        ipa = ipa.replace(d, "")
    ipa = re.sub(r"[\u0300-\u036f]+", "", ipa) 

    
    for m in REMOVE_MARKS:
        ipa = ipa.replace(m, "")

    
    ipa = ipa.strip()

    # 7) Final cleanup: collapse multiple spaces
    ipa = re.sub(r"\s+", " ", ipa).strip()

    # 8) Validate characters: map or drop unsupported single characters (rare)
    out_chars = []
    i = 0
    L = len(ipa)
    while i < L:
        
        if ipa[i].isspace():
            out_chars.append(" ")
            i += 1
            continue

        # attempt to match a multi-character symbol first: tie-bar affricate or diphthong
        two = ipa[i:i+2]
        three = ipa[i:i+3]
        matched = False

        # check tie-bar affricates (length 3 like t͡ʃ)
        if three in {"t͡ʃ", "d͡ʒ"}:
            out_chars.append(three)
            i += 3
            matched = True
            continue

        # check diphthongs (2-char)
        if two in DIPHTHONGS:
            out_chars.append(two)
            i += 2
            matched = True
            continue

        # otherwise single character
        ch = ipa[i]
        if ch in MODEL_IPA_SET:
            out_chars.append(ch)
        else:
            pass
        i += 1

    # join and collapse spaces
    normalized = "".join(out_chars)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def detect_unsupported_characters(ipa_str: str):
    """Return set of characters present in ipa_str that are not in MODEL_IPA_SET and not space."""
    return {ch for ch in ipa_str if ch != " " and ch not in MODEL_IPA_SET}



# TEXT -> eSpeak IPA (wordwise)
def text_to_espeak_ipa(text: str) -> str:
    """
    Convert a raw sentence into a sequence of eSpeak IPA words joined by spaces.
    We phonemize per-word to avoid cross-word concatenation issues.
    """
    text_simple = Punctuation(';:,.!"?()').remove(text)
    words = [w.lower() for w in text_simple.split()]
    # phonemize words individually to preserve per-word boundaries
    ipa_words = [backend.phonemize([w], separator=separator, strip=True)[0] for w in words]
    return " ".join(ipa_words)


def clean_text(text):
    return re.sub(chars_to_ignore, "", text).lower().strip()
