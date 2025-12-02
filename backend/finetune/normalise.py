import re


MODEL_IPA = {
    "a","b","d","e","f","h","i","j","k","l","m","n","o","p","s","t","u","v","w","z",
    "æ","ð","ŋ","ɑ","ɔ","ɚ","ɛ","ɡ","ɪ","ɹ","ʃ","ʊ","ʌ","ʒ","θ",
    "ˌ","͡"
}


REMOVE = ["ˈ", "ː", "̩", "ʲ", "ʷ", "̃", "̈"]

#rhotic vowels
RHOTICS = {
    "ɝ": "ɚ",
    "ɜ˞": "ɚ"
}

#diphthongs
DIPHTHONGS = {
    "oʊ": ["o","ʊ"],
    "aɪ": ["a","ɪ"],
    "aʊ": ["a","ʊ"],
    "eɪ": ["e","ɪ"],
    "ɔɪ": ["ɔ","ɪ"]
}

#affricates
AFFRICATES = {
    "tʃ": ["t","͡","ʃ"],
    "dʒ": ["d","͡","ʒ"]
}

#main
def normalize_ipa(text):

    # Step 1 — remove noise
    for mark in REMOVE:
        text = text.replace(mark, "")

    # Step 2 — rhotic vowels
    for src, tgt in RHOTICS.items():
        text = text.replace(src, tgt)

    # Step 3 — affricates
    for src, parts in AFFRICATES.items():
        text = text.replace(src, " ".join(parts))

    # Step 4 — diphthongs
    for src, parts in DIPHTHONGS.items():
        text = text.replace(src, " ".join(parts))

    # Step 5 — schwa → ʌ
    text = text.replace("ə", "ʌ")

    # Step 6 — syllabic consonants → remove diacritic
    text = text.replace("n̩", "n").replace("m̩", "m").replace("l̩", "l")

    # Step 7 — flap / tap mapping
    text = text.replace("ɾ", "t")

    # Step 8 — clean spacing
    text = re.sub(r"\s+", " ", text).strip()

    # Step 9 — split into final tokens
    tokens = []
    for piece in text.split():

        # if piece already valid, keep as a token
        if piece in MODEL_IPA:
            tokens.append(piece)
            continue

        # else split into chars
        for ch in piece:
            if ch in MODEL_IPA:
                tokens.append(ch)
            # else: drop unsupported symbol

    return " ".join(tokens)
