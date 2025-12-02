from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

backend = EspeakBackend(
    language="en-us",
    with_stress=True,
    preserve_punctuation=False
)


sep = Separator(phone="", word=" ")

print(backend.phonemize(["a"], separator=sep, strip=True))
print(backend.phonemize(["car"], separator=sep, strip=True))
print(backend.phonemize(["bird"], separator=sep, strip=True))
print(backend.phonemize(["better"], separator=sep, strip=True))
