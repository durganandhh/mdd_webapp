import subprocess
import os

def test_espeak(text, filename="test_espeak.wav"):
    print(f"Input Text: {text}")

    # Delete old file if exists
    if os.path.exists(filename):
        os.remove(filename)

    try:
        # 1) Get IPA / phonemes from eSpeak
        phoneme_result = subprocess.run(
            ["espeak", "-v", "en", "--ipa", text],
            capture_output=True,
            text=True
        )

        ipa_output = phoneme_result.stdout.strip()
        print("\nGenerated IPA / Phonemes:")
        print(ipa_output)

        # 2) Generate WAV audio
        audio_result = subprocess.run(
            ["espeak", "-v", "en", text, "-w", filename],
            capture_output=True,
            text=True
        )

        if audio_result.returncode != 0:
            print("\nError generating audio:")
            print(audio_result.stderr)
            return

        if os.path.exists(filename):
            print(f"\nSuccess! Audio generated: {filename}")
        else:
            print("\nFailed: No audio file created.")

    except FileNotFoundError:
        print("eSpeak not found. Install it first:")
        print("Ubuntu:   sudo apt install espeak")
       


# Test it
if __name__ == "__main__":
    test_espeak("These toys are made from tough plastic", "hello_espeak.wav")
