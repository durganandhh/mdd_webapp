"""Central config. Override any value with an environment variable.

MDD_MODEL          local folder OR Hugging Face repo id of the fine-tuned model
MDD_OUTPUT_DIR     where finetune/train.py saves the best model
SPEECHOCEAN_DIR    folder containing the extracted speechocean762 dataset
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get("MDD_MODEL", os.path.join(BACKEND_DIR, "models", "wav2vec2_phoneme"))
TRAIN_OUTPUT_DIR = os.environ.get("MDD_OUTPUT_DIR", MODEL_DIR)
SPEECHOCEAN_DIR = os.environ.get("SPEECHOCEAN_DIR", os.path.join(BACKEND_DIR, "data", "speechocean762"))

AUDIO_OUT_DIR = os.path.join(BACKEND_DIR, "correct_audio")
