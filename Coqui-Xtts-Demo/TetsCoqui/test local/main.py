import os
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==== CONFIGURATION ====
TEXT = "Bonjour, ceci est une démonstration de synthèse vocale locale avec XTTS."
OUTPUT_PATH = "output.wav"
SPEAKER_ID = "Aaron Dreschner"
LANGUAGE = "fr"  # ou "Auto"

# ==== DOSSIERS ====
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==== TÉLÉCHARGEMENT DU MODÈLE ====
print("📦 Téléchargement des fichiers du modèle...")
snapshot_download(
    repo_id="jimmyvu/xtts",
    local_dir=CACHE_DIR,
    allow_patterns=["*.safetensors", "*.json", "*.wav"],
    ignore_patterns=["*.onnx", "*.bin", "*.pt", "*.pth"]
)

# ==== CHARGEMENT DU MODÈLE ====
print("🚀 Chargement du modèle...")
config = XttsConfig()
config.load_json(os.path.join(CACHE_DIR, "config.json"))

model = Xtts.init_from_config(config)
model.load_safetensors_checkpoint(config, checkpoint_dir=CACHE_DIR)

if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.cpu()

# ==== PRÉPARATION DU LOCUTEUR ====
print("🎙️ Récupération du speaker embarqué...")
speaker_data = model.speaker_manager.speakers[SPEAKER_ID]
gpt_latent, speaker_embed = speaker_data["gpt_cond_latent"], speaker_data["speaker_embedding"]

# ==== SYNTHÈSE ====
print("🧠 Génération audio...")
output = model.inference(
    text=TEXT,
    language=LANGUAGE,
    gpt_cond_latent=gpt_latent,
    speaker_embedding=speaker_embed,
    temperature=0.3,
    top_p=0.85,
    top_k=50,
    repetition_penalty=9.5,
    enable_text_splitting=True,
)

# ==== SAUVEGARDE ====
print(f"💾 Sauvegarde dans {OUTPUT_PATH}")
torchaudio.save(OUTPUT_PATH, torch.tensor(output["wav"]).unsqueeze(0), 24000)

print("✅ Terminé ! Écoute le fichier output.wav")
