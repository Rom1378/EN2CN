import re
import os
from TTS.api import TTS
import streamlit as st

@st.cache_resource
def load_tts_model():
    print("Chargement du modèle TTS...")
    model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
    # Liste des langues disponibles
    print("Langues disponibles :", model.languages)

# Liste des speakers disponibles
    print("Speakers disponibles :", model.speakers)
    return model

def srt_to_text(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Enlever les numéros de séquence (lignes composées uniquement de chiffres)
    content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)

    # Enlever les timestamps
    content = re.sub(r'^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}\s*$', '', content, flags=re.MULTILINE)

    # Enlever les lignes vides restantes
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    # Joindre en une seule chaîne
    text = ' '.join(lines)

    # Supprimer les caractères non désirés
    text = re.sub(r"[^a-zA-Z0-9\s.,?!:;'\"-]+", '', text)

    return text

import os

def text_to_audio_coqui(text,language="en"):
    tts_model = load_tts_model()
    output_path = os.path.join(os.getcwd(), "output.wav")  # Corrigé 'oautput' en 'output'
    
    # Prendre le premier speaker s'il existe
    speaker = tts_model.speakers[0] if hasattr(tts_model, 'speakers') and tts_model.speakers else None
    
    tts_model.tts_to_file(text=text, file_path=output_path, speaker=speaker,language=language)
    return output_path


def main():
    srt_file = os.path.join(os.getcwd(), "test.srt")
    if not os.path.exists(srt_file):
        print(f"Le fichier {srt_file} n'existe pas. Merci de placer un test.srt dans le dossier courant.")
        return

    print(f"Lecture du fichier : {srt_file}")
    text = srt_to_text(srt_file)
    print("Texte extrait des sous-titres :")
    print(text)

    print("Génération de l'audio TTS...")
    audio_path = text_to_audio_coqui(text, language="en")  # ou "fr", "de", etc.

    print(f"Audio généré et sauvegardé dans : {audio_path}")

if __name__ == "__main__":
    main()
