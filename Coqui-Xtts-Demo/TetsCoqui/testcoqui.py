import shutil
import os
from gradio_client import Client

# Étape 1 : Appel au modèle Gradio
client = Client("jimmyvu/Coqui-Xtts-Demo")
result = client.predict(
    input_text="Hi, I am a multilingual text-to-speech AI model.",
    speaker_id="Aaron Dreschner",
    temperature=0.3,
    top_p=0.85,
    top_k=50,
    repetition_penalty=9.5,
    language="Auto",
    api_name="/synthesize_speech"
)

# Étape 2 : Récupération du fichier audio
source_path = result[0]  # Chemin temporaire du fichier généré

# Étape 3 : Chemin du dossier où se trouve ce script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Étape 4 : Chemin final avec renommage facultatif
destination_path = os.path.join(script_dir, "audio_genere.wav")  # Vous pouvez changer le nom ici

# Étape 5 : Copie du fichier dans le dossier du script
shutil.copy(source_path, destination_path)

print(f"Fichier copié dans : {destination_path}")
