import streamlit as st
import re
import requests
import time
from io import StringIO
import yt_dlp
import tempfile
import os
import logging
import shutil
import google.generativeai as genai
from abc import ABC, abstractmethod
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
MODEL = "llama3.2"
LANGUAGE = "French"  # Target language
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Translation Service Interface
class TranslationService(ABC):
    @abstractmethod
    def translate(self, text: str, index: int, previous_text: str = None, previous_translation: str = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class OllamaTranslationService(TranslationService):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = OLLAMA_API_URL

    def is_available(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            available_models = response.json().get('models', [])
            logger.debug(f"Available Ollama models: {available_models}")
            return self.model_name in [model['name'] for model in available_models]
        except Exception as e:
            logger.error(f"Error checking Ollama API: {str(e)}")
            return False

    def translate(self, text: str, index: int, previous_text: str = None, previous_translation: str = None) -> str:
        if not text.strip():
            return ""
        
        logger.debug(f"Translating text with Ollama: {text}")
        
        context = ""
        if previous_text and previous_translation:
            context = f"""Context from previous subtitle:\nOriginal: {previous_text}\nTranslation: {previous_translation}\n\n"""
        
        prompt = f"""You are translating subtitles for a video. This is subtitle #{index}.

{context}Translate this English subtitle text to {LANGUAGE}:
    
"{text}"

Instructions for SRT subtitle translation:
- Return ONLY the translated text, nothing else
- Maintain the same line breaks exactly as in the original
- Keep the same formatting and style
- Use formal language (vous in French)
- Be concise - subtitles must be readable quickly
- Preserve timing-appropriate phrasing
- Maintain consistency with previous translations
- Translate basketball terminology appropriately
"""
        logger.debug(f"Translation prompt: {prompt}")

        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1}/3 to translate")
                response = requests.post(
                    self.api_url,
                    json={
                        'model': self.model_name,
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                translation = result['response'].strip()
                logger.debug(f"Received translation: {translation}")
                
                if not translation or len(translation) < len(text) * 0.5 or len(translation) > len(text) * 2:
                    logger.warning(f"Translation rejected (length issue): {translation}")
                    if attempt == 2:
                        return text
                    time.sleep(1)
                    continue
                return translation
            except Exception as e:
                logger.error(f"Translation error (attempt {attempt+1}/3): {str(e)}")
                if attempt == 2:
                    return text
                time.sleep(1)
        return text

class GeminiTranslationService(TranslationService):
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def is_available(self) -> bool:
        try:
            # Simple test to check if API key works
            response = self.model.generate_content("Test")
            return True
        except Exception as e:
            logger.error(f"Error checking Gemini API: {str(e)}")
            return False

    def translate(self, text: str, index: int, previous_text: str = None, previous_translation: str = None) -> str:
        if not text.strip():
            return ""
        
        logger.debug(f"Translating text with Gemini: {text}")
        
        context = ""
        if previous_text and previous_translation:
            context = f"""Context from previous subtitle:\nOriginal: {previous_text}\nTranslation: {previous_translation}\n\n"""
        
        prompt = f"""You are translating subtitles for a video. This is subtitle #{index}.

{context}Translate this English subtitle text to {LANGUAGE}:
    
"{text}"

Instructions for SRT subtitle translation:
- Return ONLY the translated text, nothing else
- Maintain the same line breaks exactly as in the original
- Keep the same formatting and style
- Use formal language (vous in French)
- Be concise - subtitles must be readable quickly
- Preserve timing-appropriate phrasing
- Maintain consistency with previous translations
- Translate basketball terminology appropriately
"""
        logger.debug(f"Translation prompt: {prompt}")

        try:
            response = self.model.generate_content(prompt)
            translation = response.text.strip()
            logger.debug(f"Received translation: {translation}")
            
            if not translation or len(translation) < len(text) * 0.5 or len(translation) > len(text) * 2:
                logger.warning(f"Translation rejected (length issue): {translation}")
                return text
            return translation
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

# --- Subtitle Download Logic ---
def download_subtitles(url):
    """Download video and subtitles from URL using yt-dlp"""
    logger.debug(f"Starting download for URL: {url}")
    
    # Create a permanent temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")
    
    try:
        # First, get available formats
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            logger.debug("Getting available formats")
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])
            logger.debug(f"Available formats: {[f.get('format_id') for f in formats]}")
            
            # Find the best format that's available
            best_format = None
            for f in formats:
                if f.get('height', 0) <= 720 and f.get('vcodec', 'none') != 'none':
                    best_format = f.get('format_id')
                    break
            
            if not best_format:
                best_format = 'best'  # Fallback to best available format
        
        # First download the video
        video_opts = {
            'format': best_format,
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'verbose': True,
        }
        
        logger.debug(f"Using format: {best_format}")
        logger.debug("Downloading video...")
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            video_info = ydl.extract_info(url, download=True)
            logger.debug(f"Video downloaded: {video_info.get('title', 'Unknown title')}")
        
        # Then try to download subtitles separately
        subtitle_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,  # Skip video download since we already have it
            'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'verbose': True,
        }
        
        logger.debug("Downloading subtitles...")
        try:
            with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                sub_info = ydl.extract_info(url, download=True)
                logger.debug("Subtitles downloaded successfully")
        except Exception as e:
            logger.warning(f"Error downloading subtitles: {str(e)}")
            # Try alternative subtitle download method
            try:
                subtitle_opts['writeautomaticsub'] = False
                with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                    sub_info = ydl.extract_info(url, download=True)
                    logger.debug("Subtitles downloaded with alternative method")
            except Exception as e2:
                logger.error(f"Failed to download subtitles with alternative method: {str(e2)}")
                # Continue without subtitles
        
        # Get the video file path
        video_path = os.path.join(temp_dir, 'video.mp4')
        
        # Get the subtitle file path
        subtitle_path = os.path.join(temp_dir, 'video.en.vtt')
        if not os.path.exists(subtitle_path):
            subtitle_path = os.path.join(temp_dir, 'video.en.srt')
        
        if os.path.exists(video_path):
            logger.debug(f"Video file exists at: {video_path}")
            if os.path.exists(subtitle_path):
                logger.debug(f"Subtitle file exists at: {subtitle_path}")
                return subtitle_path, video_path, video_info, temp_dir
            else:
                logger.warning("No subtitle file found, continuing with video only")
                return None, video_path, video_info, temp_dir
        else:
            logger.error(f"Video file not found at: {video_path}")
            logger.debug(f"Directory contents: {os.listdir(temp_dir)}")
            return None, None, None, temp_dir
                
    except Exception as e:
        logger.error(f"Error downloading video and subtitles: {str(e)}", exc_info=True)
        return None, None, None, temp_dir

def convert_vtt_to_srt(vtt_content):
    """Convert VTT format to SRT format"""
    # Remove VTT header and WEBVTT line
    lines = vtt_content.split('\n')
    while lines and (lines[0].startswith('WEBVTT') or not lines[0].strip()):
        lines.pop(0)
    
    # Convert timestamps and format
    srt_lines = []
    counter = 1
    i = 0
    
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
            
        # Timestamp line
        if '-->' in lines[i]:
            timestamp = lines[i].replace('.', ',')
            srt_lines.append(str(counter))
            srt_lines.append(timestamp)
            counter += 1
            i += 1
            
            # Text lines
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i])
                i += 1
            srt_lines.append('\n'.join(text_lines))
            srt_lines.append('')
        else:
            i += 1
    
    return '\n'.join(srt_lines)

# --- Translation Logic ---
def translate_text(text, index, previous_text=None, previous_translation=None, retries=3):
    if not text.strip():
        return ""
    
    logger.debug(f"Translating text: {text}")
    
    context = ""
    if previous_text and previous_translation:
        context = f"""Context from previous subtitle:\nOriginal: {previous_text}\nTranslation: {previous_translation}\n\n"""
    
    prompt = f"""You are translating subtitles for a video. This is subtitle #{index}.

{context}Translate this English subtitle text to {LANGUAGE}:
    
"{text}"

Instructions for SRT subtitle translation:
- Return ONLY the translated text, nothing else
- Maintain the same line breaks exactly as in the original
- Keep the same formatting and style
- Use formal language (vous in French)
- Be concise - subtitles must be readable quickly
- Preserve timing-appropriate phrasing
- Maintain consistency with previous translations
- Translate basketball terminology appropriately
"""
    logger.debug(f"Translation prompt: {prompt}")

    for attempt in range(retries):
        try:
            logger.debug(f"Attempt {attempt + 1}/{retries} to translate")
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    'model': MODEL,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            translation = result['response'].strip()
            logger.debug(f"Received translation: {translation}")
            
            if not translation or len(translation) < len(text) * 0.5 or len(translation) > len(text) * 2:
                logger.warning(f"Translation rejected (length issue): {translation}")
                if attempt == retries - 1:
                    return text
                time.sleep(1)
                continue
            return translation
        except Exception as e:
            logger.error(f"Translation error (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                return text
            time.sleep(1)
    return text

def parse_srt(content):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
    entries = list(re.finditer(pattern, content, re.DOTALL))
    return entries

def translate_srt(content, translator):
    entries = parse_srt(content)
    result = []
    previous_text = None
    previous_translation = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Group subtitles into batches of 5
    BATCH_SIZE = 10
    batches = []
    current_batch = []
    
    for match in entries:
        index = match.group(1)
        start_time = match.group(2)
        end_time = match.group(3)
        text = match.group(4).strip()
        
        current_batch.append({
            'index': index,
            'start_time': start_time,
            'end_time': end_time,
            'text': text
        })
        
        if len(current_batch) == BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
    
    if current_batch:  # Add any remaining entries
        batches.append(current_batch)
    
    # Process batches
    for batch_idx, batch in enumerate(batches):
        status_text.text(f"Translating batch {batch_idx + 1}/{len(batches)}")
        
        # Prepare batch text
        batch_text = "\n\n".join([
            f"Subtitle {entry['index']}:\n{entry['text']}"
            for entry in batch
        ])
        
        # Translate the batch
        batch_prompt = f"""Translate these English subtitles to {LANGUAGE}. Each subtitle is numbered and should be translated separately:

{batch_text}

Instructions:
- Return ONLY the translations, one per line
- Keep the same numbering format
- Maintain the same line breaks
- Use formal language
- Be concise
- Preserve timing-appropriate phrasing
- Maintain consistency
"""
        
        try:
            batch_translation = translator.translate(batch_text, batch_idx, previous_text, previous_translation)
            
            # Split the batch translation into individual translations
            translations = batch_translation.split("\n\n")
            
            # Process each translation in the batch
            for i, entry in enumerate(batch):
                if i < len(translations):
                    translated_text = translations[i].strip()
                    # Remove the subtitle number if present
                    translated_text = re.sub(r'^Subtitle \d+:\s*', '', translated_text)
                else:
                    translated_text = entry['text']  # Fallback to original if translation failed
                
                result.append(f"{entry['index']}\n{entry['start_time']} --> {entry['end_time']}\n{translated_text}\n")
                
                # Update context for next batch
                previous_text = entry['text']
                previous_translation = translated_text
        
        except Exception as e:
            logger.error(f"Error translating batch {batch_idx + 1}: {str(e)}")
            # If batch translation fails, translate individually
            for entry in batch:
                translated_text = translator.translate(
                    entry['text'],
                    entry['index'],
                    previous_text,
                    previous_translation
                )
                result.append(f"{entry['index']}\n{entry['start_time']} --> {entry['end_time']}\n{translated_text}\n")
                previous_text = entry['text']
                previous_translation = translated_text
        
        progress_bar.progress((batch_idx + 1) / len(batches))
    
    status_text.empty()
    progress_bar.empty()
    return "\n".join(result)

def display_srt_side_by_side(content, translated_content):
    orig_entries = parse_srt(content)
    trans_entries = parse_srt(translated_content)
    st.subheader("Original vs Translated Subtitles")
    for orig, trans in zip(orig_entries, trans_entries):
        st.markdown(f"**{orig.group(2)} → {orig.group(3)}**")
        cols = st.columns(2)
        cols[0].markdown(f"<div style='white-space: pre-wrap'>{orig.group(4).strip()}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='white-space: pre-wrap; color: #2b9348'>{trans.group(4).strip()}</div>", unsafe_allow_html=True)
        st.markdown("---")

def create_subtitle_json(srt_content):
    """Convert SRT content to JSON format for the video player"""
    entries = parse_srt(srt_content)
    subtitles = []
    for match in entries:
        start_time = match.group(2)
        end_time = match.group(3)
        text = match.group(4).strip()
        
        # Convert SRT time format to seconds
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        
        subtitles.append({
            'start': start_seconds,
            'end': end_seconds,
            'text': text
        })
    return json.dumps(subtitles)

def time_to_seconds(time_str):
    """Convert SRT time format (HH:MM:SS,mmm) to seconds"""
    h, m, s = time_str.replace(',', '.').split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def display_video_with_subtitles(video_path, original_srt, translated_srt):
    """Display video with both original and translated subtitles using Streamlit's native video player"""
    # Create columns for video and subtitles
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Convert SRT to VTT format for the original subtitles
        original_vtt = convert_srt_to_vtt(original_srt)
        # Convert SRT to VTT format for the translated subtitles
        translated_vtt = convert_srt_to_vtt(translated_srt)
        
        # Create subtitles dictionary for multiple tracks
        subtitles = {
            "Original": original_vtt,
            "Translated": translated_vtt
        }
        
        # Display the video using Streamlit's video player with subtitles
        st.video(video_path, subtitles=subtitles)
    
    with col2:
        st.markdown("### Subtitle Display")
        st.markdown("---")
        
        # Display subtitle files for download
        st.markdown("### Download Subtitles")
        st.download_button(
            label="Download Original SRT",
            data=original_srt,
            file_name="original.srt",
            mime="text/plain"
        )
        st.download_button(
            label="Download Translated SRT",
            data=translated_srt,
            file_name="translated.srt",
            mime="text/plain"
        )

def convert_srt_to_vtt(srt_content):
    """Convert SRT format to VTT format"""
    # Add VTT header
    vtt_content = "WEBVTT\n\n"
    
    # Convert each subtitle entry
    entries = parse_srt(srt_content)
    for match in entries:
        # Convert timestamp format from SRT to VTT
        start_time = match.group(2).replace(',', '.')
        end_time = match.group(3).replace(',', '.')
        text = match.group(4).strip()
        
        # Add the entry to VTT content
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    
    return vtt_content

# --- Streamlit UI ---
st.set_page_config(page_title="Subtitle Translator", layout="wide")
st.title("Subtitle Translator (EN → FR)")

st.write("Enter a video URL to automatically download and translate its subtitles to French.")

# Translation service selection
translation_service = st.radio(
    "Select Translation Service",
    ["Gemini API", "Ollama"],
    help="Choose which translation service to use"
)

# Initialize translation service
translator = None
if translation_service == "Gemini API":
    api_key = st.text_input("Enter your Gemini API key", type="password")
    if api_key:
        translator = GeminiTranslationService(api_key)
elif translation_service == "Ollama":
    translator = OllamaTranslationService(MODEL)

# Check if selected translation service is available
if translator and not translator.is_available():
    st.error(f"{translation_service} is not available. Please check your configuration.")
    st.stop()

video_url = st.text_input("Enter video URL", placeholder="https://www.youtube.com/watch?v=...")

if video_url and translator:
    if st.button("Process Video and Translate Subtitles"):
        try:
            with st.spinner("Downloading video and subtitles..."):
                logger.debug(f"Starting download for: {video_url}")
                subtitle_path, video_path, video_info, temp_dir = download_subtitles(video_url)
                
                if subtitle_path and video_path:
                    logger.debug(f"Successfully downloaded video and subtitles")
                    
                    try:
                        # Read and convert subtitles
                        logger.debug("Reading subtitle file")
                        with open(subtitle_path, 'r', encoding='utf-8') as f:
                            vtt_content = f.read()
                        logger.debug(f"Read {len(vtt_content)} characters from subtitle file")
                        
                        srt_content = convert_vtt_to_srt(vtt_content)
                        logger.debug("Converted VTT to SRT format")
                        
                        with st.spinner("Translating subtitles..."):
                            translated_srt = translate_srt(srt_content, translator)
                        st.success("Translation complete!")
                        
                        # Display video with subtitles
                        display_video_with_subtitles(video_path, srt_content, translated_srt)
                        
                        # Show side-by-side comparison
                        display_srt_side_by_side(srt_content, translated_srt)
                        
                        # Download buttons
                        st.download_button(
                            label="Download Translated SRT",
                            data=translated_srt,
                            file_name="translated_fr.srt",
                            mime="text/plain"
                        )
                    finally:
                        # Clean up the temporary directory
                        logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    logger.error("No subtitle path or video path returned from download_subtitles")
                    st.error("Could not download video or subtitles. Please check the URL and try again.")
        except Exception as e:
            logger.error(f"Error in main process: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}") 