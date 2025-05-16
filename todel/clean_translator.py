import os
import re
import logging
import requests
import time
from pathlib import Path
from tqdm import tqdm

# Configuration
MODEL = "llama3.2"
LANGUAGE = "French"  # Target language
INPUT_DIR = "tmpInput"
OUTPUT_DIR = "tmpOutput"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def translate_text(text, index, previous_text=None, previous_translation=None, retries=3):
    """Send text to Ollama API for translation with context."""
    if not text.strip():
        return ""
    
    # Build context section if available
    context = ""
    if previous_text and previous_translation:
        context = f"""Context from previous subtitle:
Original: {previous_text}
Translation: {previous_translation}

"""
        
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

    for attempt in range(retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
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
            
            # Safety check - if translation is empty or significantly longer/shorter, return original
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

def process_srt_file(input_file, output_file):
    """Process an SRT file line by line, preserving timestamps and structure."""
    logger.info(f"Processing: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse SRT entries
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        entries = re.finditer(pattern, content, re.DOTALL)
        
        result = []
        entry_count = len(re.findall(pattern, content, re.DOTALL))
        
        # Store previous data for context
        previous_text = None
        previous_translation = None
        
        # Show progress bar
        with tqdm(total=entry_count, desc="Translating") as pbar:
            for match in entries:
                index = match.group(1)
                start_time = match.group(2)
                end_time = match.group(3)
                text = match.group(4).strip()
                
                # Translate the text with context
                logger.info(f"Translating entry {index}: {text}")
                translated_text = translate_text(
                    text, 
                    index, 
                    previous_text, 
                    previous_translation
                )
                logger.info(f"Translation: {translated_text}")
                # Store for context
                previous_text = text
                previous_translation = translated_text
                
                # Format the entry back
                result.append(index)
                result.append(f"{start_time} --> {end_time}")
                result.append(translated_text)
                result.append("")  # Empty line between entries
                
                pbar.update(1)
        
        # Write the result
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(result))
            
        logger.info(f"Translation complete: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")

def main():
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Process all SRT files in the input directory
    input_path = Path(INPUT_DIR)
    srt_files = list(input_path.glob("*.srt"))
    
    logger.info(f"Found {len(srt_files)} SRT files to process")
    
    for srt_file in srt_files:
        output_file = output_path / f"{srt_file.stem}_fr.srt"
        process_srt_file(srt_file, output_file)

if __name__ == "__main__":
    main() 