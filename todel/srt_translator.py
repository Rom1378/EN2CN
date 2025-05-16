import os
import re
import json
import logging
import time
import subprocess
import sys
from pathlib import Path
import requests
from typing import List, Dict, Tuple
from tqdm import tqdm

translationLanguage = "French"
#model = "llama3"
model = "gemma3"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ollama_running() -> bool:
    """Check if Ollama is running by attempting to connect to its API."""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama() -> None:
    """Start Ollama if it's not running."""
    if not check_ollama_running():
        logger.info("Ollama is not running. Starting Ollama...")
        try:
            if sys.platform == "win32":
                # Windows
                subprocess.Popen(["ollama", "serve"], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Unix-like systems
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            
            # Wait for Ollama to start
            max_retries = 30
            retry_count = 0
            while not check_ollama_running() and retry_count < max_retries:
                time.sleep(1)
                retry_count += 1
                logger.info(f"Waiting for Ollama to start... ({retry_count}/{max_retries})")
            
            if check_ollama_running():
                logger.info("Ollama started successfully")
            else:
                logger.error("Failed to start Ollama after maximum retries")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error starting Ollama: {str(e)}")
            sys.exit(1)
    else:
        logger.info("Ollama is already running")

class SRTTranslator:
    def __init__(self, input_dir: str, output_dir: str, batch_size: int = 5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.total_files = 0
        self.processed_files = 0
        self.total_entries = 0
        self.processed_entries = 0
        self.context = []  # Store previous translations for context
        
    def parse_srt(self, content: str) -> List[Dict]:
        """Parse SRT content into a list of subtitle entries."""
        entries = []
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            entry = {
                'index': int(match.group(1)),
                'start_time': match.group(2),
                'end_time': match.group(3),
                'text': match.group(4).strip()
            }
            entries.append(entry)
        
        logger.info(f"Parsed {len(entries)} subtitle entries")
        return entries

    def format_srt(self, entries: List[Dict]) -> str:
        """Format subtitle entries back into SRT format."""
        srt_content = []
        for entry in entries:
            srt_content.append(f"{entry['index']}")
            srt_content.append(f"{entry['start_time']} --> {entry['end_time']}")
            srt_content.append(entry['text'])
            srt_content.append("")
        
        return "\n".join(srt_content)

    def translate_single(self, text: str, index: int) -> str:
        """Translate a single text with context from previous translations."""
        try:
            logger.info(f"Translating entry {index}: {text}")
            start_time = time.time()
            
            # Build context from previous translations (keep last 3 for context)
            context_entries = self.context[-3:] if self.context else []
            context_str = "\n".join([f"{i+1}. {entry}" for i, entry in enumerate(context_entries)])
            
            prompt = f"""You are a professional translator specializing in sports content. Translate the following English text to {translationLanguage}.
            
            Context: This is a basketball tutorial video about finishing moves like Kyrie Irving.
            The speaker is explaining how to learn Kyrie Irving's finishing moves.
            
            Previous translations for context:
            {context_str}
            
            Translate this text:
            {text}
            
            Rules:
            1. Maintain the same formatting and line breaks
            2. Use consistent formality (use 'vous' for formal French)
            3. Keep translation natural and idiomatic
            4. Preserve the original meaning
            5. Only output the translation, nothing else
            6. Do not add any explanations or notes
            7. Maintain consistency with previous translations
            8. Translate basketball terms appropriately
            """
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False
                }
            )
            response.raise_for_status()
            result = response.json()
            
            translation = result['response'].strip()
            
            # Clean up translation
            lines = translation.split('\n')
            unique_lines = []
            for line in lines:
                if line.strip() and line not in unique_lines:
                    unique_lines.append(line)
            translation = '\n'.join(unique_lines)
            
            # Log translation
            logger.info(f"Original: {text}")
            logger.info(f"Translated: {translation}")
            
            # Add to context for future translations
            self.context.append(f"Original: {text}\nTranslation: {translation}")
            
            end_time = time.time()
            logger.info(f"Translation completed in {end_time - start_time:.2f} seconds")
            
            return translation
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

    def process_file(self, srt_file: Path) -> None:
        """Process a single SRT file."""
        try:
            logger.info(f"\nProcessing file: {srt_file.name}")
            start_time = time.time()
            
            # Reset context for new file
            self.context = []
            
            # Read the SRT file
            content = srt_file.read_text(encoding='utf-8')
            
            # Parse the SRT content
            entries = self.parse_srt(content)
            self.total_entries += len(entries)
            
            # Translate each entry individually with context
            for i, entry in enumerate(entries):
                logger.info(f"Translating entry {i+1}/{len(entries)}")
                entry['text'] = self.translate_single(entry['text'], i+1)
                self.processed_entries += 1
                
                # Log progress
                progress = (self.processed_entries / self.total_entries) * 100
                logger.info(f"Progress: {progress:.1f}% ({self.processed_entries}/{self.total_entries} entries)")
            
            # Format back to SRT
            translated_content = self.format_srt(entries)
            
            # Save the translated file
            output_file = self.output_dir / f"{srt_file.stem}_fr.srt"
            output_file.write_text(translated_content, encoding='utf-8')
            
            end_time = time.time()
            logger.info(f"Successfully translated {srt_file.name} in {end_time - start_time:.2f} seconds")
            self.processed_files += 1
            
        except Exception as e:
            logger.error(f"Error processing {srt_file.name}: {str(e)}")

    def process_directory(self) -> None:
        """Process all SRT files in the input directory."""
        srt_files = list(self.input_dir.glob('*.srt'))
        self.total_files = len(srt_files)
        logger.info(f"Found {self.total_files} SRT files to process")
        
        start_time = time.time()
        
        for srt_file in tqdm(srt_files, desc="Processing files"):
            self.process_file(srt_file)
            progress = (self.processed_files / self.total_files) * 100
            logger.info(f"Overall progress: {progress:.1f}% ({self.processed_files}/{self.total_files} files)")
        
        end_time = time.time()
        logger.info(f"\nTranslation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Processed {self.processed_files} files and {self.processed_entries} subtitle entries")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate SRT files using Ollama Gemma 3')
    parser.add_argument('input_dir', help='Input directory containing SRT files')
    parser.add_argument('output_dir', help='Output directory for translated SRT files')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of subtitle entries to translate in each batch')
    
    args = parser.parse_args()
    
    # Check and start Ollama if needed
    start_ollama()
    
    translator = SRTTranslator(args.input_dir, args.output_dir, args.batch_size)
    translator.process_directory()

if __name__ == '__main__':
    main() 