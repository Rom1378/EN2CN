import os
import logging
import google.generativeai as genai
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TranslationService(ABC):
    @abstractmethod
    def translate(self, text: str, index: int, previous_text: str = None, previous_translation: str = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class GeminiTranslationService(TranslationService):
    def __init__(self, api_key: str, target_language: str):
        """Initialize the Gemini translation service.
        
        Args:
            api_key (str): The Gemini API key
            target_language (str): The language to translate to
        """
        if not api_key:
            raise ValueError("api_key is required")
        if not target_language:
            raise ValueError("target_language is required")
            
        self.target_language = target_language
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

{context}Translate this English subtitle text to {self.target_language}:
    
"{text}"

Instructions for SRT subtitle translation:
- Return ONLY the translated text, nothing else
- Maintain the same line breaks exactly as in the original
- Keep the same formatting and style
- Use formal language
- Be concise - subtitles must be readable quickly
- Preserve timing-appropriate phrasing
- Maintain consistency with previous translations
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