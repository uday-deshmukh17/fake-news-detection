import logging
from deep_translator import GoogleTranslator
from langdetect import detect

logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to 'en'")
        return 'en'

def translate_text(text: str, source: str = 'auto', target: str = 'en') -> str:
    """
    Translate text using deep-translator (no API key required)
    
    Args:
        text: Text to translate
        source: Source language ('auto', 'hi', 'en')
        target: Target language ('en', 'hi')
    
    Returns:
        Translated text
    """
    try:
        # Skip if already in target language
        if source == target and source != 'auto':
            return text
        
        # Handle empty text
        if not text or len(text.strip()) == 0:
            return text
        
        # Translate
        translator = GoogleTranslator(source=source, target=target)
        
        # Split long text into chunks (5000 char limit)
        max_length = 4500
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            for chunk in chunks:
                try:
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                except Exception as e:
                    logger.warning(f"Chunk translation failed: {e}")
                    translated_chunks.append(chunk)
            translated = ' '.join(translated_chunks)
        else:
            translated = translator.translate(text)
        
        logger.info(f"Translated from {source} to {target}")
        return translated
    
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original if translation fails

def translate_batch(texts: list, source: str = 'auto', target: str = 'en') -> list:
    """Translate multiple texts"""
    try:
        translator = GoogleTranslator(source=source, target=target)
        translated_texts = []
        
        for text in texts:
            try:
                translated = translator.translate(text)
                translated_texts.append(translated)
            except Exception as e:
                logger.warning(f"Failed to translate: {e}")
                translated_texts.append(text)
        
        return translated_texts
    
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        return texts

def is_hindi(text: str) -> bool:
    """Check if text is in Hindi"""
    try:
        lang = detect(text)
        return lang == 'hi'
    except:
        # Check for Devanagari script
        return any('\u0900' <= char <= '\u097F' for char in text)

def is_english(text: str) -> bool:
    """Check if text is in English"""
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return text.isascii()