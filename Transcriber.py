from transformers import pipeline
import logging
import tempfile
import os
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    A class to handle audio transcription using Whisper model.
    """
    
    def __init__(self, model_name: str = "openai/whisper-small", lang: str = "en"):
        """
        Initialize the AudioTranscriber with specified model.
        
        Args:
            model_name (str): Name of the Whisper model to use
        """
        self.model_name = model_name
        self.lang = lang
        self.asr_pipe = pipeline("automatic-speech-recognition", model=model_name)
        logger.info(f"Initialized AudioTranscriber with model: {model_name}")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe the audio file using Whisper model.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict[str, Any]: Dictionary containing transcription result or error
        """
        if not audio_path:
            return {"success": False, "error": "No audio found."}

        temp_file = None
        try:
            # Create a temporary file if needed
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                temp_file = tmp.name
                with open(audio_path, 'rb') as src:
                    tmp.write(src.read())

            # Transcribe the audio
            result = self.asr_pipe(temp_file)
            transcription = result["text"]
            logger.info("Transcription completed successfully")
            
            return {
                "success": True,
                "text": transcription,
                "error": None
            }
                   
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "success": False,
                "text": None,
                "error": f"Error during transcription: {str(e)}"
            }
            
        finally:
            # Clean up temporary files
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Cleaned up input file: {audio_path}")