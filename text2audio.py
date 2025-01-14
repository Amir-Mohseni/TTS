import torch
import sys
import os
import soundfile as sf
from typing import Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToAudio:
    def __init__(self, model_dir: Optional[str] = None, sample_rate: int = 24000):
        """
        Initialize Text to Audio converter.
        
        Args:
            model_dir (str): Path to Kokoro model directory
            sample_rate (int): Audio sample rate
        """
        # Get the Kokoro model directory
        if model_dir is None:
            current_dir = Path(__file__).parent
            model_dir = str(current_dir / "Kokoro_82M")
        
        sys.path.append(model_dir)
        
        # Import Kokoro modules
        from models import build_model
        from kokoro import generate
        self.generate = generate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(str(Path(model_dir) / 'kokoro-v0_19.pth'), self.device)
        self.voice_names = [
            'af',  # Default voice is a 50-50 mix of Bella & Sarah
            'af_bella', 'af_sarah', 'am_adam', 'am_michael',
            'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
            'af_nicole', 'af_sky',
        ]
        self.voice_name = None
        self.voice_pack = None
        self.sample_rate = sample_rate
        logger.info(f"Initialized TextToAudio with device: {self.device}")
    
    def load_voice(self, voice_type: int) -> None:
        """Load voice by index."""
        if voice_type < 0 or voice_type >= len(self.voice_names):
            voice_type = 0
            logger.warning(f"Invalid voice type. Using default voice")
            
        voice_name = self.voice_names[voice_type]
        self.load_voice_by_name(voice_name)
    
    def load_voice_by_name(self, voice_name: str) -> None:
        """Load voice by name."""
        if voice_name not in self.voice_names:
            raise ValueError(f"Voice name {voice_name} not found.")
            
        self.voice_name = voice_name
        voice_path = 'Kokoro_82M/voices/' + f'{voice_name}.pt'
        self.voice_pack = torch.load(str(voice_path), weights_only=True).to(self.device)
        logger.info(f'Loaded voice: {voice_name}')
    
    def generate_audio(self, text: str, voice_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate audio from text.
        
        Args:
            text (str): Input text
            voice_name (str, optional): Voice to use
            
        Returns:
            Tuple[str, str]: Path to generated audio file and phonemes
        """
        if voice_name:
            self.load_voice_by_name(voice_name)
        elif self.voice_pack is None:
            self.load_voice(0)
            
        try:
            audio, out_ps = self.generate(self.model, text, self.voice_pack, 
                                        lang=self.voice_name[0])
            return audio, out_ps
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
        
if __name__ == "__main__":
    tta = TextToAudio()
    tta.load_voice(0)
    audio, phonemes = tta.generate_audio("Hello, world!")
    # Save the audio to a file
    audio_path = "hello_world.wav"
    sf.write(audio_path, audio, tta.sample_rate)
    print(f"Audio saved to: {audio_path}")
    print(f"Phonemes: {phonemes}")
