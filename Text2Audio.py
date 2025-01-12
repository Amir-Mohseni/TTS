import torch
import sys
import os

# Get the absolute path to the Kokoro_82M directory
current_dir = os.path.dirname(os.path.abspath(__file__))
kokoro_dir = os.path.join(current_dir, "Kokoro_82M")

# Add the Kokoro_82M directory to the Python path
sys.path.append(kokoro_dir)

# Import the modules
from models import build_model
from kokoro import generate
from IPython.display import display, Audio
import soundfile as sf

class TTS:
    def __init__(self, voice_pack=None, voice_name=None, sample_rate=24000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(kokoro_dir + '/' + 'kokoro-v0_19.pth', self.device)
        self.voice_names = [
            'af', # Default voice is a 50-50 mix of Bella & Sarah
            'af_bella', 'af_sarah', 'am_adam', 'am_michael',
            'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
            'af_nicole', 'af_sky',
        ]
        self.voice_name = voice_name
        self.voice_pack = voice_pack
        self.sample_rate = sample_rate
    
    def load_voice(self, voice_type: int):
        if voice_type < 0 or voice_type >= len(self.voice_names):
            voice_type = 0
            print(f"Invalid voice type. Using the default voice")
        VOICE_NAME = self.voice_names[voice_type]
        self.voice_name = VOICE_NAME
        VOICEPACK = torch.load(kokoro_dir + '/' + f'voices/{VOICE_NAME}.pt', weights_only=True).to(self.device)
        print(f'Loaded voice: {VOICE_NAME}')
        self.voice_pack = VOICEPACK
        
    def __call__(self, text: str, voice_type: int = 0):
        if self.voice_pack is None:
            self.load_voice(voice_type)
        audio, out_ps = generate(self.model, text, self.voice_pack, lang=self.voice_name[0])
        return audio, out_ps
    
    def play_audio(self, audio, out_ps):
        # Display the 24khz audio and print the output phonemes
        display(Audio(data=audio, rate=self.sample_rate, autoplay=True))
        print(out_ps)


# Example usage
if __name__ == "__main__":
    tts = TTS()
    audio, out_ps = tts("Once a year, go someplace you've never been before.", voice_type=0)
    # Save the audio to a file
    output_file = "output_audio.wav"
    sf.write(output_file, audio, tts.sample_rate)
    print(f"Audio saved as {output_file}")

    # Optionally play the audio
    tts.play_audio(audio, out_ps)