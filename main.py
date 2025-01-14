import gradio as gr
from Text2Audio import TextToAudio
from Transcriber import AudioTranscriber
import logging
from typing import Tuple, Optional
import os
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedApp:
    def __init__(self):
        """Initialize both converters."""
        self.tts = TextToAudio()
        self.transcriber = AudioTranscriber()
        
    def text_to_audio(self, text: str, voice_name: str) -> Tuple[str, str]:
        """Convert text to audio."""
        try:
            audio, phonemes = self.tts.generate_audio(text, voice_name)
            return audio, f"Phonemes: {phonemes}"
        except Exception as e:
            logger.error(f"Error in text_to_audio: {e}")
            return None, f"Error: {str(e)}"
        
    def text_to_audio_file(self, text: str, voice_name: str) -> str:
        """Convert text to audio and save to file."""
        try:
            audio, phonemes = self.text_to_audio(text, voice_name)
    
            os.makedirs("saved_audio", exist_ok=True)
            audio_path = "saved_audio/temp.wav"
            sf.write(audio_path, audio, self.tts.sample_rate)
            return audio_path, phonemes
        except Exception as e:
            logger.error(f"Error in text_to_audio_file: {e}")
            return None, f"Error: {str(e)}"
    
    def audio_to_text(self, audio) -> str:
        """Convert audio to text."""
        if not audio:
            return "No audio provided"
        
        result = self.transcriber.transcribe(audio)
        if result["success"]:
            return result["text"]
        return f"Error: {result['error']}"
    
    def chain_conversion(self, text: str, voice_name: str) -> str:
        """Convert text to audio and then back to text."""
        try:
            # First convert text to audio
            audio_path, _ = self.text_to_audio_file(text, voice_name)
            if not audio_path:
                return "Failed to generate audio"
                
            # Then convert audio back to text
            result = self.transcriber.transcribe(audio_path)
            if result["success"]:
                return result["text"]
            return f"Error in transcription: {result['error']}"
            
        except Exception as e:
            logger.error(f"Error in chain conversion: {e}")
            return f"Error: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface."""
    app = CombinedApp()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Text and Audio Conversion Tool")
        
        with gr.Tabs():
            # Text to Audio tab
            with gr.Tab("Text to Audio"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to convert to speech...",
                            lines=3
                        )
                        voice_dropdown = gr.Dropdown(
                            choices=app.tts.voice_names,
                            value=app.tts.voice_names[0],
                            label="Voice"
                        )
                        tts_button = gr.Button("🎯 Convert to Audio")
                    
                    with gr.Column():
                        audio_output = gr.Audio(label="Generated Audio", type="filepath")
                        phonemes_output = gr.Textbox(label="Phonemes", lines=2)
                
                tts_button.click(
                    fn=app.text_to_audio_file,
                    inputs=[text_input, voice_dropdown],
                    outputs=[audio_output, phonemes_output]
                )
            
            # Audio to Text tab
            with gr.Tab("Audio to Text"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Record or Upload Audio"
                        )
                        stt_button = gr.Button("🎯 Convert to Text")
                    
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Transcribed Text",
                            lines=3
                        )
                
                stt_button.click(
                    fn=app.audio_to_text,
                    inputs=audio_input,
                    outputs=text_output
                )
            
            # Chain Conversion tab
            with gr.Tab("Chain Conversion"):
                with gr.Row():
                    with gr.Column():
                        chain_text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to convert to speech and back...",
                            lines=3
                        )
                        chain_voice_dropdown = gr.Dropdown(
                            choices=app.tts.voice_names,
                            value=app.tts.voice_names[0],
                            label="Voice"
                        )
                        chain_button = gr.Button("🔄 Convert Text ➡️ Audio ➡️ Text")
                    
                    with gr.Column():
                        chain_output = gr.Textbox(
                            label="Results",
                            lines=5
                        )
                
                chain_button.click(
                    fn=app.chain_conversion,
                    inputs=[chain_text_input, chain_voice_dropdown],
                    outputs=chain_output
                )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )