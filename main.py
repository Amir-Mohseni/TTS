import gradio as gr
from text2audio import TextToAudio
from transcriber import AudioTranscriber
from llm import LLM
import logging
from typing import Tuple, Optional
import os
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class App:
    def __init__(self):
        """Initialize converters and LLM."""
        self.tts = TextToAudio()
        self.transcriber = AudioTranscriber()
        self.llm = LLM("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.conversation_history = []
        self.audio_files = []
        
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
            audio_path = self.create_audio_file(audio)
            return audio_path, phonemes
        except Exception as e:
            logger.error(f"Error in text_to_audio_file: {e}")
            return None, f"Error: {str(e)}"

    def create_audio_file(self, audio, is_tuple: bool = False) -> str:
        os.makedirs("saved_audio", exist_ok=True)
        audio_path = f"saved_audio/audio_{len(self.audio_files)}.wav"
        if is_tuple:
            # Unpack the tuple
            samplerate, waveform = audio
        else:
            # Assume audio is a tuple
            samplerate, waveform = self.tts.sample_rate, audio
        sf.write(audio_path, waveform, samplerate)
        logger.info(f"Audio file saved to: {audio_path}")
        self.audio_files.append(audio_path)
        return audio_path
    
    def audio_to_text(self, audio_numpy) -> str:
        """Convert audio to text."""
        if audio_numpy is None:
            return "No audio provided"
        
        audio_path = self.create_audio_file(audio_numpy, is_tuple=True)
        result = self.transcriber.transcribe(audio_path)
        
        if result["success"]:
            return result["text"]
        return f"Error: {result['error']}"
    
    def process_conversation_turn(self, audio_input, voice_name: str) -> Tuple[str, str, str]:
        """Process one turn of audio conversation with the LLM."""
        try:
            # Convert audio to text
            print(audio_input)
            user_text = self.audio_to_text(audio_input)
            if user_text.startswith("Error:"):
                return None, user_text, "Conversation failed at transcription"
            
            # Process with LLM
            if len(self.conversation_history) == 0:
                messages = self.llm.new_conversation(user_text)
            else:
                messages = self.llm.add_user_query(self.conversation_history, user_text)
                
            new_messages = self.llm.generate(messages)
            llm_response = new_messages[-1]["content"]
            self.conversation_history = new_messages
            
            # Convert LLM response to audio
            audio_path, _ = self.text_to_audio_file(llm_response, voice_name)
            
            # Format conversation history for display
            display_text = self._format_conversation_history()
            
            return audio_path, display_text
            
        except Exception as e:
            logger.error(f"Error in conversation turn: {e}")
            return None, f"Error: {str(e)}", ""
            
    def _format_conversation_history(self) -> str:
        """Format conversation history for display."""
        formatted = ""
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role != "system":
                formatted += f"{role.capitalize()}: {content}\n\n"
        return formatted
    
    def reset_conversation(self) -> str:
        """Reset the conversation history."""
        self.conversation_history = []
        for audio_file in self.audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        self.audio_files = []
        return "Conversation reset. Start a new conversation!"

def create_gradio_interface():
    """Create the Gradio interface."""
    app = App()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Text and Audio Conversion Tool")
        
        with gr.Tabs():
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
                        audio_output = gr.Audio(label="Generated Audio", type="numpy")
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
                            type="numpy",
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
            
            # Audio Conversation tab
            with gr.Tab("Audio Conversation"):
                gr.Markdown("Have a conversation with the AI using voice!")
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Speak or Upload Audio",
                        )
                        voice_dropdown = gr.Dropdown(
                            choices=app.tts.voice_names,
                            value=app.tts.voice_names[0],
                            label="AI Voice"
                        )
                        send_button = gr.Button("🎤 Send Message")
                        reset_button = gr.Button("🔄 Reset Conversation")
                    
                    with gr.Column():
                        conversation_output = gr.TextArea(
                            label="Conversation History",
                            lines=10,
                            interactive=False
                        )
                        ai_audio_output = gr.Audio(
                            label="AI Response",
                            type="filepath"
                        )
                
                send_button.click(
                    fn=app.process_conversation_turn,
                    inputs=[audio_input, voice_dropdown],
                    outputs=[ai_audio_output, conversation_output]
                )
                
                reset_button.click(
                    fn=app.reset_conversation,
                    inputs=[],
                    outputs=[conversation_output]
                )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )