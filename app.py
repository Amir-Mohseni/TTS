import os
import logging
import tempfile
from threading import Lock

import gradio as gr
import soundfile as sf

from Text2Audio import TTS

locker = Lock()
tts = TTS()

def synthesize_audio(text_str: str, voice_name: str):
    """Convert text to speech using the specified voice type."""
    if not text_str.strip():
        logging.info("No text provided.")
        return None  # No audio

    with locker:
        try:
            # Synthesize audio
            tts.load_voice(voice_name)
            audio_array, *_ = tts(text_str)
            
            # Write to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "output.wav")
            sf.write(temp_path, audio_array, tts.sample_rate)
            
            # Return the path for Gradio's audio component
            return temp_path
        except Exception as e:
            logging.error(f"Error generating audio: {str(e)}")
            return None

def build_app():
    logging.basicConfig(level=logging.INFO)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 align="center">Text-to-Speech</h1>
            1. Paste your text below<br>
            2. Select a voice<br>
            3. Click Generate and listen to the result
            """
        )
        with gr.Row(variant="panel"):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Type something here..."
            )

        with gr.Row(variant="panel"):
            voice_dropdown = gr.Dropdown(
                label="Select Voice",
                choices=list(tts.voice_names),
                value=0,
            )

        with gr.Row(variant="panel"):
            generate_button = gr.Button("Generate")

        with gr.Row(variant="panel"):
            audio_output = gr.Audio(label="Generated Audio")

        # Bind the button to our function
        generate_button.click(
            fn=synthesize_audio,
            inputs=[text_input, voice_dropdown],
            outputs=audio_output
        )

    return demo

# HF Spaces looks for a variable named "app" in app.py
app = build_app()

# Optional for local debugging; won't really matter on HF Spaces.
if __name__ == "__main__":
    app.launch()