from fastapi import FastAPI
import gradio as gr
from Text2Audio import TTS
import os
from tempfile import NamedTemporaryFile
import soundfile as sf

# Initialize the FastAPI app
app = FastAPI()

# Create a TTS instance
tts = TTS()

def text_to_speech(text, voice_type=0):
    """
    Converts text to speech using the TTS class.

    Args:
        text (str): The input text to convert to speech.
        voice_type (int): The voice type index for the TTS.

    Returns:
        str: The path to the generated audio file.
    """
    # Generate the audio using TTS
    audio, _ = tts(text, voice_type=voice_type)
    
    # Save the audio to a temporary file
    temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Write the audio to the file
    sf.write(temp_file_path, audio, tts.sample_rate)
    return temp_file_path

# Gradio Interface
def generate_audio_ui(text, voice_type):
    """
    Generates audio and provides a link to the file.

    Args:
        text (str): The input text.
        voice_type (int): The selected voice type.

    Returns:
        tuple: Path to audio and phoneme output.
    """
    audio_path = text_to_speech(text, voice_type)
    return audio_path

# Define Gradio inputs and outputs
text_input = gr.Textbox(label="Enter text to convert to speech")
voice_dropdown = gr.Dropdown(
    choices=list(range(len(tts.voice_names))),
    label="Select Voice Type",
    type="index",
    value=0
)
audio_output = gr.Audio(label="Generated Audio")

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_audio_ui,
    inputs=[text_input, voice_dropdown],
    outputs=audio_output,
    title="Text-to-Speech Generator",
    description="Enter text and select a voice type to generate speech audio."
)

# Mount Gradio app on FastAPI
@app.get("/")
def read_root():
    return {"message": "Go to /gradio for the Gradio interface"}

@app.get("/gradio")
def gradio_app():
    return interface.launch(share=False, server_name="0.0.0.0", server_port=7860)