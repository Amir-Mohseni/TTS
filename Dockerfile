# Use a base image with Python and Linux
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure Git LFS
RUN git lfs install

# Download Whisper model
RUN mkdir -p /app/openai/whisper-small && \
    git clone https://huggingface.co/openai/whisper-small /app/openai/whisper-small

# Download SmolLM model
RUN mkdir -p /app/meta-llama/Llama-3.2-1B-Instruct && \
    git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct /app/meta-llama/Llama-3.2-1B-Instruct

# Optionally download Kokoro-82M
RUN mkdir -p /app/Kokoro_82M && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M /app/Kokoro_82M

# Copy all Python files to the container
COPY *.py ./

# Set the default command (optional)
#CMD ["bash"]
CMD ["python", "main.py"]