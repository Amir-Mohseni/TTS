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

# Clone the Hugging Face repository
RUN git lfs install && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M /app/Kokoro_82M && \
    git clone https://huggingface.co/openai/whisper-small /app/openai/whisper-small

# Copy all Python file to the container
COPY *.py ./

# Set the default command (optional)
CMD ["bash"]
#CMD ["python", "main.py"]