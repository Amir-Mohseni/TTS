# Use a base image with Python and Linux
FROM python:3.9-slim
# Set the working directory inside the container
WORKDIR /app
# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    espeak-ng && \
    rm -rf /var/lib/apt/lists/*
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Clone the Hugging Face repository
RUN git lfs install && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M /app/Kokoro_82M
# Copy the Python file to the container
COPY Text2Audio.py .
# Set the default command (optional)
CMD ["bash"]