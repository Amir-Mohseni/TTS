# Use the base image from Hugging Face documentation
FROM python:3.9

# Add a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies for TTS
USER root
RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
USER user

# Copy and install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY --chown=user . /app

# Set the command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]