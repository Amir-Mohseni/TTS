# Text-to-Speech (TTS) Application

This repository contains a Text-to-Speech (TTS) application. The main script is `Text2Audio.py`, which can be run using Docker. The model used is based on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

## Prerequisites

- Docker installed on your machine

## Running the Application

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Amir-Mohseni/tts.git
    cd tts
    ```

2. **Build the Docker image:**

    ```sh
    docker build -t tts-app .
    ```

3. **Run the Docker container:**

    ```sh
    docker run -it --rm -p 7860:7860 tts-app
    ```

    This will start the application and map port 7860 of the container to port 7860 on your host machine.

4. **Access the application:**

    Open your web browser and navigate to `http://127.0.0.1:7860` to use the TTS application.

## Files

- `app.py`: The gradio application.
- `Text2Audio.py`: The tts script.
- `Dockerfile`: The Dockerfile used to build the Docker image.
- `requirements.txt`: The file containing Python dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
