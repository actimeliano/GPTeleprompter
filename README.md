# GPTeleprompter

GPTeleprompter is a real-time audio transcription and analysis tool designed to assist podcast creators and livestreamers. It uses advanced speech recognition and natural language processing to transcribe audio in multiple languages and provide insights.

## Current Features

- Real-time audio transcription using OpenAI's Whisper model
- Support for multiple languages
- WebSocket-based communication for live transcription updates
- Audio level checking to ensure quality transcriptions
- Filtering of repeated content and cleaning of transcriptions
- Web-based interface for easy interaction

## Planned Future Features

- Live summary generation of the ongoing content using GPT
- Topic extraction and display in a side panel
- Suggestion of future topics based on the current conversation and user-provided files
- Enhanced user interface with separate panels for transcription, summary, and topic suggestions

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/GPTeleprompter.git
   cd GPTeleprompter
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. Run the main application:
   ```
   python main.py
   ```

5. Open a web browser and navigate to `http://localhost:8080` (or the port specified in your configuration).

## Usage

1. Click the "Start Transcription" button to begin capturing audio from your microphone.
2. Speak or play your audio content.
3. The transcription will appear in real-time on the web interface.
4. Use the language selector to change the transcription language if needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- The open-source community for various libraries and tools used in this project
