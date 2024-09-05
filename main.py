import asyncio
import websockets
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
import os
import logging
import json
from openai import OpenAI
import numpy as np
import io
import secrets
import time
import wave
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set up OpenAI API key
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

client = OpenAI(api_key=openai_api_key)

# Generate a secret key for WebSocket authentication
WS_SECRET_KEY = secrets.token_urlsafe(32)

def check_audio_levels(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    max_amplitude = np.max(np.abs(audio_np))
    rms = np.sqrt(np.mean(np.square(audio_np)))
    logging.info(f"Server-side - Max amplitude: {max_amplitude}, RMS: {rms}")
    return max_amplitude > 0.1 and rms > 0.01  # Adjust these thresholds as needed

def normalize_audio(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    max_amplitude = np.max(np.abs(audio_np))
    if max_amplitude > 0:
        normalized_audio = audio_np / max_amplitude
    else:
        normalized_audio = audio_np
    return normalized_audio.tobytes()

class WebSocketHandler:
    def __init__(self):
        self.active_connections = set()
        self.previous_transcription = ""

    def filter_repeated_content(self, text):
        # Remove excessive repetitions
        words = text.split()
        filtered_words = []
        for word in words:
            if len(filtered_words) < 2 or word.lower() != filtered_words[-1].lower() or word.lower() != filtered_words[-2].lower():
                filtered_words.append(word)
        return ' '.join(filtered_words)

    def clean_transcription(self, text):
        # Remove excessive punctuation and spaces
        text = re.sub(r'([.!?])\1+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def handle(self, request):
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        
        logging.info(f"New WebSocket connection attempt from {request.remote}")
        try:
            # Perform authentication and get language
            auth_message = await websocket.receive_json()
            if auth_message.get('secret') != WS_SECRET_KEY:
                await websocket.send_json({"error": "Authentication failed"})
                await websocket.close()
                return websocket

            selected_language = auth_message.get('language', '')

            self.active_connections.add(websocket)
            logging.info(f"Authenticated WebSocket connection from {request.remote}")

            audio_buffer = []
            last_transcription_time = time.time()

            async for msg in websocket:
                if msg.type == web.WSMsgType.BINARY:
                    audio_data = np.frombuffer(msg.data, dtype=np.float32)
                    audio_buffer.extend(audio_data)

                    if check_audio_levels(audio_data):
                        # Check if we have enough audio data or if enough time has passed
                        if len(audio_buffer) >= 16000 * 30 or time.time() - last_transcription_time >= 30:
                            wav_file = io.BytesIO()
                            with wave.open(wav_file, 'wb') as wav:
                                wav.setnchannels(1)
                                wav.setsampwidth(2)
                                wav.setframerate(16000)
                                wav_data = normalize_audio((np.array(audio_buffer) * 32767).astype(np.int16).tobytes())
                                wav.writeframes(wav_data)
                            
                            wav_file.seek(0)
                            wav_size = wav_file.getbuffer().nbytes
                            logging.info(f"WAV file created. Size: {wav_size} bytes")

                            try:
                                transcription_params = {
                                    "model": "whisper-1",
                                    "file": ("audio.wav", wav_file, "audio/wav"),
                                    "response_format": "text",
                                    "prompt": f"Transcreva o áudio em {selected_language}. Mantenha a pontuação e capitalização adequadas.",
                                    "temperature": 0.3,
                                    "language": selected_language
                                }

                                response = client.audio.transcriptions.create(**transcription_params)
                                transcription = response

                                if not transcription:
                                    logging.warning("Received empty transcription")
                                    await websocket.send_json({"warning": "Received empty transcription"})
                                else:
                                    filtered_transcription = self.filter_repeated_content(transcription)
                                    cleaned_transcription = self.clean_transcription(filtered_transcription)
                                    self.previous_transcription += " " + cleaned_transcription
                                    await websocket.send_json({"transcription": cleaned_transcription})
                            except Exception as e:
                                logging.error(f"Error during transcription: {e}", exc_info=True)
                                await websocket.send_json({"error": f"Transcription error: {str(e)}"})
                            finally:
                                wav_file.close()
                            
                            # Reset buffer and update last transcription time
                            audio_buffer = []
                            last_transcription_time = time.time()
                    else:
                        logging.warning("Low audio levels detected. Skipping transcription.")
                        await websocket.send_json({"warning": "Low audio levels detected. Please speak louder or check your microphone."})
                elif msg.type == web.WSMsgType.ERROR:
                    logging.error(f'WebSocket connection closed with exception {websocket.exception()}')

        except Exception as e:
            logging.error(f"Error in WebSocket handler: {e}", exc_info=True)
        finally:
            self.active_connections.remove(websocket)
        
        return websocket

    async def close_all_connections(self):
        for ws in self.active_connections:
            await ws.close()

async def index(request):
    logging.info("Serving index.html")
    return web.FileResponse('./index.html')

async def get_ws_secret(request):
    return web.json_response({"secret": WS_SECRET_KEY})

async def chat(request):
    try:
        data = await request.json()
        messages = data.get('messages', [])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Using one of your available models
            messages=messages,
            max_tokens=150
        )
        
        return web.json_response({
            "text": response.choices[0].message.content.strip()
        })
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def main():
    port = int(os.environ.get('PORT', 8080))
    
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws_secret', get_ws_secret)
    app.router.add_post('/chat', chat)
    app.router.add_static('/', '.', show_index=True)
    
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    for route in list(app.router.routes()):
        cors.add(route)
    
    ws_handler = WebSocketHandler()
    app.router.add_get('/ws', ws_handler.handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logging.info(f"Server started on http://0.0.0.0:{port}")
    
    try:
        await asyncio.Future()
    finally:
        await ws_handler.close_all_connections()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())