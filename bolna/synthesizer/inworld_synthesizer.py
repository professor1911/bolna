import asyncio
import json
import os
import traceback
import uuid
import websockets
import aiohttp
import base64
from collections import deque

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet

logger = configure_logger(__name__)


class InWorldSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id, model="inworld-tts", audio_format="wav", sampling_rate="16000",
                 stream=True, buffer_size=400, temperature=0.5, synthesizer_key=None,
                 caching=True, **kwargs):
        super().__init__(kwargs.get("task_manager_instance", None), stream)
        self.api_key = os.environ.get("INWORLD_API_KEY") if synthesizer_key is None else synthesizer_key
        self.voice = voice
        self.voice_id = voice_id
        self.model = model
        self.stream = stream
        self.sampling_rate = int(sampling_rate)
        self.audio_format = audio_format
        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.temperature = temperature
        self.caching = caching
        
        # InWorld API endpoints
        self.inworld_host = os.getenv("INWORLD_API_HOST", "api.inworld.ai")
        self.api_url = f"https://{self.inworld_host}/v1/text-to-speech"
        self.ws_url = f"wss://{self.inworld_host}/v1/text-to-speech/stream"
        
        if self.caching:
            self.cache = InmemoryScalarCache()
            
        self.synthesized_characters = 0
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.websocket_holder = {"websocket": None}
        self.conversation_ended = False
        self.current_text = ""
        self.context_id = None

    def get_format(self, format, sampling_rate):
        """Return the appropriate audio format for InWorld API"""
        if self.use_mulaw:
            return "ulaw_8000"
        return f"wav_{sampling_rate}"

    def get_engine(self):
        return self.model

    async def handle_interruption(self):
        """Handle interruption by canceling current synthesis"""
        try:
            if self.context_id and self.websocket_holder["websocket"]:
                interrupt_message = {
                    "type": "interrupt",
                    "context_id": self.context_id
                }
                await self.websocket_holder["websocket"].send(json.dumps(interrupt_message))
                self.context_id = str(uuid.uuid4())
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")

    async def establish_websocket_connection(self):
        """Establish WebSocket connection to InWorld TTS API"""
        if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                self.websocket_holder["websocket"] = await websockets.connect(
                    self.ws_url,
                    extra_headers=headers
                )
                logger.info("InWorld WebSocket connection established")
                
                # Send initial configuration
                config_message = {
                    "type": "config",
                    "voice_id": self.voice_id,
                    "model": self.model,
                    "audio_format": self.get_format(self.audio_format, self.sampling_rate),
                    "sampling_rate": self.sampling_rate,
                    "temperature": self.temperature
                }
                await self.websocket_holder["websocket"].send(json.dumps(config_message))
                
            except Exception as e:
                logger.error(f"Failed to establish InWorld WebSocket connection: {e}")
                self.websocket_holder["websocket"] = None

    async def sender(self, text, sequence_id, end_of_llm_stream=False):
        """Send text to InWorld TTS for synthesis"""
        try:
            if self.conversation_ended:
                return

            if not self.should_synthesize_response(sequence_id):
                logger.info(
                    f"Not synthesizing text as the sequence_id ({sequence_id}) is not in the current sequence_ids.")
                await self.flush_synthesizer_stream()
                return

            # Ensure WebSocket connection is established
            if self.websocket_holder["websocket"] is None or self.websocket_holder["websocket"].closed:
                await self.establish_websocket_connection()

            # Wait for connection to be ready
            while (self.websocket_holder["websocket"] is None or 
                   self.websocket_holder["websocket"].closed):
                logger.info("Waiting for InWorld WebSocket connection to be established...")
                await asyncio.sleep(1)

            if text != "":
                self.current_text += text
                
                # Send text in chunks
                for text_chunk in self.text_chunker(text):
                    if not self.should_synthesize_response(sequence_id):
                        logger.info(
                            f"Not synthesizing text as the sequence_id ({sequence_id}) is not current (inner loop).")
                        await self.flush_synthesizer_stream()
                        return
                        
                    try:
                        message = {
                            "type": "synthesize",
                            "text": text_chunk.strip(),
                            "voice_id": self.voice_id,
                            "context_id": self.context_id or str(uuid.uuid4())
                        }
                        
                        if self.context_id is None:
                            self.context_id = message["context_id"]
                            
                        await self.websocket_holder["websocket"].send(json.dumps(message))
                        self.synthesized_characters += len(text_chunk)
                        
                    except Exception as e:
                        logger.error(f"Error sending text chunk to InWorld: {e}")
                        return

            # Mark end of stream if this is the final chunk
            if end_of_llm_stream:
                self.last_text_sent = True
                try:
                    end_message = {
                        "type": "end_stream",
                        "context_id": self.context_id
                    }
                    await self.websocket_holder["websocket"].send(json.dumps(end_message))
                except Exception as e:
                    logger.error(f"Error sending end-of-stream signal to InWorld: {e}")

        except asyncio.CancelledError:
            logger.info("InWorld sender task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in InWorld sender: {e}")

    async def receiver(self):
        """Receive audio data from InWorld TTS API"""
        while True:
            try:
                if self.conversation_ended:
                    return

                if (self.websocket_holder["websocket"] is None or 
                    self.websocket_holder["websocket"].closed):
                    logger.info("InWorld WebSocket is not connected, skipping receive.")
                    await asyncio.sleep(5)
                    continue

                response = await self.websocket_holder["websocket"].recv()
                data = json.loads(response)
                
                logger.debug(f"InWorld response: {data.get('type', 'unknown')}")

                if data.get("type") == "audio" and "audio_data" in data:
                    # Audio data is typically base64 encoded
                    audio_data = base64.b64decode(data["audio_data"])
                    text_spoken = data.get("text", "")
                    
                    if not self.first_chunk_generated:
                        self.first_chunk_generated = True
                        logger.info("First InWorld audio chunk generated")
                    
                    yield audio_data, text_spoken

                elif data.get("type") == "end_of_stream":
                    logger.info("InWorld end of stream received")
                    yield b'\x00', ""

                elif data.get("type") == "error":
                    logger.error(f"InWorld API error: {data.get('message', 'Unknown error')}")
                    yield b'\x00', ""

                elif self.last_text_sent and data.get("type") == "synthesis_complete":
                    logger.info("InWorld synthesis complete")
                    yield b'\x00', ""

            except websockets.exceptions.ConnectionClosed:
                logger.info("InWorld WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error in InWorld receiver: {e}")
                traceback.print_exc()

    async def __send_http_payload(self, payload):
        """Send HTTP request to InWorld TTS API for non-streaming synthesis"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data
                    else:
                        logger.error(f"InWorld HTTP Error: {response.status} - {await response.text()}")
                        return None
        except Exception as e:
            logger.error(f"Error in InWorld HTTP request: {e}")
            return None

    async def synthesize(self, text):
        """Synthesize text using HTTP API (non-streaming)"""
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "audio_format": self.get_format(self.audio_format, self.sampling_rate),
            "sampling_rate": self.sampling_rate,
            "temperature": self.temperature
        }
        
        if self.caching:
            cache_key = f"inworld_tts_{hash(json.dumps(payload, sort_keys=True))}"
            cached_audio = self.cache.get(cache_key)
            if cached_audio:
                logger.info("Returning cached InWorld audio")
                return cached_audio

        audio_data = await self.__send_http_payload(payload)
        
        if audio_data and self.caching:
            self.cache.set(cache_key, audio_data)
        
        return audio_data

    async def generate(self):
        """Generator for streaming synthesis"""
        if self.stream:
            async for audio_chunk, text_spoken in self.receiver():
                if audio_chunk == b'\x00':
                    break
                yield audio_chunk, text_spoken
        else:
            # For non-streaming, synthesize all queued text
            text_to_synthesize = ""
            while not self.text_queue.empty():
                text_to_synthesize += self.text_queue.popleft()
            
            if text_to_synthesize:
                audio_data = await self.synthesize(text_to_synthesize)
                if audio_data:
                    yield audio_data, text_to_synthesize

    def push(self, text):
        """Push text to the synthesis queue"""
        self.text_queue.append(text)

    def get_synthesized_characters(self):
        """Return the number of characters synthesized"""
        return self.synthesized_characters

    async def flush_synthesizer_stream(self):
        """Flush any remaining data in the stream"""
        try:
            if self.websocket_holder["websocket"] and not self.websocket_holder["websocket"].closed:
                flush_message = {
                    "type": "flush",
                    "context_id": self.context_id
                }
                await self.websocket_holder["websocket"].send(json.dumps(flush_message))
        except Exception as e:
            logger.error(f"Error flushing InWorld synthesizer stream: {e}")

    async def cleanup(self):
        """Clean up resources"""
        self.conversation_ended = True
        try:
            if self.websocket_holder["websocket"] and not self.websocket_holder["websocket"].closed:
                await self.websocket_holder["websocket"].close()
        except Exception as e:
            logger.error(f"Error during InWorld cleanup: {e}")

    def supports_websocket(self):
        """Return True if this synthesizer supports WebSocket streaming"""
        return True
