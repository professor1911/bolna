import asyncio
import traceback
import os
import json
import time
import websockets
from urllib.parse import urlencode
from dotenv import load_dotenv
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, InvalidHandshake

from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class AssemblyAITranscriber(BaseTranscriber):
    def __init__(self, telephony_provider, input_queue=None, model='best', stream=True, language="en", 
                 sampling_rate="16000", encoding="pcm_s16le", output_queue=None, keywords=None,
                 process_interim_results="true", **kwargs):
        super().__init__(input_queue)
        self.language = language
        self.stream = stream
        self.provider = telephony_provider
        self.model = model
        self.sampling_rate = int(sampling_rate)
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('ASSEMBLYAI_API_KEY'))
        self.assemblyai_host = os.getenv('ASSEMBLYAI_HOST', 'api.assemblyai.com')
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.keywords = keywords
        self.audio_cursor = 0.0
        self.transcription_cursor = 0.0
        self.interruption_signalled = False
        
        self.audio_submitted = False
        self.audio_submission_time = None
        self.num_frames = 0
        self.connection_start_time = None
        self.process_interim_results = process_interim_results
        self.audio_frame_duration = 0.0
        self.connected_via_dashboard = kwargs.get("enforce_streaming", True)
        
        # Message states
        self.curr_message = ''
        self.finalized_transcript = ""
        self.final_transcript = ""
        self.is_transcript_sent_for_processing = False
        self.websocket_connection = None
        self.connection_authenticated = False
        
        # AssemblyAI specific
        self.heartbeat_task = None
        self.sender_task = None
        self.session_token = None

    def get_assemblyai_ws_url(self):
        """Generate AssemblyAI WebSocket URL with parameters"""
        params = {
            'sample_rate': str(self.sampling_rate),
            'word_boost': json.dumps(self.keywords.split(",")) if self.keywords else '[]',
            'encoding': 'pcm_s16le'  # AssemblyAI expects this format for streaming
        }
        
        # Adjust encoding and sample rate based on provider
        if self.provider in ('twilio', 'exotel', 'plivo'):
            self.encoding = 'pcm_mulaw' if self.provider == "twilio" else "pcm_s16le"
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.2  # With twilio we are sending 200ms at a time
            params['sample_rate'] = str(self.sampling_rate)
            params['encoding'] = self.encoding
        elif self.provider == "web_based_call":
            self.sampling_rate = 16000
            self.audio_frame_duration = 0.256
            params['sample_rate'] = str(self.sampling_rate)
            params['encoding'] = 'pcm_s16le'
        elif not self.connected_via_dashboard:
            params['sample_rate'] = '16000'
            params['encoding'] = 'pcm_s16le'
        
        if self.provider == "playground":
            self.sampling_rate = 8000
            self.audio_frame_duration = 0.0

        # Build WebSocket URL
        websocket_url = f"wss://{self.assemblyai_host}/v2/realtime/ws?{urlencode(params)}"
        return websocket_url

    async def send_heartbeat(self, ws: ClientConnection):
        """Send periodic heartbeat messages to keep connection alive"""
        try:
            while True:
                # AssemblyAI uses a simple ping for keepalive
                try:
                    await ws.ping()
                except ConnectionClosedError as e:
                    logger.info(f"Connection closed while sending heartbeat: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break
                    
                await asyncio.sleep(30)  # Send ping every 30 seconds
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error('Error in send_heartbeat: ' + str(e))
            raise

    async def toggle_connection(self):
        """Close WebSocket connection and cleanup tasks"""
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        if self.sender_task is not None:
            self.sender_task.cancel()
        
        if self.websocket_connection is not None:
            try:
                await self.websocket_connection.close()
                logger.info("WebSocket connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing websocket connection: {e}")
            finally:
                self.websocket_connection = None
                self.connection_authenticated = False

    async def _check_and_process_end_of_stream(self, ws_data_packet, ws):
        """Check for end-of-stream marker and close connection if found"""
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            # Send terminate message to AssemblyAI
            terminate_msg = {"terminate_session": True}
            await ws.send(json.dumps(terminate_msg))
            return True  # Indicates end of processing
        return False

    def get_meta_info(self):
        return self.meta_info

    async def sender_stream(self, ws: ClientConnection):
        """Send audio data to AssemblyAI WebSocket"""
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                
                # Initialize new request
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info')
                    self.audio_submitted = True
                    self.audio_submission_time = time.time()
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id

                end_of_stream = await self._check_and_process_end_of_stream(ws_data_packet, ws)
                if end_of_stream:
                    break
                    
                self.num_frames += 1
                # Save the audio cursor here
                self.audio_cursor = self.num_frames * self.audio_frame_duration
                
                # Send audio data - AssemblyAI expects base64 encoded audio
                audio_data = ws_data_packet.get('data')
                if audio_data:
                    import base64
                    encoded_audio = base64.b64encode(audio_data).decode('utf-8')
                    message = {"audio_data": encoded_audio}
                    
                    try:
                        await ws.send(json.dumps(message))
                    except ConnectionClosedError as e:
                        logger.error(f"Connection closed while sending data: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Error sending data to websocket: {e}")
                        break
                        
        except asyncio.CancelledError:
            logger.info("Sender stream task cancelled")
            raise
        except Exception as e:
            logger.error('Error in sender_stream: ' + str(e))
            raise

    async def receiver(self, ws: ClientConnection):
        """Receive and process messages from AssemblyAI WebSocket"""
        async for msg in ws:
            try:
                data = json.loads(msg)

                # Set connection start time if not set
                if self.connection_start_time is None:
                    self.connection_start_time = (time.time() - (self.num_frames * self.audio_frame_duration))

                message_type = data.get("message_type", "")

                if message_type == "SessionBegins":
                    logger.info("AssemblyAI session started")
                    self.session_token = data.get("session_id")
                    continue

                elif message_type == "PartialTranscript":
                    transcript = data.get("text", "").strip()
                    if transcript:
                        logger.debug(f"Received partial transcript: {transcript}")
                        packet_data = {
                            "type": "interim_transcript_received",
                            "content": transcript
                        }
                        yield create_ws_data_packet(packet_data, self.meta_info)

                elif message_type == "FinalTranscript":
                    transcript = data.get("text", "").strip()
                    if transcript:
                        logger.info(f"Received final transcript: {transcript}")
                        self.final_transcript += f' {transcript}'
                        
                        if not self.is_transcript_sent_for_processing:
                            packet_data = {
                                "type": "transcript",
                                "content": self.final_transcript.strip()
                            }
                            self.is_transcript_sent_for_processing = True
                            self.final_transcript = ""
                            yield create_ws_data_packet(packet_data, self.meta_info)

                elif message_type == "SessionTerminated":
                    logger.info("AssemblyAI session terminated")
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

                elif "error" in data:
                    logger.error(f"AssemblyAI error: {data['error']}")
                    continue

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing AssemblyAI message: {e}")
                self.interruption_signalled = False

    async def push_to_transcriber_queue(self, data_packet):
        """Push transcription results to output queue"""
        await self.transcriber_output_queue.put(data_packet)

    async def assemblyai_connect(self):
        """Establish WebSocket connection to AssemblyAI with proper error handling"""
        try:
            websocket_url = self.get_assemblyai_ws_url()
            additional_headers = {
                'Authorization': self.api_key
            }
            
            logger.info(f"Attempting to connect to AssemblyAI websocket: {websocket_url}")
            
            assemblyai_ws = await asyncio.wait_for(
                websockets.connect(websocket_url, additional_headers=additional_headers),
                timeout=10.0  # 10 second timeout
            )
            
            self.websocket_connection = assemblyai_ws
            self.connection_authenticated = True
            logger.info("Successfully connected to AssemblyAI websocket")
            
            return assemblyai_ws
            
        except asyncio.TimeoutError:
            logger.error("Timeout while connecting to AssemblyAI websocket")
            raise ConnectionError("Timeout while connecting to AssemblyAI websocket")
        except InvalidHandshake as e:
            logger.error(f"Invalid handshake during AssemblyAI websocket connection: {e}")
            raise ConnectionError(f"Invalid handshake during AssemblyAI websocket connection: {e}")
        except ConnectionClosedError as e:
            logger.error(f"AssemblyAI websocket connection closed unexpectedly: {e}")
            raise ConnectionError(f"AssemblyAI websocket connection closed unexpectedly: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to AssemblyAI websocket: {e}")
            raise ConnectionError(f"Unexpected error connecting to AssemblyAI websocket: {e}")

    async def run(self):
        """Start the transcription task"""
        try:
            self.transcription_task = asyncio.create_task(self.transcribe())
        except Exception as e:
            logger.error(f"Error starting transcription task: {e}")

    async def transcribe(self):
        """Main transcription method"""
        assemblyai_ws = None
        try:
            start_time = time.perf_counter()
            
            try:
                assemblyai_ws = await self.assemblyai_connect()
            except (ValueError, ConnectionError) as e:
                logger.error(f"Failed to establish AssemblyAI connection: {e}")
                await self.toggle_connection()
                return
            
            if not self.connection_time:
                self.connection_time = round((time.perf_counter() - start_time) * 1000)

            if self.stream:
                self.sender_task = asyncio.create_task(self.sender_stream(assemblyai_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(assemblyai_ws))
                
                try:
                    async for message in self.receiver(assemblyai_ws):
                        if self.connection_on:
                            await self.push_to_transcriber_queue(message)
                        else:
                            logger.info("Closing the AssemblyAI connection")
                            terminate_msg = {"terminate_session": True}
                            await assemblyai_ws.send(json.dumps(terminate_msg))
                            break
                except ConnectionClosedError as e:
                    logger.error(f"AssemblyAI websocket connection closed during streaming: {e}")
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise

        except (ValueError, ConnectionError) as e:
            logger.error(f"Connection error in transcribe: {e}")
            await self.toggle_connection()
        except Exception as e:
            logger.error(f"Unexpected error in transcribe: {e}")
            await self.toggle_connection()
        finally:
            if assemblyai_ws is not None:
                try:
                    await assemblyai_ws.close()
                    logger.info("AssemblyAI websocket closed in finally block")
                except Exception as e:
                    logger.error(f"Error closing websocket in finally block: {e}")
                finally:
                    self.websocket_connection = None
                    self.connection_authenticated = False
            
            if hasattr(self, 'sender_task') and self.sender_task is not None:
                self.sender_task.cancel()
            if hasattr(self, 'heartbeat_task') and self.heartbeat_task is not None:
                self.heartbeat_task.cancel()
            
            await self.push_to_transcriber_queue(
                create_ws_data_packet("transcriber_connection_closed", getattr(self, 'meta_info', {}))
            )
