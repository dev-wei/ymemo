"""AWS Transcribe Streaming provider implementation."""

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator, Optional, Dict
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from ...core.interfaces import TranscriptionProvider, AudioConfig, TranscriptionResult
from ...utils.exceptions import AWSTranscribeError, TranscriptionProviderError


logger = logging.getLogger(__name__)


class AWSTranscribeHandler(TranscriptResultStreamHandler):
    """Handler for AWS Transcribe streaming events."""
    
    def __init__(self, transcript_result_stream, result_queue: asyncio.Queue):
        super().__init__(transcript_result_stream)
        self.result_queue = result_queue
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Handle incoming transcript events."""
        results = transcript_event.transcript.results
        
        for result in results:
            if result.alternatives:
                alternative = result.alternatives[0]
                
                # Extract speaker information if available
                speaker_id = None
                if hasattr(result, 'channel_id') and result.channel_id:
                    speaker_id = f"Speaker-{result.channel_id}"
                elif hasattr(alternative, 'items') and alternative.items:
                    # Check if any items have speaker labels
                    for item in alternative.items:
                        if hasattr(item, 'speaker') and item.speaker:
                            speaker_id = f"Speaker-{item.speaker}"
                            break
                
                transcription_result = TranscriptionResult(
                    text=alternative.transcript,
                    speaker_id=speaker_id,
                    confidence=getattr(alternative, 'confidence', 0.0),
                    start_time=getattr(result, 'start_time', 0.0),
                    end_time=getattr(result, 'end_time', 0.0),
                    is_partial=result.is_partial
                )
                
                await self.result_queue.put(transcription_result)


class AWSTranscribeProvider(TranscriptionProvider):
    """AWS Transcribe Streaming transcription provider."""
    
    def __init__(self, region: str = 'us-east-1', language_code: str = 'en-US', profile_name: Optional[str] = None):
        self.region = region
        self.language_code = language_code
        self.profile_name = profile_name or os.getenv('AWS_PROFILE')
        self.client = None
        self.stream = None
        self.result_queue = asyncio.Queue()
        self._streaming_task = None
        
        # Track utterances for proper partial result handling
        self.active_utterances: Dict[str, int] = {}  # result_id -> sequence_number
        self.utterance_counter = 0
        
    async def start_stream(self, audio_config: AudioConfig) -> None:
        """Start the AWS Transcribe streaming session."""
        try:
            # Create boto3 session with profile if specified
            if self.profile_name:
                logger.info(f"🔑 Using AWS profile: {self.profile_name}")
                session = boto3.Session(profile_name=self.profile_name)
            else:
                logger.info("🔑 Using default AWS credentials")
                session = boto3.Session()
            
            logger.info(f"🚀 Initializing AWS Transcribe client (region: {self.region})")
            self.client = TranscribeStreamingClient(region=self.region)
            
            logger.info(f"🎯 Starting stream transcription (language: {self.language_code}, sample_rate: {audio_config.sample_rate})")
            
            # Start stream transcription
            self.stream = await self.client.start_stream_transcription(
                language_code=self.language_code,
                media_sample_rate_hz=audio_config.sample_rate,
                media_encoding='pcm'
            )
            
            logger.info("✅ AWS Transcribe stream connection established")
            
            # Start the handler task
            self._streaming_task = asyncio.create_task(
                self._handle_stream_events()
            )
            
            logger.info("🔄 AWS Transcribe event handler started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start AWS Transcribe stream: {e}")
            logger.error(f"❌ Error details: {str(e)}")
            raise AWSTranscribeError(f"Failed to start AWS Transcribe stream: {e}") from e
    
    async def _handle_stream_events(self):
        """Handle streaming events from AWS Transcribe."""
        try:
            logger.info("🎧 Starting to listen for AWS Transcribe events...")
            event_count = 0
            
            async for event in self.stream.output_stream:
                event_count += 1
                logger.debug(f"📥 Received event #{event_count} from AWS Transcribe: {type(event)}")
                
                # Check if this is a TranscriptEvent
                if hasattr(event, 'transcript') and event.transcript:
                    logger.debug(f"📝 Processing transcript event #{event_count}")
                    await self._process_transcript_event(event.transcript)
                else:
                    logger.debug(f"ℹ️  Non-transcript event received: {type(event)}")
                    
        except Exception as e:
            logger.error(f"❌ Error handling stream events: {e}")
            import traceback
            traceback.print_exc()
            raise AWSTranscribeError(f"Error handling stream events: {e}") from e
    
    async def _process_transcript_event(self, transcript):
        """Process a transcript event and extract results."""
        try:
            # Access results from the transcript object
            results = transcript.results if hasattr(transcript, 'results') else []
            
            # Only log if there are actual results to avoid spam
            if results:
                logger.info(f"🔍 Processing transcript with {len(results)} results")
            else:
                logger.debug(f"🔍 Processing transcript with 0 results (empty)")
            
            for i, result in enumerate(results):
                logger.debug(f"📋 Result #{i+1}: {result}")
                
                if hasattr(result, 'alternatives') and result.alternatives:
                    alternative = result.alternatives[0]
                    text = alternative.transcript if hasattr(alternative, 'transcript') else ''
                    is_partial = result.is_partial if hasattr(result, 'is_partial') else False
                    confidence = getattr(alternative, 'confidence', 0.0)
                    
                    if text.strip():  # Only process non-empty text
                        # Generate result ID for tracking related partial results
                        result_id = getattr(result, 'result_id', str(uuid.uuid4()))
                        
                        # Create utterance ID and sequence number
                        if result_id not in self.active_utterances:
                            self.utterance_counter += 1
                            self.active_utterances[result_id] = 0
                            utterance_id = f"utterance_{self.utterance_counter}"
                        else:
                            utterance_id = f"utterance_{self.utterance_counter}"
                        
                        # Increment sequence number for this result
                        self.active_utterances[result_id] += 1
                        sequence_number = self.active_utterances[result_id]
                        
                        # Clean up completed utterances
                        if not is_partial:
                            del self.active_utterances[result_id]
                        
                        logger.info(f"💬 AWS Transcribe returned text: '{text}' (partial: {is_partial}, confidence: {confidence:.2f}, utterance: {utterance_id}, seq: {sequence_number})")
                        
                        transcription_result = TranscriptionResult(
                            text=text,
                            speaker_id=None,  # Basic implementation without speaker ID
                            confidence=confidence,
                            start_time=getattr(result, 'start_time', 0.0),
                            end_time=getattr(result, 'end_time', 0.0),
                            is_partial=is_partial,
                            result_id=result_id,
                            utterance_id=utterance_id,
                            sequence_number=sequence_number
                        )
                        
                        await self.result_queue.put(transcription_result)
                        logger.debug(f"✅ Added transcription result to queue: '{text}' (utterance: {utterance_id})")
                    else:
                        logger.debug(f"⚠️  Skipping empty text result")
                else:
                    logger.debug(f"⚠️  No alternatives in result #{i+1}")
                        
        except Exception as e:
            logger.error(f"❌ Error processing transcript event: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise here to keep the stream going
    
    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio data to AWS Transcribe."""
        if self.stream and self.stream.input_stream:
            try:
                await self.stream.input_stream.send_audio_event(audio_chunk=audio_chunk)
                logger.debug(f"📡 Sent audio chunk to AWS Transcribe: {len(audio_chunk)} bytes")
            except Exception as e:
                logger.error(f"❌ Failed to send audio to AWS Transcribe: {e}")
                logger.error(f"❌ Send error details: {str(e)}")
                raise AWSTranscribeError(f"Failed to send audio to AWS Transcribe: {e}") from e
        else:
            logger.warning(f"⚠️  Cannot send audio - stream not available (stream: {self.stream is not None}, input_stream: {self.stream.input_stream is not None if self.stream else False})")
    
    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Get transcription results as they become available."""
        while True:
            try:
                # Wait for results with timeout to allow for graceful shutdown
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                yield result
            except asyncio.TimeoutError:
                # Continue polling for results
                continue
            except asyncio.CancelledError:
                logger.info("🛑 AWS Transcribe: Transcription generator cancelled")
                break
            except Exception as e:
                logger.error(f"❌ AWS Transcribe: Error getting transcription result: {e}")
                break
    
    async def stop_stream(self) -> None:
        """Stop the transcription stream and cleanup resources."""
        logger.info("🛑 AWS Transcribe: Stopping stream...")
        
        try:
            # Step 1: Stop the input stream
            if self.stream and self.stream.input_stream:
                try:
                    logger.info("🛑 AWS Transcribe: Ending input stream...")
                    await asyncio.wait_for(self.stream.input_stream.end_stream(), timeout=1.0)
                    logger.info("✅ AWS Transcribe: Input stream ended")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ AWS Transcribe: Input stream end timed out")
                except Exception as e:
                    logger.warning(f"⚠️ AWS Transcribe: Error ending input stream: {e}")
            
            # Step 2: Cancel the streaming task
            if self._streaming_task and not self._streaming_task.done():
                try:
                    logger.info("🛑 AWS Transcribe: Cancelling streaming task...")
                    self._streaming_task.cancel()
                    await asyncio.wait_for(self._streaming_task, timeout=0.5)
                    logger.info("✅ AWS Transcribe: Streaming task cancelled")
                except asyncio.CancelledError:
                    logger.info("✅ AWS Transcribe: Streaming task cancelled")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ AWS Transcribe: Streaming task cancellation timed out")
                except Exception as e:
                    logger.warning(f"⚠️ AWS Transcribe: Error cancelling streaming task: {e}")
            
            logger.info("✅ AWS Transcribe: Stream stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ AWS Transcribe: Error stopping stream: {e}")
            # Don't re-raise - we want cleanup to always complete
        finally:
            # Always clear references
            self.stream = None
            self.client = None
            self.handler = None
            self._streaming_task = None
            logger.info("🛑 AWS Transcribe: Cleanup completed")