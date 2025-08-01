"""Dual AWS Transcribe provider using separate mono connections."""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core.interfaces import TranscriptionProvider, AudioConfig, TranscriptionResult
from ...utils.exceptions import AWSTranscribeError, TranscriptionProviderError
from ..channel_splitter import AudioChannelSplitter, SplitResult
from .aws_transcribe import AWSTranscribeProvider


logger = logging.getLogger(__name__)


class ChannelState(Enum):
    """State of individual transcription channel."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    FAILED = "failed"
    STOPPING = "stopping"


@dataclass
class ChannelStatus:
    """Status information for a transcription channel."""
    state: ChannelState = ChannelState.INACTIVE
    provider: Optional[AWSTranscribeProvider] = None
    last_result_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    results_received: int = 0
    bytes_sent: int = 0


class AWSTranscribeDualProvider(TranscriptionProvider):
    """
    Dual AWS Transcribe provider using separate mono connections.
    
    This provider splits stereo audio into separate left/right channels and
    processes each through its own AWS Transcribe connection, providing
    more reliable transcription than AWS's dual-channel feature.
    """
    
    def __init__(
        self, 
        region: str = 'us-east-1', 
        language_code: str = 'en-US', 
        profile_name: Optional[str] = None,
        audio_format: str = 'int16'
    ):
        """
        Initialize dual AWS Transcribe provider.
        
        Args:
            region: AWS region for Transcribe service
            language_code: Language code for transcription
            profile_name: AWS profile name for authentication
            audio_format: Audio format for channel splitting
        """
        # Store configuration
        self.region = region
        self.language_code = language_code
        self.profile_name = profile_name
        self.audio_format = audio_format
        
        # Initialize channel splitter
        self.channel_splitter = AudioChannelSplitter(audio_format=audio_format)
        
        # Channel management
        self.left_channel = ChannelStatus()
        self.right_channel = ChannelStatus()
        
        # Result queues
        self.result_queue: Optional[asyncio.Queue] = None
        
        # Provider state
        self.is_active = False
        self.start_time = 0.0
        self.total_chunks_processed = 0
        self.fallback_mode = False  # True if running in mono fallback
        self.fallback_channel: Optional[str] = None  # 'left' or 'right'
        
        # Connection monitoring
        self.connection_health_callback = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'left_results': 0,
            'right_results': 0,
            'merged_results': 0,
            'left_errors': 0,
            'right_errors': 0,
            'split_errors': 0,
            'fallback_activations': 0
        }
        
        logger.info(f"🏗️ AWSTranscribeDualProvider initialized: region={region}, language={language_code}")
    
    async def start_stream(self, audio_config: AudioConfig) -> None:
        """
        Start dual AWS Transcribe streams.
        
        Args:
            audio_config: Audio configuration (must be stereo)
        """
        try:
            logger.info(f"🚀 Dual AWS: Starting dual transcription streams")
            logger.info(f"🚀 Dual AWS: Audio config - {audio_config.channels} channels, {audio_config.sample_rate}Hz")
            
            # Validate stereo configuration
            if audio_config.channels != 2:
                raise ValueError(f"Dual provider requires stereo input (2 channels), got {audio_config.channels}")
            
            # Create result queue
            self.result_queue = asyncio.Queue()
            self.is_active = True
            self.start_time = time.time()
            
            # Create mono audio config for individual providers
            mono_config = AudioConfig(
                sample_rate=audio_config.sample_rate,
                channels=1,  # Each provider gets mono
                chunk_size=audio_config.chunk_size,
                format=audio_config.format
            )
            
            # Initialize channel providers
            await self._initialize_channel_providers(mono_config)
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._monitor_channel_health())
            
            logger.info("✅ Dual AWS: Both transcription channels started successfully")
            
        except Exception as e:
            logger.error(f"❌ Dual AWS: Failed to start streams: {e}")
            await self._cleanup_channels()
            raise AWSTranscribeError(f"Failed to start dual AWS streams: {e}") from e
    
    async def _initialize_channel_providers(self, mono_config: AudioConfig) -> None:
        """Initialize both channel providers."""
        try:
            # Initialize left channel provider
            logger.info("🚀 Dual AWS: Initializing left channel (Speaker A)")
            self.left_channel.provider = AWSTranscribeProvider(
                region=self.region,
                language_code=self.language_code,
                profile_name=self.profile_name
            )
            self.left_channel.state = ChannelState.STARTING
            await self.left_channel.provider.start_stream(mono_config)
            self.left_channel.state = ChannelState.ACTIVE
            logger.info("✅ Dual AWS: Left channel (Speaker A) started")
            
            # Initialize right channel provider
            logger.info("🚀 Dual AWS: Initializing right channel (Speaker B)")
            self.right_channel.provider = AWSTranscribeProvider(
                region=self.region,
                language_code=self.language_code,
                profile_name=self.profile_name
            )
            self.right_channel.state = ChannelState.STARTING
            await self.right_channel.provider.start_stream(mono_config)
            self.right_channel.state = ChannelState.ACTIVE
            logger.info("✅ Dual AWS: Right channel (Speaker B) started")
            
        except Exception as e:
            # If one channel fails during initialization, clean up and raise
            logger.error(f"❌ Dual AWS: Channel initialization failed: {e}")
            await self._cleanup_channels()
            raise
    
    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send stereo audio chunk to both channels.
        
        Args:
            audio_chunk: Stereo audio data
        """
        if not self.is_active or not self.result_queue:
            logger.warning("⚠️ Dual AWS: Cannot send audio - provider not active")
            return
        
        try:
            self.total_chunks_processed += 1
            
            # Split stereo audio into left/right channels
            split_result = self.channel_splitter.split_stereo_chunk(audio_chunk)
            
            if not split_result.split_successful:
                logger.error(f"❌ Dual AWS: Channel splitting failed: {split_result.error_message}")
                self.stats['split_errors'] += 1
                return
            
            # Send to channels based on current mode
            if self.fallback_mode:
                await self._send_audio_fallback_mode(split_result)
            else:
                await self._send_audio_dual_mode(split_result)
            
            # Log detailed analysis for first few chunks
            if self.total_chunks_processed <= 10:
                logger.info(f"🎵 Dual AWS: Audio chunk #{self.total_chunks_processed}")
                logger.info(f"   📊 Original: {len(audio_chunk)} bytes")
                logger.info(f"   🎚️  Left: {split_result.left_metrics.activity_level} - {len(split_result.left_channel)} bytes")
                logger.info(f"   🎚️  Right: {split_result.right_metrics.activity_level} - {len(split_result.right_channel)} bytes")
                
                # Warn about channel issues
                if split_result.left_metrics.is_silent and split_result.right_metrics.is_silent:
                    logger.warning("⚠️ Both channels silent - no audio to process")
                elif split_result.left_metrics.is_silent:
                    logger.warning("⚠️ Left channel (Speaker A) silent - only processing right channel")
                elif split_result.right_metrics.is_silent:
                    logger.warning("⚠️ Right channel (Speaker B) silent - only processing left channel")
            
        except Exception as e:
            logger.error(f"❌ Dual AWS: Error sending audio: {e}")
            # Don't raise - continue processing other chunks
    
    async def _send_audio_dual_mode(self, split_result: SplitResult) -> None:
        """Send audio in normal dual mode."""
        send_tasks = []
        
        # Send to left channel if active
        if (self.left_channel.state == ChannelState.ACTIVE and 
            self.left_channel.provider and 
            not split_result.left_metrics.is_silent):
            
            task = asyncio.create_task(
                self._send_to_channel(
                    self.left_channel, 
                    split_result.left_channel, 
                    "Left"
                )
            )
            send_tasks.append(task)
        
        # Send to right channel if active
        if (self.right_channel.state == ChannelState.ACTIVE and 
            self.right_channel.provider and 
            not split_result.right_metrics.is_silent):
            
            task = asyncio.create_task(
                self._send_to_channel(
                    self.right_channel, 
                    split_result.right_channel, 
                    "Right"
                )
            )
            send_tasks.append(task)
        
        # Wait for all sends to complete
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
    
    async def _send_audio_fallback_mode(self, split_result: SplitResult) -> None:
        """Send audio in fallback mode (only one channel)."""
        if self.fallback_channel == "left" and self.left_channel.state == ChannelState.ACTIVE:
            if not split_result.left_metrics.is_silent:
                await self._send_to_channel(self.left_channel, split_result.left_channel, "Left")
        elif self.fallback_channel == "right" and self.right_channel.state == ChannelState.ACTIVE:
            if not split_result.right_metrics.is_silent:
                await self._send_to_channel(self.right_channel, split_result.right_channel, "Right")
    
    async def _send_to_channel(self, channel: ChannelStatus, audio_data: bytes, channel_name: str) -> None:
        """Send audio data to a specific channel."""
        try:
            if channel.provider:
                await channel.provider.send_audio(audio_data)
                channel.bytes_sent += len(audio_data)
                
                # Reset error count on successful send
                if channel.error_count > 0:
                    logger.info(f"✅ {channel_name} channel: Connection recovered")
                    channel.error_count = 0
                    channel.last_error = None
                    
        except Exception as e:
            channel.error_count += 1
            channel.last_error = str(e)
            
            if channel_name.lower() == "left":
                self.stats['left_errors'] += 1
            else:
                self.stats['right_errors'] += 1
            
            # Log error but don't fail the entire operation
            if channel.error_count <= 5:  # Avoid log spam
                logger.error(f"❌ {channel_name} channel: Send error #{channel.error_count}: {e}")
            
            # Consider channel failed after multiple errors
            if channel.error_count >= 10:
                await self._handle_channel_failure(channel, channel_name)
    
    async def _handle_channel_failure(self, failed_channel: ChannelStatus, channel_name: str) -> None:
        """Handle channel failure and potentially activate fallback mode."""
        logger.warning(f"⚠️ {channel_name} channel: Failed after {failed_channel.error_count} errors")
        failed_channel.state = ChannelState.FAILED
        
        # Determine fallback strategy
        left_active = self.left_channel.state == ChannelState.ACTIVE
        right_active = self.right_channel.state == ChannelState.ACTIVE
        
        if not self.fallback_mode:
            if left_active and not right_active:
                logger.warning("🔄 Dual AWS: Activating fallback mode - using only left channel (Speaker A)")
                self.fallback_mode = True
                self.fallback_channel = "left"
                self.stats['fallback_activations'] += 1
            elif right_active and not left_active:
                logger.warning("🔄 Dual AWS: Activating fallback mode - using only right channel (Speaker B)")
                self.fallback_mode = True
                self.fallback_channel = "right"
                self.stats['fallback_activations'] += 1
            elif not left_active and not right_active:
                logger.error("❌ Dual AWS: Both channels failed - transcription unavailable")
                self.is_active = False
                
                if self.connection_health_callback:
                    self.connection_health_callback(False, "Both transcription channels failed")
    
    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Get transcription results from both channels.
        
        Yields:
            TranscriptionResult with appropriate speaker labeling
        """
        if not self.result_queue:
            logger.error("❌ Dual AWS: No result queue available")
            return
        
        # Start result collection tasks for both channels
        collection_tasks = []
        
        if self.left_channel.provider and self.left_channel.state == ChannelState.ACTIVE:
            task = asyncio.create_task(
                self._collect_channel_results(self.left_channel.provider, "Speaker A", "left")
            )
            collection_tasks.append(task)
        
        if self.right_channel.provider and self.right_channel.state == ChannelState.ACTIVE:
            task = asyncio.create_task(
                self._collect_channel_results(self.right_channel.provider, "Speaker B", "right")
            )
            collection_tasks.append(task)
        
        # Start result collection
        if collection_tasks:
            logger.info(f"🔊 Dual AWS: Starting result collection from {len(collection_tasks)} channels")
        
        try:
            # Yield results as they come from the unified queue
            while self.is_active or not self.result_queue.empty():
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                    self.stats['merged_results'] += 1
                    yield result
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info("🛑 Dual AWS: Result collection cancelled")
        finally:
            # Cancel collection tasks
            for task in collection_tasks:
                if not task.done():
                    task.cancel()
            
            logger.info("🛑 Dual AWS: Result collection stopped")
    
    async def _collect_channel_results(
        self, 
        provider: AWSTranscribeProvider, 
        speaker_label: str, 
        channel_name: str
    ) -> None:
        """Collect results from a single channel provider."""
        try:
            logger.info(f"🔊 {channel_name.title()} channel: Starting result collection for {speaker_label}")
            
            async for result in provider.get_transcription():
                # Update result with proper speaker labeling
                enhanced_result = TranscriptionResult(
                    text=result.text,
                    speaker_id=speaker_label,  # Override with Speaker A/B
                    confidence=result.confidence,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    is_partial=result.is_partial,
                    result_id=result.result_id,
                    utterance_id=result.utterance_id,
                    sequence_number=result.sequence_number
                )
                
                # Add to unified result queue
                if self.result_queue:
                    await self.result_queue.put(enhanced_result)
                    
                    # Update channel statistics
                    if channel_name == "left":
                        self.left_channel.results_received += 1
                        self.left_channel.last_result_time = time.time()
                        self.stats['left_results'] += 1
                    else:
                        self.right_channel.results_received += 1
                        self.right_channel.last_result_time = time.time()
                        self.stats['right_results'] += 1
                    
                    logger.debug(f"📝 {speaker_label}: '{result.text}' (confidence: {result.confidence:.2f})")
                    
        except asyncio.CancelledError:
            logger.info(f"🛑 {channel_name.title()} channel: Result collection cancelled")
        except Exception as e:
            logger.error(f"❌ {channel_name.title()} channel: Result collection error: {e}")
            # Channel failure will be handled by health monitor
    
    async def _monitor_channel_health(self) -> None:
        """Monitor health of both channels."""
        try:
            logger.info("🔍 Dual AWS: Starting channel health monitor")
            
            while self.is_active:
                current_time = time.time()
                
                # Check left channel health
                await self._check_channel_health(self.left_channel, "Left", current_time)
                
                # Check right channel health  
                await self._check_channel_health(self.right_channel, "Right", current_time)
                
                # Log periodic statistics
                if int(current_time - self.start_time) % 30 == 0:  # Every 30 seconds
                    self._log_channel_statistics()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
        except asyncio.CancelledError:
            logger.info("🛑 Dual AWS: Health monitor cancelled")
        except Exception as e:
            logger.error(f"❌ Dual AWS: Health monitor error: {e}")
    
    async def _check_channel_health(self, channel: ChannelStatus, channel_name: str, current_time: float) -> None:
        """Check health of individual channel."""
        if channel.state != ChannelState.ACTIVE or not channel.provider:
            return
        
        # Check for result timeout (no results for extended period)
        time_since_last_result = current_time - channel.last_result_time
        if channel.last_result_time > 0 and time_since_last_result > 60.0:  # 60 seconds
            logger.warning(f"⚠️ {channel_name} channel: No results for {time_since_last_result:.0f}s")
            
            if self.connection_health_callback:
                self.connection_health_callback(
                    False, 
                    f"{channel_name} channel timeout: no results for {time_since_last_result:.0f}s"
                )
    
    def _log_channel_statistics(self) -> None:
        """Log current channel statistics."""
        runtime = time.time() - self.start_time
        
        logger.info(f"📊 Dual AWS Statistics (runtime: {runtime:.1f}s):")
        logger.info(f"   🎚️  Left Channel: {self.left_channel.results_received} results, "
                   f"{self.left_channel.bytes_sent:,} bytes sent, {self.left_channel.error_count} errors")
        logger.info(f"   🎚️  Right Channel: {self.right_channel.results_received} results, "
                   f"{self.right_channel.bytes_sent:,} bytes sent, {self.right_channel.error_count} errors")
        logger.info(f"   📈 Total Results: {self.stats['merged_results']} merged from {self.total_chunks_processed} chunks")
        logger.info(f"   🔄 Fallback Mode: {self.fallback_mode} ({'active on ' + self.fallback_channel if self.fallback_mode else 'disabled'})")
        
        # Channel splitter statistics
        splitter_stats = self.channel_splitter.get_statistics()
        logger.info(f"   🔀 Channel Splitter: {splitter_stats['left_silence_rate']:.1f}% left silence, "
                   f"{splitter_stats['right_silence_rate']:.1f}% right silence")
    
    async def stop_stream(self) -> None:
        """Stop both transcription streams."""
        logger.info("🛑 Dual AWS: Stopping dual transcription streams")
        
        try:
            self.is_active = False
            
            # Cancel health monitoring
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop both channels
            await self._cleanup_channels()
            
            # Log final statistics
            self._log_final_statistics()
            
            logger.info("✅ Dual AWS: All streams stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Dual AWS: Error stopping streams: {e}")
        finally:
            # Clean up references
            self.result_queue = None
            self.health_check_task = None
    
    async def _cleanup_channels(self) -> None:
        """Clean up both channel providers."""
        cleanup_tasks = []
        
        # Clean up left channel
        if self.left_channel.provider:
            self.left_channel.state = ChannelState.STOPPING
            task = asyncio.create_task(self._cleanup_single_channel(self.left_channel, "Left"))
            cleanup_tasks.append(task)
        
        # Clean up right channel
        if self.right_channel.provider:
            self.right_channel.state = ChannelState.STOPPING
            task = asyncio.create_task(self._cleanup_single_channel(self.right_channel, "Right"))
            cleanup_tasks.append(task)
        
        # Wait for all cleanups to complete
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    async def _cleanup_single_channel(self, channel: ChannelStatus, channel_name: str) -> None:
        """Clean up a single channel provider."""
        try:
            if channel.provider:
                logger.info(f"🛑 {channel_name} channel: Stopping provider")
                await channel.provider.stop_stream()
                logger.info(f"✅ {channel_name} channel: Provider stopped")
        except Exception as e:
            logger.error(f"❌ {channel_name} channel: Cleanup error: {e}")
        finally:
            channel.provider = None
            channel.state = ChannelState.INACTIVE
    
    def _log_final_statistics(self) -> None:
        """Log final statistics."""
        runtime = time.time() - self.start_time
        
        logger.info(f"📊 Final Dual AWS Statistics:")
        logger.info(f"   ⏱️  Total Runtime: {runtime:.1f}s")
        logger.info(f"   📦 Audio Chunks: {self.total_chunks_processed}")
        logger.info(f"   📝 Results: Left={self.stats['left_results']}, Right={self.stats['right_results']}, Merged={self.stats['merged_results']}")
        logger.info(f"   ❌ Errors: Left={self.stats['left_errors']}, Right={self.stats['right_errors']}, Split={self.stats['split_errors']}")
        logger.info(f"   🔄 Fallback Activations: {self.stats['fallback_activations']}")
        
        # Channel splitter final stats
        splitter_stats = self.channel_splitter.get_statistics()
        logger.info(f"   🔀 Final Channel Analysis: "
                   f"Left silence: {splitter_stats['left_silence_rate']:.1f}%, "
                   f"Right silence: {splitter_stats['right_silence_rate']:.1f}%")
    
    def set_connection_health_callback(self, callback) -> None:
        """Set callback for connection health notifications."""
        self.connection_health_callback = callback
    
    def get_required_channels(self) -> int:
        """Get required number of channels (always 2 for dual provider)."""
        return 2