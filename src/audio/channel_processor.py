"""Audio channel processing utilities for real-time channel conversion.

Provides efficient algorithms for converting multi-channel audio to mono or other channel
configurations with minimal latency for real-time transcription applications.
"""

import logging
import struct
from typing import Union, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class MixingStrategy(Enum):
    """Audio channel mixing strategies."""
    AVERAGE = "average"        # Average all channels
    LEFT_ONLY = "left"        # Use left channel only  
    RIGHT_ONLY = "right"      # Use right channel only
    WEIGHTED = "weighted"     # Weighted average (center channels get more weight)


class AudioChannelProcessor:
    """High-performance audio channel processor for real-time conversion.
    
    Optimized for minimal latency (~0.1ms) channel conversion to support
    transcription providers with different channel requirements.
    """
    
    def __init__(self, mixing_strategy: MixingStrategy = MixingStrategy.AVERAGE):
        """Initialize the channel processor.
        
        Args:
            mixing_strategy: Strategy for mixing multiple channels
        """
        self.mixing_strategy = mixing_strategy
        self._format_processors = {
            'int16': self._process_int16,
            'int24': self._process_int24,
            'int32': self._process_int32,
            'float32': self._process_float32
        }
        
        # Conversion tracking for debugging
        self._conversion_count = 0
        self._debug_sample_logging = True  # Enable for detailed debugging
        
        logger.info(f"🔄 AudioChannelProcessor: Initialized with {mixing_strategy.value} mixing strategy")
    
    def _analyze_conversion_quality(self, input_data: bytes, output_data: bytes, 
                                   source_channels: int, target_channels: int,
                                   audio_format: str) -> dict:
        """Analyze the quality of channel conversion for debugging.
        
        Args:
            input_data: Original audio data
            output_data: Converted audio data  
            source_channels: Number of input channels
            target_channels: Number of output channels
            audio_format: Audio format string
            
        Returns:
            Dict with conversion analysis
        """
        try:
            # Calculate sample information
            bytes_per_sample = 2 if audio_format == 'int16' else 4  # int32/float32 use 4 bytes
            input_sample_count = len(input_data) // (bytes_per_sample * source_channels)
            output_sample_count = len(output_data) // (bytes_per_sample * target_channels)
            
            # Unpack samples for analysis
            format_char = 'h' if audio_format == 'int16' else ('i' if audio_format == 'int32' else 'f')
            
            input_samples_total = len(input_data) // bytes_per_sample
            output_samples_total = len(output_data) // bytes_per_sample
            
            if input_samples_total > 0:
                input_samples = struct.unpack(f'<{input_samples_total}{format_char}', input_data)
            else:
                input_samples = []
                
            if output_samples_total > 0:
                output_samples = struct.unpack(f'<{output_samples_total}{format_char}', output_data)
            else:
                output_samples = []
            
            # Calculate amplitude statistics
            input_max = max(abs(s) for s in input_samples) if input_samples else 0
            input_avg = sum(abs(s) for s in input_samples) / len(input_samples) if input_samples else 0
            
            output_max = max(abs(s) for s in output_samples) if output_samples else 0
            output_avg = sum(abs(s) for s in output_samples) / len(output_samples) if output_samples else 0
            
            # Channel-specific analysis for dual-channel input
            channel_analysis = {}
            if source_channels >= 4 and target_channels == 2:
                # Analyze 4→2 channel conversion
                for i in range(min(4, source_channels)):
                    channel_samples = input_samples[i::source_channels]
                    if channel_samples:
                        channel_max = max(abs(s) for s in channel_samples)
                        channel_avg = sum(abs(s) for s in channel_samples) / len(channel_samples)
                        channel_analysis[f"input_ch{i}"] = {"max": channel_max, "avg": channel_avg}
                
                # Analyze output channels
                if target_channels == 2:
                    left_samples = output_samples[0::2]
                    right_samples = output_samples[1::2]
                    
                    if left_samples:
                        left_max = max(abs(s) for s in left_samples)
                        left_avg = sum(abs(s) for s in left_samples) / len(left_samples)
                        channel_analysis["output_left"] = {"max": left_max, "avg": left_avg}
                    
                    if right_samples:
                        right_max = max(abs(s) for s in right_samples)
                        right_avg = sum(abs(s) for s in right_samples) / len(right_samples)
                        channel_analysis["output_right"] = {"max": right_max, "avg": right_avg}
            
            return {
                "input_sample_count": input_sample_count,
                "output_sample_count": output_sample_count,
                "input_max_amplitude": input_max,
                "input_avg_amplitude": input_avg,
                "output_max_amplitude": output_max,
                "output_avg_amplitude": output_avg,
                "amplitude_ratio": output_avg / input_avg if input_avg > 0 else 0,
                "channel_analysis": channel_analysis,
                "size_reduction": len(output_data) / len(input_data) if len(input_data) > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"Conversion analysis failed: {e}"}
    
    def convert_channels(self, 
                        audio_data: bytes, 
                        source_channels: int, 
                        target_channels: int,
                        audio_format: str) -> bytes:
        """Convert audio between different channel configurations.
        
        Args:
            audio_data: Raw audio data bytes
            source_channels: Number of input channels
            target_channels: Number of output channels  
            audio_format: Audio format ('int16', 'int24', 'int32', 'float32')
            
        Returns:
            Converted audio data as bytes
            
        Raises:
            ValueError: If unsupported format or channel configuration
        """
        if source_channels == target_channels:
            # No conversion needed
            return audio_data
        
        if audio_format not in self._format_processors:
            raise ValueError(f"Unsupported audio format: {audio_format}")
        
        if source_channels < 1 or target_channels < 1:
            raise ValueError("Channel counts must be positive")
        
        # Enhanced conversion logic for AWS Transcribe 2-channel support
        if target_channels == 1 and source_channels > 1:
            return self._convert_to_mono(audio_data, source_channels, audio_format)
        elif target_channels == 2 and source_channels > 2:
            return self._convert_to_dual_channel(audio_data, source_channels, audio_format)
        elif source_channels == 1 and target_channels > 1:
            return self._convert_from_mono(audio_data, target_channels, audio_format)
        else:
            raise NotImplementedError(f"Channel conversion {source_channels}→{target_channels} not yet implemented")
    
    def convert_to_optimal_channels(self, 
                                   audio_data: bytes, 
                                   source_channels: int,
                                   audio_format: str) -> tuple[bytes, int]:
        """Convert audio to optimal channel configuration for transcription.
        
        Uses specific channel processing strategy:
        - 1-2 channels → 1 channel (mono) - Mix down stereo/mono to mono
        - 3-4 channels → 2 channels - Ch1&2→Channel A, Ch3&4→Channel B for speaker separation
        - >4 channels → Error (not supported)
        
        Args:
            audio_data: Raw audio data bytes
            source_channels: Number of input channels
            audio_format: Audio format string
            
        Returns:
            Tuple of (converted_audio_data, output_channels)
            
        Raises:
            ValueError: If source_channels > 4 (not supported)
        """
        # Increment conversion counter for debugging
        self._conversion_count += 1
        
        if source_channels <= 0:
            raise ValueError("Source channels must be positive")
        elif source_channels <= 2:
            # 1-2 channels → convert to mono
            if source_channels == 1:
                # Already mono - no conversion needed
                logger.debug(f"🔄 ChannelProcessor #{self._conversion_count}: No conversion needed (already mono)")
                return audio_data, 1
            else:
                # 2 channels → mix down to mono
                logger.debug(f"🔄 ChannelProcessor #{self._conversion_count}: Converting 2ch → 1ch (mono)")
                converted_data = self._convert_to_mono(audio_data, source_channels, audio_format)
                
                # Debug analysis for first few conversions
                if self._debug_sample_logging and self._conversion_count <= 10:
                    analysis = self._analyze_conversion_quality(audio_data, converted_data, 
                                                              source_channels, 1, audio_format)
                    if "error" not in analysis:
                        logger.info(f"🔍 ChannelProcessor Analysis #{self._conversion_count} (2ch→1ch):")
                        logger.info(f"   📊 Amplitude: {analysis['input_avg_amplitude']:.1f} → "
                                   f"{analysis['output_avg_amplitude']:.1f} (ratio: {analysis['amplitude_ratio']:.3f})")
                        logger.info(f"   📦 Samples: {analysis['input_sample_count']} → {analysis['output_sample_count']}")
                        logger.info(f"   📏 Size: {len(audio_data)} → {len(converted_data)} bytes ({analysis['size_reduction']:.3f})")
                    else:
                        logger.warning(f"⚠️ ChannelProcessor Analysis #{self._conversion_count}: {analysis['error']}")
                
                return converted_data, 1
        elif source_channels <= 4:
            # 3-4 channels → convert to dual-channel for speaker separation
            # Ch1&2 → Channel A, Ch3&4 → Channel B
            logger.debug(f"🔄 ChannelProcessor #{self._conversion_count}: Converting {source_channels}ch → 2ch (dual-channel)")
            converted_data = self._convert_to_dual_channel(audio_data, source_channels, audio_format)
            
            # Debug analysis for first few conversions and every 100th after that
            should_analyze = (self._debug_sample_logging and self._conversion_count <= 20) or \
                            (self._conversion_count % 100 == 0)
            
            if should_analyze:
                analysis = self._analyze_conversion_quality(audio_data, converted_data, 
                                                          source_channels, 2, audio_format)
                if "error" not in analysis:
                    logger.info(f"🔍 ChannelProcessor Analysis #{self._conversion_count} ({source_channels}ch→2ch):")
                    logger.info(f"   📊 Input amplitude: Max={analysis['input_max_amplitude']}, "
                               f"Avg={analysis['input_avg_amplitude']:.1f}")
                    logger.info(f"   📊 Output amplitude: Max={analysis['output_max_amplitude']}, "
                               f"Avg={analysis['output_avg_amplitude']:.1f}")
                    logger.info(f"   📊 Amplitude ratio: {analysis['amplitude_ratio']:.3f}")
                    logger.info(f"   📦 Samples: {analysis['input_sample_count']} → {analysis['output_sample_count']}")
                    logger.info(f"   📏 Size: {len(audio_data)} → {len(converted_data)} bytes ({analysis['size_reduction']:.3f})")
                    
                    # Channel-specific analysis
                    channel_analysis = analysis.get('channel_analysis', {})
                    if channel_analysis:
                        logger.info(f"   🎚️  Channel details:")
                        for ch_name, ch_data in channel_analysis.items():
                            if isinstance(ch_data, dict) and 'max' in ch_data:
                                logger.info(f"     - {ch_name}: Max={ch_data['max']}, Avg={ch_data['avg']:.1f}")
                    
                    # Critical check: If output is silence, flag it
                    if analysis['output_max_amplitude'] < 10:
                        logger.warning(f"⚠️ ChannelProcessor #{self._conversion_count}: "
                                     f"Output amplitude very low ({analysis['output_max_amplitude']}) - possible silence!")
                else:
                    logger.warning(f"⚠️ ChannelProcessor Analysis #{self._conversion_count}: {analysis['error']}")
            
            return converted_data, 2
        else:
            # >4 channels → not supported
            raise ValueError(f"Unsupported channel count: {source_channels}. Maximum 4 channels supported. "
                           f"Please use an audio device with 4 or fewer channels.")
    
    def _convert_to_mono(self, audio_data: bytes, source_channels: int, audio_format: str) -> bytes:
        """Convert multi-channel audio to mono using the configured mixing strategy.
        
        Args:
            audio_data: Raw multi-channel audio data
            source_channels: Number of source channels
            audio_format: Audio format string
            
        Returns:
            Mono audio data as bytes
        """
        processor = self._format_processors[audio_format]
        return processor(audio_data, source_channels, 1)
    
    def _convert_to_dual_channel(self, audio_data: bytes, source_channels: int, audio_format: str) -> bytes:
        """Convert multi-channel audio to dual-channel using intelligent channel grouping.
        
        Strategy:
        - Channels 1+2 → Channel A (left/front speakers) 
        - Channels 3+4 → Channel B (right/rear speakers)
        - Additional channels grouped into available slots
        
        Args:
            audio_data: Raw multi-channel audio data
            source_channels: Number of source channels (must be > 2)
            audio_format: Audio format string
            
        Returns:
            Dual-channel (stereo) audio data as bytes
        """
        if source_channels <= 2:
            raise ValueError("Dual-channel conversion requires more than 2 source channels")
            
        processor = self._format_processors[audio_format]
        return processor(audio_data, source_channels, 2)
    
    def _convert_from_mono(self, audio_data: bytes, target_channels: int, audio_format: str) -> bytes:
        """Convert mono audio to multi-channel by duplicating the mono signal.
        
        Args:
            audio_data: Raw mono audio data
            target_channels: Number of target channels
            audio_format: Audio format string
            
        Returns:
            Multi-channel audio data as bytes
        """
        processor = self._format_processors[audio_format]  
        return processor(audio_data, 1, target_channels)
    
    def _process_int16(self, audio_data: bytes, source_channels: int, target_channels: int) -> bytes:
        """Process 16-bit integer audio data.
        
        Optimized for minimal latency using efficient integer operations.
        """
        if target_channels == 2 and source_channels > 2:
            # Multi-channel to dual-channel conversion (intelligent grouping)
            sample_count = len(audio_data) // (2 * source_channels)  # 2 bytes per int16 sample
            samples = struct.unpack(f'<{sample_count * source_channels}h', audio_data)
            
            dual_samples = []
            
            for i in range(sample_count):
                # Channel A: Average of channels 0 and 1 (if available)
                channel_a_sum = samples[i * source_channels + 0]  # Channel 0
                if source_channels > 1:
                    channel_a_sum += samples[i * source_channels + 1]  # Channel 1
                    channel_a = channel_a_sum // 2
                else:
                    channel_a = channel_a_sum
                
                # Channel B: Average of channels 2 and 3 (if available)  
                if source_channels > 2:
                    channel_b_sum = samples[i * source_channels + 2]  # Channel 2
                    if source_channels > 3:
                        channel_b_sum += samples[i * source_channels + 3]  # Channel 3
                        channel_b = channel_b_sum // 2
                    else:
                        channel_b = channel_b_sum
                    
                    # If more channels exist (5, 6, 7, 8...), add them to appropriate groups
                    if source_channels > 4:
                        # Add remaining channels to channel groups alternately
                        extra_a = 0
                        extra_b = 0
                        extra_count_a = 0
                        extra_count_b = 0
                        
                        for ch in range(4, source_channels):
                            if ch % 2 == 0:  # Even channels go to A
                                extra_a += samples[i * source_channels + ch]
                                extra_count_a += 1
                            else:  # Odd channels go to B
                                extra_b += samples[i * source_channels + ch]
                                extra_count_b += 1
                        
                        # Average the extras into the main channels
                        if extra_count_a > 0:
                            channel_a = (channel_a * 2 + extra_a) // (2 + extra_count_a)
                        if extra_count_b > 0:
                            channel_b = (channel_b * 2 + extra_b) // (2 + extra_count_b)
                else:
                    # Only 3 channels - duplicate channel A for channel B
                    channel_b = channel_a
                
                dual_samples.extend([channel_a, channel_b])
            
            return struct.pack(f'<{len(dual_samples)}h', *dual_samples)
        
        elif target_channels == 1 and source_channels > 1:
            # Multi-channel to mono conversion
            sample_count = len(audio_data) // (2 * source_channels)  # 2 bytes per int16 sample
            samples = struct.unpack(f'<{sample_count * source_channels}h', audio_data)
            
            mono_samples = []
            
            if self.mixing_strategy == MixingStrategy.AVERAGE:
                # Average all channels - most common and balanced approach
                for i in range(sample_count):
                    channel_sum = 0
                    for ch in range(source_channels):
                        channel_sum += samples[i * source_channels + ch]
                    mono_samples.append(channel_sum // source_channels)
            
            elif self.mixing_strategy == MixingStrategy.LEFT_ONLY:
                # Use left channel only (channel 0)
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels])
            
            elif self.mixing_strategy == MixingStrategy.RIGHT_ONLY:
                # Use right channel only (channel 1, or last channel if not stereo)
                right_channel = 1 if source_channels >= 2 else 0
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels + right_channel])
            
            elif self.mixing_strategy == MixingStrategy.WEIGHTED:
                # Weighted average - give more weight to front channels
                weights = self._get_channel_weights(source_channels)
                for i in range(sample_count):
                    weighted_sum = 0
                    for ch in range(source_channels):
                        weighted_sum += samples[i * source_channels + ch] * weights[ch]
                    mono_samples.append(int(weighted_sum))
            
            return struct.pack(f'<{len(mono_samples)}h', *mono_samples)
        
        elif source_channels == 1 and target_channels > 1:
            # Mono to multi-channel conversion (duplicate mono signal)
            sample_count = len(audio_data) // 2  # 2 bytes per int16 sample
            mono_samples = struct.unpack(f'<{sample_count}h', audio_data)
            
            multi_samples = []
            for sample in mono_samples:
                for _ in range(target_channels):
                    multi_samples.append(sample)
            
            return struct.pack(f'<{len(multi_samples)}h', *multi_samples)
        
        else:
            raise NotImplementedError(f"int16 conversion {source_channels}→{target_channels} not implemented")
    
    def _process_int24(self, audio_data: bytes, source_channels: int, target_channels: int) -> bytes:
        """Process 24-bit integer audio data."""
        # 24-bit processing is more complex due to non-standard byte alignment
        # For now, convert to 32-bit, process, then back to 24-bit
        raise NotImplementedError("int24 processing not yet implemented")
    
    def _process_int32(self, audio_data: bytes, source_channels: int, target_channels: int) -> bytes:
        """Process 32-bit integer audio data."""
        if target_channels == 2 and source_channels > 2:
            # Multi-channel to dual-channel conversion (intelligent grouping)
            sample_count = len(audio_data) // (4 * source_channels)  # 4 bytes per int32 sample
            samples = struct.unpack(f'<{sample_count * source_channels}i', audio_data)
            
            dual_samples = []
            
            for i in range(sample_count):
                # Channel A: Average of channels 0 and 1
                channel_a_sum = samples[i * source_channels + 0]
                if source_channels > 1:
                    channel_a_sum += samples[i * source_channels + 1]
                    channel_a = channel_a_sum // 2
                else:
                    channel_a = channel_a_sum
                
                # Channel B: Average of channels 2 and 3
                if source_channels > 2:
                    channel_b_sum = samples[i * source_channels + 2]
                    if source_channels > 3:
                        channel_b_sum += samples[i * source_channels + 3]
                        channel_b = channel_b_sum // 2
                    else:
                        channel_b = channel_b_sum
                    
                    # Handle additional channels
                    if source_channels > 4:
                        extra_a = sum(samples[i * source_channels + ch] for ch in range(4, source_channels, 2))
                        extra_b = sum(samples[i * source_channels + ch] for ch in range(5, source_channels, 2))
                        extra_count_a = len(range(4, source_channels, 2))
                        extra_count_b = len(range(5, source_channels, 2))
                        
                        if extra_count_a > 0:
                            channel_a = (channel_a * 2 + extra_a) // (2 + extra_count_a)
                        if extra_count_b > 0:
                            channel_b = (channel_b * 2 + extra_b) // (2 + extra_count_b)
                else:
                    channel_b = channel_a
                
                dual_samples.extend([channel_a, channel_b])
            
            return struct.pack(f'<{len(dual_samples)}i', *dual_samples)
        
        elif target_channels == 1 and source_channels > 1:
            # Multi-channel to mono conversion
            sample_count = len(audio_data) // (4 * source_channels)  # 4 bytes per int32 sample
            samples = struct.unpack(f'<{sample_count * source_channels}i', audio_data)
            
            mono_samples = []
            
            if self.mixing_strategy == MixingStrategy.AVERAGE:
                for i in range(sample_count):
                    channel_sum = 0
                    for ch in range(source_channels):
                        channel_sum += samples[i * source_channels + ch]
                    mono_samples.append(channel_sum // source_channels)
            
            elif self.mixing_strategy == MixingStrategy.LEFT_ONLY:
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels])
            
            elif self.mixing_strategy == MixingStrategy.RIGHT_ONLY:
                right_channel = 1 if source_channels >= 2 else 0
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels + right_channel])
            
            elif self.mixing_strategy == MixingStrategy.WEIGHTED:
                weights = self._get_channel_weights(source_channels)
                for i in range(sample_count):
                    weighted_sum = 0
                    for ch in range(source_channels):
                        weighted_sum += samples[i * source_channels + ch] * weights[ch]
                    mono_samples.append(int(weighted_sum))
            
            return struct.pack(f'<{len(mono_samples)}i', *mono_samples)
        
        elif source_channels == 1 and target_channels > 1:
            # Mono to multi-channel conversion
            sample_count = len(audio_data) // 4  # 4 bytes per int32 sample
            mono_samples = struct.unpack(f'<{sample_count}i', audio_data)
            
            multi_samples = []
            for sample in mono_samples:
                for _ in range(target_channels):
                    multi_samples.append(sample)
            
            return struct.pack(f'<{len(multi_samples)}i', *multi_samples)
        
        else:
            raise NotImplementedError(f"int32 conversion {source_channels}→{target_channels} not implemented")
    
    def _process_float32(self, audio_data: bytes, source_channels: int, target_channels: int) -> bytes:
        """Process 32-bit float audio data."""
        if target_channels == 2 and source_channels > 2:
            # Multi-channel to dual-channel conversion (intelligent grouping)
            sample_count = len(audio_data) // (4 * source_channels)  # 4 bytes per float32 sample
            samples = struct.unpack(f'<{sample_count * source_channels}f', audio_data)
            
            dual_samples = []
            
            for i in range(sample_count):
                # Channel A: Average of channels 0 and 1
                channel_a_sum = samples[i * source_channels + 0]
                if source_channels > 1:
                    channel_a_sum += samples[i * source_channels + 1]
                    channel_a = channel_a_sum / 2.0
                else:
                    channel_a = channel_a_sum
                
                # Channel B: Average of channels 2 and 3
                if source_channels > 2:
                    channel_b_sum = samples[i * source_channels + 2]
                    if source_channels > 3:
                        channel_b_sum += samples[i * source_channels + 3]
                        channel_b = channel_b_sum / 2.0
                    else:
                        channel_b = channel_b_sum
                    
                    # Handle additional channels
                    if source_channels > 4:
                        extra_a = sum(samples[i * source_channels + ch] for ch in range(4, source_channels, 2))
                        extra_b = sum(samples[i * source_channels + ch] for ch in range(5, source_channels, 2))
                        extra_count_a = len(range(4, source_channels, 2))
                        extra_count_b = len(range(5, source_channels, 2))
                        
                        if extra_count_a > 0:
                            channel_a = (channel_a * 2.0 + extra_a) / (2.0 + extra_count_a)
                        if extra_count_b > 0:
                            channel_b = (channel_b * 2.0 + extra_b) / (2.0 + extra_count_b)
                else:
                    channel_b = channel_a
                
                dual_samples.extend([channel_a, channel_b])
            
            return struct.pack(f'<{len(dual_samples)}f', *dual_samples)
        
        elif target_channels == 1 and source_channels > 1:
            # Multi-channel to mono conversion
            sample_count = len(audio_data) // (4 * source_channels)  # 4 bytes per float32 sample
            samples = struct.unpack(f'<{sample_count * source_channels}f', audio_data)
            
            mono_samples = []
            
            if self.mixing_strategy == MixingStrategy.AVERAGE:
                for i in range(sample_count):
                    channel_sum = 0.0
                    for ch in range(source_channels):
                        channel_sum += samples[i * source_channels + ch]
                    mono_samples.append(channel_sum / source_channels)
            
            elif self.mixing_strategy == MixingStrategy.LEFT_ONLY:
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels])
            
            elif self.mixing_strategy == MixingStrategy.RIGHT_ONLY:
                right_channel = 1 if source_channels >= 2 else 0
                for i in range(sample_count):
                    mono_samples.append(samples[i * source_channels + right_channel])
            
            elif self.mixing_strategy == MixingStrategy.WEIGHTED:
                weights = self._get_channel_weights(source_channels)
                for i in range(sample_count):
                    weighted_sum = 0.0
                    for ch in range(source_channels):
                        weighted_sum += samples[i * source_channels + ch] * weights[ch]
                    mono_samples.append(weighted_sum)
            
            return struct.pack(f'<{len(mono_samples)}f', *mono_samples)
        
        elif source_channels == 1 and target_channels > 1:
            # Mono to multi-channel conversion
            sample_count = len(audio_data) // 4  # 4 bytes per float32 sample
            mono_samples = struct.unpack(f'<{sample_count}f', audio_data)
            
            multi_samples = []
            for sample in mono_samples:
                for _ in range(target_channels):
                    multi_samples.append(sample)
            
            return struct.pack(f'<{len(multi_samples)}f', *multi_samples)
        
        else:
            raise NotImplementedError(f"float32 conversion {source_channels}→{target_channels} not implemented")
    
    def _get_channel_weights(self, num_channels: int) -> list:
        """Get channel weights for weighted mixing strategy.
        
        Args:
            num_channels: Number of input channels
            
        Returns:
            List of weights for each channel (sum = 1.0)
        """
        if num_channels <= 2:
            # Stereo or mono - equal weights
            return [1.0 / num_channels] * num_channels
        elif num_channels == 4:
            # Quadraphonic: L, R, LS, RS - front channels get more weight
            return [0.35, 0.35, 0.15, 0.15]  # Front 70%, rear 30%
        elif num_channels == 6:
            # 5.1 surround: L, R, C, LFE, LS, RS - front and center get more weight
            return [0.25, 0.25, 0.25, 0.05, 0.10, 0.10]  # Front + center 75%
        else:
            # Unknown configuration - equal weights
            weight = 1.0 / num_channels
            return [weight] * num_channels
    
    def get_processing_info(self) -> dict:
        """Get information about the current processor configuration.
        
        Returns:
            Dict with processor configuration details
        """
        return {
            'mixing_strategy': self.mixing_strategy.value,
            'supported_formats': list(self._format_processors.keys()),
            'optimized_for': 'real-time low-latency processing'
        }


def create_channel_processor(mixing_strategy: str = "average") -> AudioChannelProcessor:
    """Factory function to create an AudioChannelProcessor.
    
    Args:
        mixing_strategy: Mixing strategy name ('average', 'left', 'right', 'weighted')
        
    Returns:
        Configured AudioChannelProcessor instance
        
    Raises:
        ValueError: If mixing strategy is not supported
    """
    try:
        strategy = MixingStrategy(mixing_strategy.lower())
        return AudioChannelProcessor(strategy)
    except ValueError:
        valid_strategies = [s.value for s in MixingStrategy]
        raise ValueError(f"Invalid mixing strategy '{mixing_strategy}'. Valid options: {valid_strategies}")