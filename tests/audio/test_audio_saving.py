"""Tests for audio saving functionality including channel splitting and file writing.

Tests the audio file writer, dual channel saver, and channel splitter
components that are used for debugging audio quality issues.

Migrated and adapted from root directory test_raw_audio_debug.py and debug_test.py
"""

import os
import struct
import tempfile
import time

import pytest

from src.audio.audio_file_writer import AudioFileWriter, DualChannelAudioSaver
from src.audio.channel_splitter import AudioChannelSplitter
from tests.base.base_test import BaseTest


class TestAudioFileWriter(BaseTest):
    """Test the AudioFileWriter component."""

    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory for audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_audio_file_writer_creation(self, temp_audio_dir):
        """Test AudioFileWriter can be created with proper parameters."""
        file_path = os.path.join(temp_audio_dir, "test_audio.wav")

        writer = AudioFileWriter(
            file_path=file_path,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            max_duration=10,
        )

        assert writer is not None
        assert str(writer.file_path) == file_path
        assert writer.sample_rate == 16000
        assert writer.channels == 1

    def test_audio_file_writer_recording_lifecycle(self, temp_audio_dir):
        """Test complete recording lifecycle."""
        file_path = os.path.join(temp_audio_dir, "test_recording.wav")

        writer = AudioFileWriter(
            file_path=file_path,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            max_duration=5,
        )

        # Test start recording
        assert writer.start_recording() is True
        assert writer.is_recording is True

        # Write some test audio data (1024 samples of sine wave pattern)
        test_samples = []
        for i in range(1024):
            # Simple sine wave pattern
            sample_value = int(1000 * (1 if i % 100 < 50 else -1))
            test_samples.append(sample_value)

        test_data = struct.pack("<" + "h" * len(test_samples), *test_samples)

        # Write several chunks
        for i in range(5):
            success = writer.write_audio_data(test_data)
            assert success is True

        # Stop recording
        stats = writer.stop_recording()
        assert stats is not None
        assert "duration_seconds" in stats
        assert "file_path" in stats
        assert stats["duration_seconds"] > 0

        # Verify file was created
        assert os.path.exists(file_path)
        file_size = os.path.getsize(file_path)
        assert file_size > 44  # More than just WAV header

    def test_audio_file_writer_duration_calculation(self, temp_audio_dir):
        """Test that duration is calculated correctly from samples, not wall clock time."""
        file_path = os.path.join(temp_audio_dir, "test_duration.wav")

        writer = AudioFileWriter(
            file_path=file_path,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            max_duration=10,
        )

        writer.start_recording()

        # Write exactly 1 second worth of audio (16000 samples at 16kHz)
        chunk_size = 1024
        chunks_needed = 16000 // chunk_size  # ~15.6 chunks = 1 second

        for _i in range(chunks_needed):
            test_samples = [1000 if j % 2 == 0 else -1000 for j in range(chunk_size)]
            test_data = struct.pack("<" + "h" * len(test_samples), *test_samples)
            writer.write_audio_data(test_data)

        # Add small delay to test that duration is based on samples, not wall clock
        time.sleep(0.5)

        stats = writer.stop_recording()

        # Duration should be close to 1 second (based on samples), not ~1.5 seconds (wall clock)
        assert 0.9 <= stats["duration_seconds"] <= 1.1
        assert (
            abs(stats["duration_seconds"] - 1.0) < 0.1
        )  # Should be very close to 1 second


class TestDualChannelAudioSaver(BaseTest):
    """Test the DualChannelAudioSaver component."""

    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory for audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_dual_channel_saver_creation(self, temp_audio_dir):
        """Test DualChannelAudioSaver creation and initialization."""
        saver = DualChannelAudioSaver(
            save_path=temp_audio_dir, sample_rate=16000, duration=10
        )

        assert saver is not None
        file_paths = saver.get_file_paths()
        assert "left" in file_paths
        assert "right" in file_paths
        assert temp_audio_dir in file_paths["left"]
        assert temp_audio_dir in file_paths["right"]

    def test_dual_channel_recording_lifecycle(self, temp_audio_dir):
        """Test complete dual channel recording lifecycle."""
        saver = DualChannelAudioSaver(
            save_path=temp_audio_dir, sample_rate=16000, duration=5
        )

        # Start recording
        assert saver.start_recording() is True
        assert saver.is_active is True

        # Create test audio for both channels
        chunk_size = 1024
        left_samples = [1000 if i % 50 < 25 else -1000 for i in range(chunk_size)]
        right_samples = [1500 if i % 30 < 15 else -1500 for i in range(chunk_size)]

        left_data = struct.pack("<" + "h" * len(left_samples), *left_samples)
        right_data = struct.pack("<" + "h" * len(right_samples), *right_samples)

        # Write several chunks to both channels
        for _i in range(10):
            left_success = saver.write_left_audio(left_data)
            right_success = saver.write_right_audio(right_data)
            assert left_success is True
            assert right_success is True

        # Stop recording
        stats = saver.stop_recording()
        assert stats is not None
        assert "left_channel" in stats
        assert "right_channel" in stats

        # Verify both files were created
        file_paths = saver.get_file_paths()
        for _channel, file_path in file_paths.items():
            assert os.path.exists(file_path)
            file_size = os.path.getsize(file_path)
            assert file_size > 44  # More than just WAV header

    def test_dual_channel_saver_without_recording(self, temp_audio_dir):
        """Test behavior when stopping without starting recording."""
        saver = DualChannelAudioSaver(
            save_path=temp_audio_dir, sample_rate=16000, duration=5
        )

        # Try to stop without starting
        stats = saver.stop_recording()
        # Should handle gracefully (might return None or empty stats)
        assert stats is None or isinstance(stats, dict)


class TestAudioChannelSplitter(BaseTest):
    """Test the AudioChannelSplitter component."""

    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory for audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_channel_splitter_creation(self, temp_audio_dir):
        """Test AudioChannelSplitter creation."""
        splitter = AudioChannelSplitter(
            audio_format="int16",
            enable_audio_saving=True,
            audio_save_path=temp_audio_dir,
            sample_rate=16000,
            save_duration=10,
        )

        assert splitter is not None
        assert splitter.enable_audio_saving is True

    def test_stereo_chunk_splitting(self, temp_audio_dir):
        """Test splitting stereo audio chunks."""
        splitter = AudioChannelSplitter(
            audio_format="int16",
            enable_audio_saving=False,  # Disable saving for this test
            sample_rate=16000,
        )

        # Create test stereo audio chunk
        chunk_size = 1024  # sample pairs
        stereo_samples = []

        for i in range(chunk_size):
            left_sample = 1000 + (i % 100)  # Left channel with variation
            right_sample = 2000 + (i % 80)  # Right channel with different variation
            stereo_samples.extend([left_sample, right_sample])

        stereo_chunk = struct.pack("<" + "h" * len(stereo_samples), *stereo_samples)

        # Split the chunk
        result = splitter.split_stereo_chunk(stereo_chunk)

        assert result.split_successful is True
        assert result.error_message is None
        assert len(result.left_channel) > 0
        assert len(result.right_channel) > 0

        # Verify channels have expected sizes
        expected_mono_size = len(stereo_chunk) // 2  # Half the size for mono
        assert len(result.left_channel) == expected_mono_size
        assert len(result.right_channel) == expected_mono_size

        # Verify metrics
        assert result.left_metrics is not None
        assert result.right_metrics is not None
        assert isinstance(result.left_metrics.activity_level, str)
        assert isinstance(result.right_metrics.activity_level, str)
        assert result.left_metrics.activity_level in [
            "silent",
            "very_quiet",
            "quiet",
            "normal",
            "loud",
            "very_loud",
        ]
        assert result.right_metrics.activity_level in [
            "silent",
            "very_quiet",
            "quiet",
            "normal",
            "loud",
            "very_loud",
        ]

    def test_channel_splitter_with_audio_saving(self, temp_audio_dir):
        """Test channel splitter with audio saving enabled."""
        splitter = AudioChannelSplitter(
            audio_format="int16",
            enable_audio_saving=True,
            audio_save_path=temp_audio_dir,
            sample_rate=16000,
            save_duration=5,
        )

        # Create and process multiple stereo chunks
        chunk_size = 1024

        for chunk_idx in range(20):  # Process 20 chunks
            stereo_samples = []

            for sample_idx in range(chunk_size):
                # Create distinguishable patterns for left/right
                left_sample = 1000 if (chunk_idx + sample_idx) % 100 < 50 else -1000
                right_sample = 1500 if (chunk_idx + sample_idx) % 60 < 30 else -1500
                stereo_samples.extend([left_sample, right_sample])

            stereo_chunk = struct.pack("<" + "h" * len(stereo_samples), *stereo_samples)

            result = splitter.split_stereo_chunk(stereo_chunk)
            assert result.split_successful is True

        # Get statistics
        stats = splitter.get_statistics()
        assert stats is not None
        assert stats["total_chunks_processed"] == 20
        assert stats["total_bytes_processed"] > 0

        # Stop audio saving
        save_result = splitter.stop_audio_saving()
        if save_result:  # Only check if saving was actually active
            assert "left_channel" in save_result
            assert "right_channel" in save_result

            # Verify files were created
            for _channel_name, channel_stats in save_result.items():
                if isinstance(channel_stats, dict) and "file_path" in channel_stats:
                    file_path = channel_stats["file_path"]
                    assert os.path.exists(file_path)
                    file_size = os.path.getsize(file_path)
                    assert file_size > 44  # More than just header

    def test_invalid_stereo_chunk_handling(self):
        """Test handling of invalid stereo chunks."""
        splitter = AudioChannelSplitter(audio_format="int16", enable_audio_saving=False)

        # Test with odd number of samples (invalid for stereo)
        invalid_samples = [1000, 2000, 3000]  # 3 samples (not divisible by 2)
        invalid_chunk = struct.pack("<" + "h" * len(invalid_samples), *invalid_samples)

        result = splitter.split_stereo_chunk(invalid_chunk)
        assert result.split_successful is False
        assert result.error_message is not None
        assert len(result.error_message) > 0

    def test_empty_chunk_handling(self):
        """Test handling of empty audio chunks."""
        splitter = AudioChannelSplitter(audio_format="int16", enable_audio_saving=False)

        # Test with empty chunk (empty chunks are handled successfully, not as errors)
        result = splitter.split_stereo_chunk(b"")
        assert result.split_successful is True
        assert result.error_message is None
        assert len(result.left_channel) == 0
        assert len(result.right_channel) == 0


class TestAudioSavingIntegration(BaseTest):
    """Test integration between audio saving components."""

    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory for audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_realistic_audio_processing_pipeline(self, temp_audio_dir):
        """Test a realistic audio processing pipeline with saving."""
        # Simulate the pattern used in the real application
        splitter = AudioChannelSplitter(
            audio_format="int16",
            enable_audio_saving=True,
            audio_save_path=temp_audio_dir,
            sample_rate=16000,
            save_duration=3,  # Short duration for test
        )

        # Simulate PyAudio input pattern
        chunk_size = 1024
        sample_rate = 16000
        duration_seconds = 2
        total_chunks = (duration_seconds * sample_rate) // chunk_size

        successful_chunks = 0

        for chunk_idx in range(total_chunks):
            # Create realistic stereo audio data
            stereo_samples = []

            for sample_idx in range(chunk_size):
                time_offset = (chunk_idx * chunk_size + sample_idx) / sample_rate

                # Left channel: 440Hz-ish pattern
                left_sample = int(1000 * (1 if int(time_offset * 440) % 2 == 0 else -1))
                # Right channel: 880Hz-ish pattern
                right_sample = int(
                    1500 * (1 if int(time_offset * 880) % 2 == 0 else -1)
                )

                stereo_samples.extend([left_sample, right_sample])

            # Pack and process
            stereo_chunk = struct.pack("<" + "h" * len(stereo_samples), *stereo_samples)
            result = splitter.split_stereo_chunk(stereo_chunk)

            if result.split_successful:
                successful_chunks += 1

        # Verify processing was successful
        assert successful_chunks == total_chunks

        # Get final statistics
        stats = splitter.get_statistics()
        assert stats["total_chunks_processed"] == total_chunks
        assert stats["total_bytes_processed"] > 0

        # Stop and verify audio saving
        save_result = splitter.stop_audio_saving()
        if save_result:
            for channel in ["left_channel", "right_channel"]:
                if channel in save_result:
                    channel_stats = save_result[channel]
                    if isinstance(channel_stats, dict) and "file_path" in channel_stats:
                        file_path = channel_stats["file_path"]
                        assert os.path.exists(file_path)

                        # Verify the file has reasonable content
                        file_size = os.path.getsize(file_path)
                        assert file_size > 1000  # Should be substantial

                        # Duration should be reasonable
                        duration = channel_stats.get("duration_seconds", 0)
                        assert 1.5 <= duration <= 2.5  # Close to expected 2 seconds
