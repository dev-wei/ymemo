"""Audio testing utilities for generating and manipulating test audio data."""

import math
import os
import tempfile
import wave
from pathlib import Path

import numpy as np


class AudioFileGenerator:
    """Utility for generating test audio files."""

    @staticmethod
    def create_sine_wave_file(
        frequency: int = 440,
        duration: float = 1.0,
        sample_rate: int = 16000,
        amplitude: float = 0.5,
        filename: str | None = None,
    ) -> str:
        """Create a WAV file with a sine wave."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Generate sine wave
        samples = int(sample_rate * duration)
        audio_data = []

        for i in range(samples):
            sample = int(
                32767 * amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
            )
            audio_data.append(sample)

        # Convert to bytes
        audio_bytes = np.array(audio_data, dtype=np.int16).tobytes()

        # Write WAV file
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        return filename

    @staticmethod
    def create_silence_file(
        duration: float = 1.0, sample_rate: int = 16000, filename: str | None = None
    ) -> str:
        """Create a WAV file with silence."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Generate silence
        samples = int(sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.int16)

        # Write WAV file
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return filename

    @staticmethod
    def create_noise_file(
        duration: float = 1.0,
        sample_rate: int = 16000,
        noise_level: float = 0.1,
        filename: str | None = None,
    ) -> str:
        """Create a WAV file with white noise."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Generate white noise
        samples = int(sample_rate * duration)
        audio_data = np.random.uniform(-1, 1, samples) * noise_level * 32767
        audio_data = audio_data.astype(np.int16)

        # Write WAV file
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return filename

    @staticmethod
    def create_mixed_audio_file(
        components: list[
            tuple[int, float, float]
        ],  # [(frequency, duration, amplitude)]
        sample_rate: int = 16000,
        filename: str | None = None,
    ) -> str:
        """Create a WAV file with mixed audio components."""
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        total_duration = sum(duration for _, duration, _ in components)
        total_samples = int(sample_rate * total_duration)
        audio_data = np.zeros(total_samples, dtype=np.float32)

        sample_offset = 0
        for frequency, duration, amplitude in components:
            samples = int(sample_rate * duration)

            # Generate component
            for i in range(samples):
                sample_value = amplitude * math.sin(
                    2 * math.pi * frequency * i / sample_rate
                )
                audio_data[sample_offset + i] += sample_value

            sample_offset += samples

        # Convert to 16-bit integers
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return filename


class AudioDataGenerator:
    """Utility for generating raw audio data for testing."""

    @staticmethod
    def generate_chunk_sequence(
        chunk_size: int = 1024, num_chunks: int = 10, pattern: str = "incremental"
    ) -> list[bytes]:
        """Generate a sequence of audio chunks with different patterns."""
        chunks = []

        for i in range(num_chunks):
            if pattern == "incremental":
                # Each chunk has incrementing pattern
                chunk = bytes([(i + j) % 256 for j in range(chunk_size)])
            elif pattern == "sine":
                # Each chunk contains part of a sine wave
                chunk = []
                for j in range(chunk_size // 2):  # 16-bit samples
                    sample_index = i * (chunk_size // 2) + j
                    sample = int(
                        32767 * 0.5 * math.sin(2 * math.pi * 440 * sample_index / 16000)
                    )
                    # Convert to little-endian 16-bit
                    chunk.extend([(sample & 0xFF), ((sample >> 8) & 0xFF)])
                chunk = bytes(chunk)
            elif pattern == "silence":
                # Silent chunks
                chunk = b"\x00" * chunk_size
            elif pattern == "noise":
                # Random noise
                chunk = bytes([np.random.randint(0, 256) for _ in range(chunk_size)])
            else:
                # Default pattern
                chunk = bytes([i % 256] * chunk_size)

            chunks.append(chunk)

        return chunks

    @staticmethod
    def generate_realistic_speech_chunks(
        num_chunks: int = 20, chunk_size: int = 1024
    ) -> list[bytes]:
        """Generate chunks that simulate realistic speech patterns."""
        chunks = []

        # Speech has pauses and varying amplitude
        for i in range(num_chunks):
            if i % 5 == 0:  # Every 5th chunk is silence (pause)
                chunk = b"\x00" * chunk_size
            else:
                # Generate speech-like audio with varying frequency
                base_freq = 200 + (i % 4) * 50  # Vary frequency for speech-like quality
                amplitude = 0.3 + 0.4 * math.sin(i * 0.5)  # Varying amplitude

                chunk = []
                for j in range(chunk_size // 2):
                    sample_index = i * (chunk_size // 2) + j
                    sample = int(
                        32767
                        * amplitude
                        * math.sin(2 * math.pi * base_freq * sample_index / 16000)
                    )

                    # Add some harmonics for more realistic sound
                    sample += int(
                        32767
                        * amplitude
                        * 0.3
                        * math.sin(2 * math.pi * base_freq * 2 * sample_index / 16000)
                    )

                    # Clip and convert to bytes
                    sample = max(-32767, min(32767, sample))
                    chunk.extend([(sample & 0xFF), ((sample >> 8) & 0xFF)])

                chunk = bytes(chunk)

            chunks.append(chunk)

        return chunks


class AudioAnalyzer:
    """Utility for analyzing audio data in tests."""

    @staticmethod
    def analyze_chunk_properties(audio_chunk: bytes) -> dict:
        """Analyze properties of an audio chunk."""
        if len(audio_chunk) % 2 != 0:
            raise ValueError(
                "Audio chunk must have even number of bytes for 16-bit samples"
            )

        # Convert to 16-bit samples
        samples = []
        for i in range(0, len(audio_chunk), 2):
            sample = int.from_bytes(
                audio_chunk[i : i + 2], byteorder="little", signed=True
            )
            samples.append(sample)

        samples = np.array(samples)

        return {
            "length_bytes": len(audio_chunk),
            "length_samples": len(samples),
            "max_amplitude": int(np.max(np.abs(samples))),
            "rms_amplitude": float(np.sqrt(np.mean(samples**2))),
            "is_silence": np.all(samples == 0),
            "peak_to_peak": int(np.max(samples) - np.min(samples)),
            "zero_crossings": int(np.sum(np.diff(np.signbit(samples)))),
        }

    @staticmethod
    def detect_audio_pattern(chunks: list[bytes]) -> str:
        """Detect the pattern in a sequence of audio chunks."""
        if not chunks:
            return "empty"

        # Analyze first few chunks
        properties = [
            AudioAnalyzer.analyze_chunk_properties(chunk) for chunk in chunks[:5]
        ]

        # Check for silence
        if all(prop["is_silence"] for prop in properties):
            return "silence"

        # Check for consistent amplitude (synthetic)
        rms_values = [prop["rms_amplitude"] for prop in properties]
        if len({f"{rms:.0f}" for rms in rms_values}) == 1:
            return "synthetic"

        # Check for speech-like patterns (varying amplitude)
        if max(rms_values) > 2 * min(rms_values):
            return "speech_like"

        return "unknown"


class AudioFileManager:
    """Utility for managing test audio files."""

    def __init__(self, temp_dir: str | None = None):
        """Initialize with optional temporary directory."""
        self.temp_dir = (
            Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "audio_tests"
        )
        self.temp_dir.mkdir(exist_ok=True)
        self.created_files = []

    def create_test_file(
        self, file_type: str = "sine", duration: float = 1.0, **kwargs
    ) -> str:
        """Create a test audio file and track it for cleanup."""
        filename = str(
            self.temp_dir / f"test_{file_type}_{len(self.created_files)}.wav"
        )

        if file_type == "sine":
            AudioFileGenerator.create_sine_wave_file(
                duration=duration, filename=filename, **kwargs
            )
        elif file_type == "silence":
            AudioFileGenerator.create_silence_file(
                duration=duration, filename=filename, **kwargs
            )
        elif file_type == "noise":
            AudioFileGenerator.create_noise_file(
                duration=duration, filename=filename, **kwargs
            )
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        self.created_files.append(filename)
        return filename

    def cleanup(self):
        """Clean up all created test files."""
        for filename in self.created_files:
            try:
                if os.path.exists(filename):
                    os.unlink(filename)
            except Exception as e:
                print(f"Warning: Failed to cleanup {filename}: {e}")

        self.created_files.clear()

        # Remove temp directory if empty
        try:
            self.temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience functions for quick test file creation
def create_test_sine_wave(duration: float = 1.0, frequency: int = 440) -> str:
    """Quick function to create a sine wave test file."""
    return AudioFileGenerator.create_sine_wave_file(
        frequency=frequency, duration=duration
    )


def create_test_silence(duration: float = 1.0) -> str:
    """Quick function to create a silence test file."""
    return AudioFileGenerator.create_silence_file(duration=duration)


def create_test_audio_chunks(count: int = 10, size: int = 1024) -> list[bytes]:
    """Quick function to create test audio chunks."""
    return AudioDataGenerator.generate_chunk_sequence(
        num_chunks=count, chunk_size=size, pattern="sine"
    )
