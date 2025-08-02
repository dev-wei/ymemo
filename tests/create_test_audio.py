#!/usr/bin/env python3
"""Create test audio file for reliable testing."""

import os
import sys
import wave

import numpy as np

# Add project root to sys.path for imports (work in any environment)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def create_test_audio():
    """Create a simple test audio file with synthesized speech-like tones."""

    # Audio parameters
    sample_rate = 16000  # 16kHz as required by AWS Transcribe
    duration = 5.0  # 5 seconds

    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a simple audio signal that simulates speech patterns
    # Mix of different frequencies to simulate speech
    audio = np.zeros_like(t)

    # Add some speech-like frequency components
    # Fundamental frequency around 150Hz (typical for speech)
    audio += 0.3 * np.sin(2 * np.pi * 150 * t)

    # Add some harmonics
    audio += 0.2 * np.sin(2 * np.pi * 300 * t)
    audio += 0.1 * np.sin(2 * np.pi * 450 * t)

    # Add some higher frequency content
    audio += 0.1 * np.sin(2 * np.pi * 800 * t)

    # Add some noise to make it more realistic
    audio += 0.05 * np.random.normal(0, 1, len(t))

    # Create speech-like envelope (amplitude variations)
    envelope = np.abs(np.sin(2 * np.pi * 2 * t))  # 2Hz modulation
    audio *= envelope

    # Normalize to 16-bit range
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create test directory if it doesn't exist (relative to script location)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(test_dir, exist_ok=True)

    # Write WAV file
    wav_path = os.path.join(test_dir, "test_audio.wav")

    with wave.open(wav_path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"Created test audio file: {wav_path}")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Sample rate: {sample_rate} Hz")
    print("   - Channels: 1 (mono)")
    print("   - Format: 16-bit PCM")

    return wav_path


if __name__ == "__main__":
    try:
        wav_path = create_test_audio()
        print("\nTest audio file created successfully!")
        print(f"Path: {wav_path}")
    except Exception as e:
        print(f"Failed to create test audio: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
