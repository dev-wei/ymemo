"""Audio device enumeration utilities."""

import logging
from dataclasses import dataclass
from typing import Any

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


class AudioDeviceManager:
    """Manages audio device enumeration and selection."""

    def __init__(self):
        self.audio = None
        self._devices_cache: list[AudioDeviceInfo] | None = None
        self._last_device_count = 0

    def _initialize_pyaudio(self) -> bool:
        """Initialize PyAudio if available."""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio is not available. Install with: pip install pyaudio")
            return False

        try:
            if not self.audio:
                self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            return False

    def get_input_devices(self, refresh: bool = False) -> list[AudioDeviceInfo]:
        """Get all available input audio devices.

        Args:
            refresh: Force refresh of device list

        Returns:
            List of AudioDeviceInfo objects for input devices
        """
        if not self._initialize_pyaudio():
            return []

        try:
            current_device_count = self.audio.get_device_count()

            # Check if we need to refresh
            if (
                refresh
                or self._devices_cache is None
                or current_device_count != self._last_device_count
            ):
                self._devices_cache = self._enumerate_input_devices()
                self._last_device_count = current_device_count

            return self._devices_cache or []

        except Exception as e:
            logger.error(f"Error getting input devices: {e}")
            return []

    def _enumerate_input_devices(self) -> list[AudioDeviceInfo]:
        """Enumerate all input devices."""
        devices = []

        try:
            default_input_index = self.audio.get_default_input_device_info()["index"]
        except Exception:
            default_input_index = -1

        device_count = self.audio.get_device_count()

        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)

                # Only include devices with input channels
                if device_info["maxInputChannels"] > 0:
                    audio_device = AudioDeviceInfo(
                        index=i,
                        name=device_info["name"],
                        max_input_channels=device_info["maxInputChannels"],
                        max_output_channels=device_info["maxOutputChannels"],
                        default_sample_rate=device_info["defaultSampleRate"],
                        is_default_input=(i == default_input_index),
                    )
                    devices.append(audio_device)

            except Exception as e:
                logger.warning(f"Could not get info for device {i}: {e}")
                continue

        return devices

    def get_device_choices(self, refresh: bool = False) -> list[tuple[str, int]]:
        """Get device choices formatted for Gradio dropdown.

        Args:
            refresh: Force refresh of device list

        Returns:
            List of (display_name, device_index) tuples
        """
        devices = self.get_input_devices(refresh)

        if not devices:
            return [("No microphones detected", -1)]

        choices = []
        for device in devices:
            display_name = device.name

            # Add default indicator
            if device.is_default_input:
                display_name += " (Default)"

            # Add channel info for clarity
            if device.max_input_channels > 1:
                display_name += f" ({device.max_input_channels} channels)"

            choices.append((display_name, device.index))

        return choices

    def get_default_input_device(self) -> AudioDeviceInfo | None:
        """Get the default input device.

        Returns:
            AudioDeviceInfo for default input device, or None if not found
        """
        devices = self.get_input_devices()

        for device in devices:
            if device.is_default_input:
                return device

        # If no default found, return first available device
        if devices:
            return devices[0]

        return None

    def get_device_by_index(self, index: int) -> AudioDeviceInfo | None:
        """Get device info by index.

        Args:
            index: Device index

        Returns:
            AudioDeviceInfo or None if not found
        """
        devices = self.get_input_devices()

        for device in devices:
            if device.index == index:
                return device

        return None

    def test_device(self, device_index: int) -> bool:
        """Test if a device is working.

        Args:
            device_index: Index of device to test

        Returns:
            True if device is working, False otherwise
        """
        if not self._initialize_pyaudio():
            return False

        try:
            device_info = self.audio.get_device_info_by_index(device_index)

            # Try to open a stream briefly
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=int(device_info["defaultSampleRate"]),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
            )

            # Read a small chunk to test
            stream.read(1024, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()

            return True

        except Exception as e:
            logger.error(f"Device test failed for index {device_index}: {e}")
            return False

    def cleanup(self):
        """Cleanup PyAudio resources."""
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None

        self._devices_cache = None
        self._last_device_count = 0


# Global device manager instance
device_manager = AudioDeviceManager()


def get_audio_devices(refresh: bool = False) -> list[tuple[str, int]]:
    """Convenience function to get audio device choices.

    Args:
        refresh: Force refresh of device list

    Returns:
        List of (display_name, device_index) tuples
    """
    return device_manager.get_device_choices(refresh)


def get_supported_audio_devices(refresh: bool = False) -> list[tuple[str, int]]:
    """Get audio device choices filtering out devices with >2 channels.

    Args:
        refresh: Force refresh of device list

    Returns:
        List of (display_name, device_index) tuples for devices with ≤2 channels
    """
    all_devices = device_manager.get_device_choices(refresh)
    supported_devices = []

    for display_name, device_index in all_devices:
        try:
            device = device_manager.get_device_by_index(device_index)
            if device and device.max_input_channels <= 2:
                supported_devices.append((display_name, device_index))
            elif device:
                logger.info(
                    f"🚫 Filtering out device '{device.name}' - {device.max_input_channels} channels (only 1-2 channels supported)"
                )
        except Exception as e:
            logger.warning(f"⚠️ Error checking device {device_index}: {e}")
            # Include device if we can't determine channel count (safer to allow)
            supported_devices.append((display_name, device_index))

    logger.info(
        f"📋 Supported devices: {len(supported_devices)}/{len(all_devices)} devices support ≤2 channels"
    )
    return supported_devices


def get_default_device_index() -> int:
    """Get the default input device index.

    Returns:
        Device index, or -1 if no devices available
    """
    default_device = device_manager.get_default_input_device()
    return default_device.index if default_device else -1


def test_audio_device(device_index: int) -> bool:
    """Test if an audio device is working.

    Args:
        device_index: Index of device to test

    Returns:
        True if device is working, False otherwise
    """
    return device_manager.test_device(device_index)


def get_device_max_channels(device_index: int) -> int:
    """Get the maximum input channels supported by a device.

    Args:
        device_index: Index of device to query

    Returns:
        Maximum input channels supported by the device, or 1 if detection fails
    """
    try:
        device = device_manager.get_device_by_index(device_index)
        if device:
            max_channels = device.max_input_channels
            logger.info(
                f"🔍 Device {device_index} ({device.name}) supports max {max_channels} input channels"
            )
            return max_channels
        logger.warning(f"⚠️ Device {device_index} not found, defaulting to 1 channel")
        return 1
    except Exception as e:
        logger.error(f"❌ Error getting max channels for device {device_index}: {e}")
        return 1


def get_optimal_channels(device_index: int, requested_channels: int) -> int:
    """Get the optimal number of channels for a device.

    Args:
        device_index: Index of device to use
        requested_channels: Number of channels requested by configuration

    Returns:
        Optimal number of channels to use (min of requested and device max)
    """
    try:
        max_channels = get_device_max_channels(device_index)
        optimal_channels = min(requested_channels, max_channels)

        if optimal_channels != requested_channels:
            logger.info(
                f"🔧 Channel optimization: Requested {requested_channels}, "
                f"device max {max_channels}, using {optimal_channels}"
            )
        else:
            logger.debug(
                f"✅ Channel configuration: Using {optimal_channels} channels as requested"
            )

        return optimal_channels
    except Exception as e:
        logger.error(f"❌ Error optimizing channels for device {device_index}: {e}")
        logger.info("🔧 Falling back to mono (1 channel)")
        return 1


def validate_device_config(
    device_index: int, channels: int, sample_rate: int
) -> dict[str, Any]:
    """Validate and optimize audio configuration for a specific device.

    Args:
        device_index: Index of device to validate against
        channels: Requested number of channels
        sample_rate: Requested sample rate

    Returns:
        Dict containing validated configuration with keys:
        - channels: Optimized channel count
        - sample_rate: Validated sample rate
        - device_info: Device information dict
        - warnings: List of configuration warnings
    """
    warnings = []

    try:
        device = device_manager.get_device_by_index(device_index)
        if not device:
            warnings.append(f"Device {device_index} not found")
            return {
                "channels": 1,
                "sample_rate": sample_rate,
                "device_info": {},
                "warnings": warnings,
            }

        # Check for >2 channel devices - only mono and stereo supported
        if device.max_input_channels > 2:
            error_msg = (
                f"Device '{device.name}' has {device.max_input_channels} channels. "
                f"Only 1-2 channels supported. Please select a different audio device."
            )
            warnings.append(error_msg)
            logger.warning(f"⚠️ Device validation: {error_msg}")
            # Return error state - this device should not be used
            return {
                "channels": 0,  # Invalid channel count to indicate error
                "sample_rate": sample_rate,
                "device_info": {
                    "name": device.name,
                    "index": device.index,
                    "max_input_channels": device.max_input_channels,
                    "error": "Too many channels",
                },
                "warnings": warnings,
                "error": error_msg,
            }

        # Optimize channels
        optimal_channels = get_optimal_channels(device_index, channels)
        if optimal_channels != channels:
            warnings.append(
                f"Channel count reduced from {channels} to {optimal_channels} due to device limitations"
            )

        # Validate sample rate
        device_sample_rate = int(device.default_sample_rate)
        logger.info(f"🎚️ Device {device_index} sample rate validation:")
        logger.info(f"   📊 System config requests: {sample_rate}Hz")
        logger.info(f"   🎛️ Device default supports: {device_sample_rate}Hz")

        if sample_rate != device_sample_rate:
            logger.warning(
                f"⚠️ Sample rate mismatch - using requested {sample_rate}Hz instead of device default {device_sample_rate}Hz"
            )
            warnings.append(
                f"Requested sample rate {sample_rate}Hz differs from device default {device_sample_rate}Hz"
            )
            logger.warning(
                "   💡 This could cause audio quality issues or speed/pitch problems"
            )
        else:
            logger.info("✅ Sample rate matches device default - optimal compatibility")

        device_info = {
            "name": device.name,
            "index": device.index,
            "max_input_channels": device.max_input_channels,
            "default_sample_rate": device.default_sample_rate,
            "is_default": device.is_default_input,
        }

        # Option: Use device default sample rate if it differs significantly from requested
        # This could prevent audio corruption when devices have different defaults
        final_sample_rate = sample_rate
        if abs(sample_rate - device_sample_rate) > 1000:  # More than 1kHz difference
            logger.warning(f"⚠️ Large sample rate difference detected!")
            logger.warning(
                f"   💡 Consider using device default {device_sample_rate}Hz for better compatibility"
            )
            logger.warning(
                f"   🔧 Current: using requested {sample_rate}Hz (may cause audio issues)"
            )

        return {
            "channels": optimal_channels,
            "sample_rate": final_sample_rate,
            "device_info": device_info,
            "device_default_sample_rate": device_sample_rate,  # Add this for debugging
            "warnings": warnings,
        }

    except Exception as e:
        logger.error(f"❌ Error validating device config for {device_index}: {e}")
        warnings.append(f"Device validation failed: {e}")
        return {
            "channels": 1,
            "sample_rate": sample_rate,
            "device_info": {},
            "warnings": warnings,
        }
