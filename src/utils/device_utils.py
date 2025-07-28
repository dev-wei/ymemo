"""Audio device enumeration utilities."""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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
        self._devices_cache: Optional[List[AudioDeviceInfo]] = None
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
    
    def get_input_devices(self, refresh: bool = False) -> List[AudioDeviceInfo]:
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
            if (refresh or 
                self._devices_cache is None or 
                current_device_count != self._last_device_count):
                
                self._devices_cache = self._enumerate_input_devices()
                self._last_device_count = current_device_count
            
            return self._devices_cache or []
            
        except Exception as e:
            logger.error(f"Error getting input devices: {e}")
            return []
    
    def _enumerate_input_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate all input devices."""
        devices = []
        
        try:
            default_input_index = self.audio.get_default_input_device_info()['index']
        except Exception:
            default_input_index = -1
        
        device_count = self.audio.get_device_count()
        
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                
                # Only include devices with input channels
                if device_info['maxInputChannels'] > 0:
                    audio_device = AudioDeviceInfo(
                        index=i,
                        name=device_info['name'],
                        max_input_channels=device_info['maxInputChannels'],
                        max_output_channels=device_info['maxOutputChannels'],
                        default_sample_rate=device_info['defaultSampleRate'],
                        is_default_input=(i == default_input_index)
                    )
                    devices.append(audio_device)
                    
            except Exception as e:
                logger.warning(f"Could not get info for device {i}: {e}")
                continue
        
        return devices
    
    def get_device_choices(self, refresh: bool = False) -> List[Tuple[str, int]]:
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
    
    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
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
    
    def get_device_by_index(self, index: int) -> Optional[AudioDeviceInfo]:
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
                rate=int(device_info['defaultSampleRate']),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
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


def get_audio_devices(refresh: bool = False) -> List[Tuple[str, int]]:
    """Convenience function to get audio device choices.
    
    Args:
        refresh: Force refresh of device list
        
    Returns:
        List of (display_name, device_index) tuples
    """
    return device_manager.get_device_choices(refresh)


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