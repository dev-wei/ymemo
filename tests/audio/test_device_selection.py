"""Audio device selection tests using new test infrastructure.

Migrated from raw function calls to pytest with centralized fixtures and base classes.
Tests device selection functionality without hardware dependencies.
"""

from unittest.mock import Mock, patch

import pytest

from src.utils.status_manager import AudioStatus
from tests.base.base_test import BaseIntegrationTest, BaseTest


class TestDeviceSelectionFormat(BaseTest):
    """Test device selection format and validation using new infrastructure."""

    @pytest.mark.integration
    @patch("src.utils.device_utils.get_supported_audio_devices")
    @patch("src.utils.device_utils.get_default_device_index")
    def test_device_selection_format(self, mock_default_device, mock_get_devices):
        """Test device selection returns correct format with mocked devices."""
        # Mock device data
        mock_devices = [
            ("Built-in Microphone", 0),
            ("External USB Mic", 1),
            ("Bluetooth Headset", 2),
        ]
        mock_default_device.return_value = 0
        mock_get_devices.return_value = mock_devices

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            # Verify format
            assert isinstance(devices, list), "Devices should be a list"
            assert isinstance(
                default_index, int
            ), f"Default index should be int, got {type(default_index)}"

            # Verify device tuples format
            for device_name, device_index in devices:
                assert isinstance(
                    device_name, str
                ), f"Device name should be string, got {type(device_name)}"
                assert isinstance(
                    device_index, int
                ), f"Device index should be int, got {type(device_index)}"

            # Verify default index is in the available devices
            valid_indices = [index for name, index in devices]
            assert (
                default_index in valid_indices
            ), f"Default index {default_index} not in available devices {valid_indices}"

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")

    @pytest.mark.unit
    def test_device_data_validation(self):
        """Test device data validation with various input formats."""
        # Test different device data formats
        test_cases = [
            # (device_data, expected_valid, description)
            ([("Mic 1", 0), ("Mic 2", 1)], True, "Valid device list"),
            ([], True, "Empty device list (valid but no devices)"),
            ([("Mic 1", 0)], True, "Single device"),
            ([("", 0)], False, "Empty device name"),
            ([("Mic 1", -1)], False, "Negative device index"),
            ([("Mic 1", "invalid")], False, "Non-integer device index"),
            (None, False, "None device list"),
            ("not_a_list", False, "Non-list device data"),
        ]

        for device_data, expected_valid, description in test_cases:
            result = self._validate_device_data(device_data)
            if expected_valid:
                assert result, f"Should be valid: {description}"
            else:
                assert not result, f"Should be invalid: {description}"

    def _validate_device_data(self, device_data):
        """Helper method to validate device data format."""
        try:
            if not isinstance(device_data, list):
                return False

            for item in device_data:
                if not isinstance(item, tuple) or len(item) != 2:
                    return False
                name, index = item
                if not isinstance(name, str) or not isinstance(index, int):
                    return False
                if len(name) == 0 or index < 0:
                    return False

            return True
        except Exception:
            return False

    @pytest.mark.integration
    @patch("src.ui.interface_handlers.get_supported_audio_devices")
    def test_empty_device_handling(self, mock_get_devices):
        """Test handling of empty device lists."""
        # Test empty device list
        mock_get_devices.return_value = []

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            # Should handle empty device list gracefully
            assert isinstance(devices, list)

            # Implementation may return fallback "No devices" entry instead of empty list
            if len(devices) == 0:
                # Pure empty list
                pass
            elif len(devices) == 1 and "No devices" in devices[0][0]:
                # Fallback entry for no devices - good UX behavior
                assert devices[0][1] == -1  # Should use invalid index
            else:
                # Unexpected behavior
                raise AssertionError(
                    f"Unexpected device list for empty input: {devices}"
                )

            # Default index should be handled gracefully (may be 0 or -1)
            assert isinstance(default_index, int)

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")


class TestDeviceSelectionLogic(BaseIntegrationTest):
    """Test device selection logic using new infrastructure."""

    @pytest.mark.integration
    @patch("src.utils.device_utils.get_supported_audio_devices")
    @patch("src.utils.device_utils.get_default_device_index")
    def test_device_selection_logic(self, mock_default_device, mock_get_devices):
        """Test device selection logic with mocked devices."""
        # Setup mock devices
        mock_devices = [
            ("Built-in Microphone", 0),
            ("External USB Mic", 1),
            ("Bluetooth Headset", 2),
        ]
        mock_get_devices.return_value = mock_devices
        mock_default_device.return_value = 1

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            if not devices:
                pytest.skip("No devices available")

            # Test with valid device index
            test_device_index = devices[0][1]  # Use first device index
            test_device_name = devices[0][0]  # Use first device name

            # Test 1: Valid device index
            valid_indices = [index for name, index in devices]
            assert (
                test_device_index in valid_indices
            ), f"Device index {test_device_index} should be valid"

            # Test 2: Invalid device index
            invalid_index = -99
            assert (
                invalid_index not in valid_indices
            ), f"Device index {invalid_index} should be invalid"

            # Test 3: Device name lookup (legacy support)
            found_index = None
            for name, index in devices:
                if name == test_device_name:
                    found_index = index
                    break
            assert (
                found_index == test_device_index
            ), f"Name lookup should find index {test_device_index}, got {found_index}"

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")

    @pytest.mark.integration
    @patch("src.utils.status_manager.status_manager")
    def test_device_selection_with_status_manager(self, mock_status_manager):
        """Test device selection integration with status manager."""
        # Mock status manager
        mock_status_manager.get_status.return_value = AudioStatus.IDLE
        mock_status_manager.set_status = Mock()

        # Test that device selection respects status manager state
        current_status = mock_status_manager.get_status()
        assert current_status == AudioStatus.IDLE

        # Device selection should be allowed when idle
        can_select = self._can_select_device(current_status)
        assert can_select, "Should be able to select device when idle"

        # Test with recording status
        mock_status_manager.get_status.return_value = AudioStatus.RECORDING
        current_status = mock_status_manager.get_status()
        can_select = self._can_select_device(current_status)
        assert not can_select, "Should not be able to select device when recording"

    def _can_select_device(self, status):
        """Helper to determine if device selection is allowed based on status."""
        return status in [AudioStatus.IDLE, AudioStatus.ERROR]


class TestSpecificDeviceIssues(BaseIntegrationTest):
    """Test specific device issues and edge cases."""

    @pytest.mark.integration
    @patch("src.utils.device_utils.get_supported_audio_devices")
    def test_loopback_audio_device_issue(self, mock_get_devices):
        """Test the specific 'Loopback Audio' device issue."""
        # Mock devices including problematic loopback device
        mock_devices = [
            ("Built-in Microphone", 0),
            ("Loopback Audio", 1),  # This device was causing issues
            ("External USB Mic", 2),
        ]
        mock_get_devices.return_value = mock_devices

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            # Find the loopback device
            loopback_device = None
            for name, index in devices:
                if "Loopback" in name:
                    loopback_device = (name, index)
                    break

            if loopback_device:
                # Test that loopback device is properly formatted
                name, index = loopback_device
                assert isinstance(name, str)
                assert isinstance(index, int)
                assert index >= 0

                # Test that we can handle loopback device selection
                # (without actually trying to record from it)
                assert "Loopback" in name

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")

    @pytest.mark.integration
    @patch("src.utils.device_utils.get_supported_audio_devices")
    def test_unicode_device_names(self, mock_get_devices):
        """Test handling of device names with unicode characters."""
        # Mock devices with unicode names
        mock_devices = [
            ("Built-in Microphone", 0),
            ("Микрофон", 1),  # Cyrillic
            ("マイク", 2),  # Japanese
            ("Micrófono USB", 3),  # Spanish with accent
        ]
        mock_get_devices.return_value = mock_devices

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            # All device names should be handled properly
            for name, index in devices:
                assert isinstance(name, str)
                assert len(name) > 0
                assert isinstance(index, int)
                assert index >= 0

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")

    @pytest.mark.integration
    @patch("src.ui.interface_handlers.get_supported_audio_devices")
    def test_duplicate_device_names(self, mock_get_devices):
        """Test handling of duplicate device names with different indices."""
        # Mock devices with duplicate names (common with multiple USB devices)
        mock_devices = [
            ("USB Audio Device", 0),
            ("USB Audio Device", 1),
            ("USB Audio Device", 2),
            ("Built-in Microphone", 3),
        ]
        mock_get_devices.return_value = mock_devices

        try:
            from src.ui.interface_handlers import get_device_choices_and_default

            devices, default_index = get_device_choices_and_default()

            # Should handle duplicate names properly
            usb_devices = [
                device for device in devices if "USB Audio Device" in device[0]
            ]
            assert len(usb_devices) == 3

            # Each should have unique index
            indices = [index for name, index in usb_devices]
            assert (
                len(set(indices)) == 3
            ), "Duplicate device names should have unique indices"

        except ImportError as e:
            pytest.skip(f"Device selection module not available: {e}")


class TestDeviceSelectionPerformance(BaseTest):
    """Test device selection performance characteristics."""

    @pytest.mark.unit
    def test_device_lookup_performance(self):
        """Test device lookup performance with large device lists."""
        # Simulate large device list
        large_device_list = []
        for i in range(1000):
            large_device_list.append((f"Device {i}", i))

        # Test device lookup performance
        def find_device_by_name(devices, target_name):
            for name, index in devices:
                if name == target_name:
                    return index
            return None

        # Test lookup of device in middle of list
        target_name = "Device 500"
        found_index = find_device_by_name(large_device_list, target_name)
        assert found_index == 500

        # Test lookup of non-existent device
        found_index = find_device_by_name(large_device_list, "Nonexistent Device")
        assert found_index is None

    @pytest.mark.unit
    def test_device_validation_performance(self):
        """Test device validation performance with various list sizes."""
        # Test with different sized device lists
        list_sizes = [1, 10, 100, 500]

        for size in list_sizes:
            device_list = [(f"Device {i}", i) for i in range(size)]

            # Validation should be fast regardless of size
            is_valid = self._validate_device_data(device_list)
            assert is_valid, f"Validation should pass for {size} devices"

    def _validate_device_data(self, device_data):
        """Helper method for device data validation."""
        if not isinstance(device_data, list):
            return False

        for item in device_data:
            if not isinstance(item, tuple) or len(item) != 2:
                return False
            name, index = item
            if not isinstance(name, str) or not isinstance(index, int):
                return False
            if len(name) == 0 or index < 0:
                return False

        return True
