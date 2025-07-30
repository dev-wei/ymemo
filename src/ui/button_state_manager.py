"""Centralized button state management for the UI interface."""

import logging
from typing import Dict, Any
from dataclasses import dataclass

import gradio as gr

from ..utils.status_manager import AudioStatus
from .interface_constants import BUTTON_TEXT

logger = logging.getLogger(__name__)


@dataclass
class ButtonConfig:
    """Configuration for a single button."""
    text: str
    variant: str
    interactive: bool
    visible: bool = True


class ButtonStateManager:
    """Manages button states based on application status."""
    
    def __init__(self):
        """Initialize the button state manager."""
        self._status_button_map = self._build_status_button_map()
    
    def _build_status_button_map(self) -> Dict[AudioStatus, Dict[str, ButtonConfig]]:
        """Build the mapping from audio status to button configurations."""
        return {
            # Idle/Ready states - ready to start recording
            AudioStatus.IDLE: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="primary",
                    interactive=True
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            AudioStatus.READY: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="primary",
                    interactive=True
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            # Starting up states
            AudioStatus.INITIALIZING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["starting"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            AudioStatus.CONNECTING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["starting"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            # Active recording states
            AudioStatus.RECORDING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="primary",
                    interactive=True
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            AudioStatus.TRANSCRIBING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="primary",
                    interactive=True
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            AudioStatus.TRANSCRIPTION_DISCONNECTED: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="primary",
                    interactive=True
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            AudioStatus.RECONNECTING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="primary",
                    interactive=True
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            # Stopping state
            AudioStatus.STOPPING: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stopping"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            },
            
            # Stopped state - ready to save
            AudioStatus.STOPPED: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=True
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="primary",
                    interactive=True
                )
            },
            
            # Error state
            AudioStatus.ERROR: {
                "start_btn": ButtonConfig(
                    text=BUTTON_TEXT["start_recording"],
                    variant="secondary",
                    interactive=True
                ),
                "stop_btn": ButtonConfig(
                    text=BUTTON_TEXT["stop_recording"],
                    variant="secondary",
                    interactive=False
                ),
                "save_btn": ButtonConfig(
                    text=BUTTON_TEXT["save_meeting"],
                    variant="secondary",
                    interactive=False
                )
            }
        }
    
    def get_button_configs(self, status: AudioStatus) -> Dict[str, ButtonConfig]:
        """Get button configurations for the given status.
        
        Args:
            status: Current AudioStatus
            
        Returns:
            Dictionary mapping button names to their configurations
        """
        if status in self._status_button_map:
            return self._status_button_map[status].copy()
        
        # Fallback to IDLE state for unknown statuses
        logger.warning(f"Unknown audio status: {status}, falling back to IDLE")
        return self._status_button_map[AudioStatus.IDLE].copy()
    
    def get_gradio_updates(self, status: AudioStatus) -> Dict[str, gr.update]:
        """Get Gradio update objects for all buttons based on status.
        
        Args:
            status: Current AudioStatus
            
        Returns:
            Dictionary mapping button names to gr.update objects
        """
        configs = self.get_button_configs(status)
        
        updates = {}
        for button_name, config in configs.items():
            updates[button_name] = gr.update(
                value=config.text,
                variant=config.variant,
                interactive=config.interactive,
                visible=config.visible
            )
        
        return updates
    
    def get_button_update_tuple(self, status: AudioStatus) -> tuple:
        """Get button updates as a tuple for Gradio output.
        
        Args:
            status: Current AudioStatus
            
        Returns:
            Tuple of (start_btn_update, stop_btn_update, save_btn_update)
        """
        updates = self.get_gradio_updates(status)
        return (
            updates["start_btn"],
            updates["stop_btn"], 
            updates["save_btn"]
        )
    
    def is_button_interactive(self, status: AudioStatus, button_name: str) -> bool:
        """Check if a specific button should be interactive for given status.
        
        Args:
            status: Current AudioStatus
            button_name: Name of the button to check
            
        Returns:
            True if button should be interactive, False otherwise
        """
        configs = self.get_button_configs(status)
        return configs.get(button_name, ButtonConfig("", "", False)).interactive
    
    def get_safe_fallback_updates(self) -> Dict[str, gr.update]:
        """Get safe fallback button updates for error scenarios.
        
        Returns:
            Dictionary of safe button updates
        """
        return {
            "start_btn": gr.update(
                value=BUTTON_TEXT["start_recording"],
                variant="primary",
                interactive=True
            ),
            "stop_btn": gr.update(
                value=BUTTON_TEXT["stop_recording"],
                variant="secondary",
                interactive=False
            ),
            "save_btn": gr.update(
                value=BUTTON_TEXT["save_meeting"],
                variant="secondary",
                interactive=False
            )
        }


# Global instance for consistent state management
button_state_manager = ButtonStateManager()