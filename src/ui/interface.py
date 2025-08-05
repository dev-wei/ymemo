"""Main interface creation for the Voice Meeting App."""

import logging

import gradio as gr

from src.config.audio_config import (
    get_audio_quality_choices,
    get_current_audio_quality_info,
)
from src.config.language_config import get_default_language, get_language_choices
from src.config.provider_config import (
    get_current_provider_from_env,
    get_display_name_from_key,
    get_provider_choices,
)
from src.managers.session_manager import get_audio_session
from src.utils.status_manager import status_manager

from .audio_quality_handlers import (
    get_current_audio_quality_info_html,
    handle_audio_quality_change,
)
from .button_state_manager import button_state_manager
from .interface_constants import (
    AVAILABLE_THEMES,
    BUTTON_TEXT,
    COPY_CONFIG,
    DEFAULT_THEME,
    DEFAULT_VALUES,
    FORM_LABELS,
    PLACEHOLDER_TEXT,
    TABLE_HEADERS,
    UI_DIMENSIONS,
    UI_TEXT,
)
from .interface_dialog_handlers import combined_update
from .interface_handlers import (
    clear_dialog,
    get_device_choices_and_default,
    handle_copy_event,
    immediate_transcription_update,
    refresh_devices,
)
from .interface_styles import APP_CSS, APP_JS
from .interface_utils import load_meetings_data
from .language_handlers import get_current_language_info, handle_language_change
from .meeting_handlers import (
    delete_meeting_by_id_input,
    submit_new_meeting,
)
from .persona_handlers import (
    create_persona_from_id,
    delete_persona_by_id_input,
    get_persona_choices,
    handle_speaker_persona_change,
    load_persona_by_id,
    refresh_persona_dropdowns,
    refresh_personas,
    submit_new_persona,
)
from .provider_handlers import get_current_provider_info, handle_provider_change
from .recording_handlers import (
    start_recording,
    stop_recording,
)

logger = logging.getLogger(__name__)


# Utility functions moved to interface_utils.py


def update_button_states():
    """Update all button states based on current recording status.

    Returns:
        Tuple of button updates for (start_btn, stop_btn, save_btn)
    """
    try:
        current_status = status_manager.current_status
        return button_state_manager.get_button_update_tuple(current_status)
    except Exception as e:
        logger.error(f"Error updating button states: {e}")
        # Return safe defaults using the manager
        safe_updates = button_state_manager.get_safe_fallback_updates()
        return (
            safe_updates["start_btn"],
            safe_updates["stop_btn"],
            safe_updates["save_btn"],
        )


# Use themes from constants
THEMES = AVAILABLE_THEMES


def create_header():
    """Create the header section of the interface."""
    with gr.Row():
        gr.Markdown(
            f"""
            # {UI_TEXT["app_title"]}
            {UI_TEXT["app_subtitle"]}
            """,
            elem_classes=["header-text"],
        )


def create_sidebar():
    """Create collapsible sidebar with Settings and Performance sections."""
    with gr.Sidebar(
        open=False, position="left", width=UI_DIMENSIONS["sidebar_width"]
    ) as sidebar:
        # Settings Section
        gr.Markdown(UI_TEXT["sidebar_settings_title"])
        with gr.Group():
            # Theme selector
            theme_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_THEMES.keys()),
                value=DEFAULT_THEME,
                label=FORM_LABELS["theme_selector"],
                interactive=True,
            )
            # Audio quality selection - now interactive with quality presets
            audio_quality_dropdown = gr.Dropdown(
                label=FORM_LABELS["audio_quality"],
                choices=get_audio_quality_choices(),
                value=get_current_audio_quality_info()['quality'],
                interactive=True,
            )
            # Audio quality information display
            audio_quality_info_display = gr.HTML(
                value=get_current_audio_quality_info_html(),
                label="Audio Quality Details",
            )
            # Language selection - now interactive with full language support
            language_dropdown = gr.Dropdown(
                label=FORM_LABELS["language_selection"],
                choices=get_language_choices(),
                value=get_default_language(),
                interactive=True,
            )
            # Language information display
            language_info_display = gr.HTML(
                value=get_current_language_info(get_default_language()),
                label=FORM_LABELS["language_info"],
            )
            # Transcription provider dropdown - now interactive
            provider_dropdown = gr.Dropdown(
                label=FORM_LABELS["transcription_provider"],
                choices=get_provider_choices(),
                value=get_display_name_from_key(get_current_provider_from_env()),
                interactive=True,
            )
            # Provider information display
            provider_info_display = gr.HTML(
                value=get_current_provider_info(
                    get_display_name_from_key(get_current_provider_from_env())
                ),
                label=FORM_LABELS["provider_details"],
            )

        # Performance Section
        gr.Markdown(UI_TEXT["sidebar_performance_title"])
        with gr.Group():
            # Connection status
            connection_status = gr.HTML(
                value=DEFAULT_VALUES["connection_status"],
                label=FORM_LABELS["connection_status"],
            )
            # Session duration
            session_duration_display = gr.Textbox(
                label=FORM_LABELS["session_duration"],
                value=DEFAULT_VALUES["duration_display"],
                interactive=False,
            )
            # Audio level placeholder
            audio_level_display = gr.Textbox(
                label=FORM_LABELS["audio_level"],
                value=DEFAULT_VALUES["audio_level"],
                interactive=False,
            )
            # Memory usage placeholder
            memory_usage_display = gr.Textbox(
                label=FORM_LABELS["memory_usage"],
                value=DEFAULT_VALUES["memory_usage"],
                interactive=False,
            )

    return (
        sidebar,
        theme_dropdown,
        audio_quality_dropdown,
        audio_quality_info_display,
        language_dropdown,
        language_info_display,
        provider_dropdown,
        provider_info_display,
        connection_status,
        session_duration_display,
        audio_level_display,
        memory_usage_display,
    )


def create_meeting_list():
    """Create the meeting list panel with delete functionality."""
    with gr.Column(elem_classes=["meeting-list-container"]):
        gr.Markdown(UI_TEXT["meeting_list_title"])

        # Meeting list dataframe in its own container
        with gr.Column(elem_classes=["meeting-panel"]):
            meeting_list = gr.Dataframe(
                headers=TABLE_HEADERS["meeting_list"],
                datatype=["number", "str", "str", "str", "str"],  # number for ID
                value=load_meetings_data(),
                interactive=False,  # Make completely readonly
                show_search="search",  # Enable search functionality
                show_fullscreen_button=True,  # Allow fullscreen viewing
                show_copy_button=True,  # Enable copying data
                show_row_numbers=True,  # Show row numbers for additional clarity
                wrap=True,  # Enable text wrapping if needed
            )

        # Delete section - positioned below the table, outside the fixed-height panel
        with gr.Column(elem_classes=["delete-controls-section"]):
            with gr.Row():
                meeting_id_input = gr.Textbox(
                    label="Meeting ID to Delete",
                    placeholder="Enter meeting ID (e.g., 1, 2, 3)",
                    scale=3,
                )
                delete_meeting_btn = gr.Button(
                    "🗑️ Delete Meeting", variant="stop", scale=1, size="sm"
                )

            # Status message for delete operations
            delete_status = gr.HTML(
                value="💡 Enter a meeting ID from the table above and click Delete",
                visible=True,
            )

        return meeting_list, meeting_id_input, delete_meeting_btn, delete_status


def create_dialog_panel():
    """Create the dialog panel with meeting fields and chatbot."""
    with gr.Column(scale=4, elem_classes=["dialog-panel"]):
        gr.Markdown(UI_TEXT["live_dialog_title"])

        # Meeting fields
        with gr.Row():
            meeting_name_field = gr.Textbox(
                label=FORM_LABELS["meeting_name"],
                placeholder=PLACEHOLDER_TEXT["meeting_name"],
                value="",
            )
            duration_field = gr.Textbox(
                label=FORM_LABELS["duration"],
                value=DEFAULT_VALUES["duration_display"],
                interactive=False,
            )

        dialog_output = gr.Chatbot(
            value=[],  # Start with empty dialog
            type="messages",
            show_label=False,
            placeholder=PLACEHOLDER_TEXT["transcription_dialog"],
            height=UI_DIMENSIONS["dialog_height"],  # Set chatbot height
            show_copy_button=True,  # Enable individual message copy buttons
            show_copy_all_button=True,  # Enable copy all messages button
            watermark=COPY_CONFIG["watermark"],  # Add watermark to copied content
        )

        return meeting_name_field, duration_field, dialog_output


# get_device_choices_and_default moved to interface_handlers.py


def create_controls():
    """Create the audio controls panel."""

    with gr.Column(scale=2, elem_classes=["control-panel"]):
        # Row 1: Audio Controls
        with gr.Row():
            with gr.Column():
                gr.Markdown(UI_TEXT["audio_controls_title"])

                # Audio device selection
                device_choices, initial_device_index = get_device_choices_and_default()

                device_dropdown = gr.Dropdown(
                    label=FORM_LABELS["audio_device"],
                    choices=device_choices,
                    value=initial_device_index,
                    interactive=True,
                    allow_custom_value=False,  # Disable custom values to prevent invalid indices
                )

                # Device refresh button
                refresh_btn = gr.Button(
                    BUTTON_TEXT["refresh_devices"], size="sm", variant="secondary"
                )

                # Recording status
                status_text = gr.Textbox(
                    label=FORM_LABELS["status"],
                    value=status_manager.get_status_message(),
                    interactive=False,
                )

                # Control buttons - Initialize with proper states using ButtonStateManager
                current_status = status_manager.current_status
                logger.info(f"🔍 Current status: {current_status}")
                button_configs = button_state_manager.get_button_configs(current_status)
                logger.info(
                    f"🔍 Start button interactive: {button_configs['start_btn'].interactive}"
                )

                with gr.Row():
                    start_btn = gr.Button(
                        button_configs["start_btn"].text,
                        variant=button_configs["start_btn"].variant,
                        interactive=button_configs["start_btn"].interactive,
                    )
                    stop_btn = gr.Button(
                        button_configs["stop_btn"].text,
                        variant=button_configs["stop_btn"].variant,
                        interactive=button_configs["stop_btn"].interactive,
                    )

                # Save meeting button
                save_meeting_btn = gr.Button(
                    button_configs["save_btn"].text,
                    variant=button_configs["save_btn"].variant,
                    interactive=button_configs["save_btn"].interactive,
                )

        # Row 2: Persona Controls
        with gr.Row():
            with gr.Column():
                gr.Markdown(UI_TEXT["persona_title"])

                # Get persona choices for both dropdowns
                persona_choices = get_persona_choices()
                default_value = (
                    persona_choices[0][1] if persona_choices else ""
                )  # Use first choice value

                # Use a Group container to ensure proper display
                with gr.Group():
                    with gr.Row():
                        speaker_a_persona = gr.Dropdown(
                            label=FORM_LABELS["speaker_a_persona"],
                            choices=persona_choices,
                            value=default_value,
                            interactive=True,
                            show_label=True,
                            container=True,
                        )
                        speaker_b_persona = gr.Dropdown(
                            label=FORM_LABELS["speaker_b_persona"],
                            choices=persona_choices,
                            value=default_value,
                            interactive=True,
                            show_label=True,
                            container=True,
                        )

        return (
            device_dropdown,
            refresh_btn,
            status_text,
            start_btn,
            stop_btn,
            save_meeting_btn,
            speaker_a_persona,
            speaker_b_persona,
        )


def create_interface(theme_name: str = DEFAULT_THEME) -> gr.Blocks:
    """Create the main Gradio interface.

    Args:
        theme_name: Name of the theme to use

    Returns:
        Gradio Blocks interface
    """
    # Get theme
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])

    # Initialize audio devices - function moved to create_controls()
    # Use styles from separate file
    css = APP_CSS
    js_func = APP_JS

    with gr.Blocks(
        title="Voice Meeting App",
        theme=theme,
        css=css,
        js=js_func,
    ) as demo:
        # Header
        create_header()

        # Sidebar - Hidden by default, collapsible
        (
            _,  # sidebar
            _,  # sidebar_theme_dropdown
            sidebar_audio_quality_dropdown,
            sidebar_audio_quality_info,
            sidebar_language_dropdown,
            sidebar_language_info,
            sidebar_provider_dropdown,
            sidebar_provider_info,
            _,  # sidebar_connection_status
            _,  # sidebar_session_duration
            _,  # sidebar_audio_level
            _,  # sidebar_memory_usage
        ) = create_sidebar()

        # Tabbed interface structure
        with gr.Tab("Meeting Transcription", id="meeting_tab"):
            # Responsive layout structure
            # Desktop: [Sidebar (hidden)] [Meeting List] [Live Dialog] [Audio Controls]
            # Mobile: [Sidebar (hidden)] [Meeting List - Full Width] then [Live Dialog] [Audio Controls]

            # Meeting List - Full width on mobile, partial on desktop
            (
                meeting_list,
                meeting_id_input,
                delete_meeting_btn,
                delete_status,
            ) = create_meeting_list()

            # Dialog and Controls - Side by side on all screens, but different proportions
            with gr.Row(elem_classes=["main-content-row"]):
                # Center panel - Live Dialog
                meeting_name_field, duration_field, dialog_output = (
                    create_dialog_panel()
                )

                # Right panel - Audio Controls
                (
                    device_dropdown,
                    refresh_btn,
                    status_text,
                    start_btn,
                    stop_btn,
                    save_meeting_btn,
                    speaker_a_persona,
                    speaker_b_persona,
                ) = create_controls()

        with gr.Tab("Persona", id="persona_tab"):
            # Persona feature placeholder
            gr.Markdown("## Persona Feature")
            gr.Markdown(
                "Create and manage personalized transcription profiles and settings."
            )

            # Persona list section (similar to meeting list)
            with gr.Column(elem_classes=["persona-list-container"]):
                gr.Markdown("### Your Personas")

                # Persona list dataframe
                with gr.Column(elem_classes=["persona-panel"]):
                    persona_list = gr.Dataframe(
                        headers=["ID", "Name", "Description", "Created", "Updated"],
                        datatype=["number", "str", "str", "str", "str"],
                        value=[],  # Start with empty data to avoid loading errors
                        interactive=False,
                        show_search=False,  # Disable search to avoid potential issues
                        show_fullscreen_button=True,
                        show_copy_button=True,
                        show_row_numbers=True,
                        wrap=True,
                    )

                # Operation section for personas
                with gr.Column(elem_classes=["delete-controls-section"]):
                    with gr.Row():
                        persona_id_input = gr.Textbox(
                            label="Persona ID to Operate",
                            placeholder="Enter persona ID (e.g., 1, 2, 3)",
                            scale=2,
                            show_label=True,
                            container=True,
                        )
                        create_persona_btn = gr.Button(
                            "➕ Create New Persona",
                            variant="primary",
                            scale=1,
                            size="sm",
                        )
                        load_persona_btn = gr.Button(
                            "📋 Load Persona", variant="secondary", scale=1, size="sm"
                        )
                        delete_persona_btn = gr.Button(
                            "🗑️ Delete Persona", variant="stop", scale=1, size="sm"
                        )

                    # Help text for persona operations
                    gr.Markdown(
                        "💡 Enter a persona ID from the table above and choose an operation"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create New Persona")
                    persona_id_field = gr.Textbox(
                        label="Persona ID",
                        placeholder="Auto-generated for new personas",
                        value="",
                        interactive=False,  # Make it readonly
                        show_label=True,
                        container=True,
                    )
                    persona_name = gr.Textbox(
                        label="Persona Name",
                        placeholder="Enter persona name (e.g., 'Professional Meeting', 'Creative Brainstorm')",
                        show_label=True,
                        container=True,
                    )
                    persona_description = gr.Textbox(
                        label="Description",
                        placeholder="Describe this persona's purpose and use case",
                        lines=3,
                        show_label=True,
                        container=True,
                    )

            with gr.Row():
                save_persona_btn = gr.Button("Save Persona", variant="primary")
                refresh_personas_btn = gr.Button("Refresh List", variant="secondary")

            gr.Markdown("Ready to create your first persona profile.")

        # Get audio session manager
        audio_session = get_audio_session()

        # State management for real-time updates
        dialog_state = gr.State([])

        # Dialog state update function moved to interface_dialog_handlers.py

        # Event handlers moved to interface_handlers.py

        # Register callback with session manager
        audio_session.add_transcription_callback(immediate_transcription_update)

        # Timer for dialog, duration, and button state updates
        timer = gr.Timer(
            value=UI_DIMENSIONS["timer_interval"]
        )  # Check for updates every 500ms

        # Wire up event handlers
        refresh_btn.click(fn=refresh_devices, outputs=[device_dropdown, status_text])

        # Language selection handler
        sidebar_language_dropdown.change(
            fn=lambda lang: (
                handle_language_change(lang)[0],
                get_current_language_info(lang),
            ),
            inputs=[sidebar_language_dropdown],
            outputs=[status_text, sidebar_language_info],
        )

        # Provider selection handler
        sidebar_provider_dropdown.change(
            fn=lambda provider, current_lang: (
                handle_provider_change(provider, current_lang)[0],
                get_current_provider_info(provider),
            ),
            inputs=[sidebar_provider_dropdown, sidebar_language_dropdown],
            outputs=[status_text, sidebar_provider_info],
        )

        # Audio quality selection handler
        sidebar_audio_quality_dropdown.change(
            fn=lambda quality: handle_audio_quality_change(quality),
            inputs=[sidebar_audio_quality_dropdown],
            outputs=[status_text, sidebar_audio_quality_info],
        )

        # Speaker persona selection handlers
        speaker_a_persona.change(
            fn=lambda persona_id: handle_speaker_persona_change(persona_id, "A"),
            inputs=[speaker_a_persona],
            outputs=[status_text],
        )

        speaker_b_persona.change(
            fn=lambda persona_id: handle_speaker_persona_change(persona_id, "B"),
            inputs=[speaker_b_persona],
            outputs=[status_text],
        )

        start_btn.click(
            fn=start_recording,
            inputs=[device_dropdown, dialog_state],
            outputs=[
                status_text,
                dialog_state,
                dialog_output,
                start_btn,
                stop_btn,
                save_meeting_btn,
            ],
        )

        stop_btn.click(
            fn=stop_recording,
            outputs=[status_text, start_btn, stop_btn, save_meeting_btn],
        )

        # Combined update function moved to interface_dialog_handlers.py

        # Timer for dialog, duration, and button state updates
        timer.tick(
            fn=combined_update,
            outputs=[
                dialog_state,
                dialog_output,
                duration_field,
                start_btn,
                stop_btn,
                save_meeting_btn,
            ],
        )

        # Clear dialog functionality - wire up chatbot's built-in clear event
        dialog_output.clear(
            fn=clear_dialog,
            outputs=[dialog_state, dialog_output],
            queue=False,  # Immediate response for clearing
        )

        # Copy event handler - wire up chatbot's copy event for analytics
        dialog_output.copy(
            fn=handle_copy_event,
            queue=False,  # Immediate response for copying
        )

        # Direct save functionality (replaces old sliding panel system)
        save_meeting_btn.click(
            fn=submit_new_meeting,
            inputs=[meeting_name_field, duration_field, dialog_output],
            outputs=[meeting_list],
        )

        # Simple ID-based delete functionality
        delete_meeting_btn.click(
            fn=delete_meeting_by_id_input,
            inputs=[meeting_id_input],
            outputs=[meeting_list, delete_status],
        ).then(
            # Clear the input field after successful operation
            fn=lambda: "",
            outputs=[meeting_id_input],
        )

        # Persona management event handlers
        save_persona_btn.click(
            fn=submit_new_persona,
            inputs=[persona_id_field, persona_name, persona_description],
            outputs=[persona_list],
        ).then(
            # Clear the form after successful submission
            fn=lambda: ("", "", ""),
            outputs=[persona_id_field, persona_name, persona_description],
        ).then(
            # Refresh persona dropdowns
            fn=refresh_persona_dropdowns,
            outputs=[speaker_a_persona, speaker_b_persona],
        )

        # Persona deletion handler
        delete_persona_btn.click(
            fn=delete_persona_by_id_input,
            inputs=[persona_id_input],
            outputs=[persona_list],
        ).then(
            # Clear the input field after operation
            fn=lambda: "",
            outputs=[persona_id_input],
        ).then(
            # Refresh persona dropdowns
            fn=refresh_persona_dropdowns,
            outputs=[speaker_a_persona, speaker_b_persona],
        )

        # Create new persona handler
        create_persona_btn.click(
            fn=create_persona_from_id,
            inputs=[persona_id_input],
            outputs=[persona_list, persona_id_field, persona_name, persona_description],
        ).then(
            # Clear the input field after operation
            fn=lambda: "",
            outputs=[persona_id_input],
        )

        # Load persona handler
        load_persona_btn.click(
            fn=load_persona_by_id,
            inputs=[persona_id_input],
            outputs=[persona_list, persona_id_field, persona_name, persona_description],
        )

        # Persona refresh handler
        refresh_personas_btn.click(
            fn=refresh_personas,
            outputs=[persona_list],
        ).then(
            # Refresh persona dropdowns
            fn=refresh_persona_dropdowns,
            outputs=[speaker_a_persona, speaker_b_persona],
        )

        # Load personas on interface start
        demo.load(
            fn=refresh_personas,
            outputs=[persona_list],
        ).then(
            # Initialize persona dropdowns on load
            fn=refresh_persona_dropdowns,
            outputs=[speaker_a_persona, speaker_b_persona],
        )

        # Note: Removed automatic button updates to prevent interference with clicks
        # Buttons are updated manually in the event handlers when needed

    return demo
