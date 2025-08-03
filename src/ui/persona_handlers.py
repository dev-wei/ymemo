"""Persona management handlers for the UI interface."""

import logging
from typing import List, Tuple

import gradio as gr

from ..managers.persona_repository import (
    PersonaRepositoryError,
    create_persona,
    delete_persona_by_id,
    get_all_personas,
)

logger = logging.getLogger(__name__)


def load_personas_data() -> List[List]:
    """Load personas data for display in the dataframe.

    Returns:
        List of persona rows for display, each containing [ID, Name, Description, Created, Updated]
    """
    try:
        logger.info("🔍 Loading personas data for UI display")
        personas = get_all_personas()

        if not personas:
            logger.info("📝 No personas found")
            return []

        # Convert personas to display format
        display_data = []
        for persona in personas:
            try:
                display_row = persona.to_display_row()
                display_data.append(display_row)
                logger.debug(f"✅ Added persona to display: {persona.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to format persona {persona.id}: {e}")
                continue

        logger.info(f"✅ Loaded {len(display_data)} personas for display")
        return display_data

    except Exception as e:
        logger.error(f"❌ Failed to load personas data: {e}")
        return []


def submit_new_persona(name: str, description: str) -> Tuple[gr.HTML, gr.Dataframe]:
    """Handle persona creation form submission.

    Args:
        name: Persona name from the form
        description: Persona description from the form

    Returns:
        Tuple of (status_message_update, dataframe_update)
    """
    try:
        logger.info(f"💾 Processing new persona submission: {name}")

        # Validate input
        if not name or not name.strip():
            error_msg = "❌ Persona name cannot be empty"
            logger.warning(error_msg)
            return (
                gr.HTML(value=error_msg, visible=True),
                gr.Dataframe(value=load_personas_data()),
            )

        # Create the persona
        persona = create_persona(
            name=name.strip(), description=description.strip() if description else None
        )

        # Success message
        success_msg = f"✅ Successfully created persona: {persona.name}"
        logger.info(success_msg)

        # Return updated UI components
        return (
            gr.HTML(value=success_msg, visible=True),
            gr.Dataframe(value=load_personas_data()),
        )

    except PersonaRepositoryError as e:
        error_msg = f"❌ Database error: {str(e)}"
        logger.error(error_msg)
        return (
            gr.HTML(value=error_msg, visible=True),
            gr.Dataframe(value=load_personas_data()),
        )
    except Exception as e:
        error_msg = f"❌ Unexpected error creating persona: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (
            gr.HTML(value=error_msg, visible=True),
            gr.Dataframe(value=load_personas_data()),
        )


def delete_persona_by_id_input(persona_id_str: str) -> Tuple[gr.Dataframe, gr.HTML]:
    """Handle persona deletion by ID from input field.

    Args:
        persona_id_str: Persona ID as string from input field

    Returns:
        Tuple of (updated_dataframe, status_message)
    """
    try:
        logger.info(f"🗑️ Processing persona deletion request: {persona_id_str}")

        # Validate input
        if not persona_id_str or not persona_id_str.strip():
            error_msg = "❌ Please enter a persona ID"
            logger.warning(error_msg)
            return (gr.Dataframe(value=load_personas_data()), gr.HTML(value=error_msg))

        # Parse persona ID
        try:
            persona_id = int(persona_id_str.strip())
        except ValueError:
            error_msg = f"❌ Invalid persona ID format: {persona_id_str}"
            logger.warning(error_msg)
            return (gr.Dataframe(value=load_personas_data()), gr.HTML(value=error_msg))

        # Attempt deletion
        success = delete_persona_by_id(persona_id)

        if success:
            success_msg = f"✅ Successfully deleted persona ID: {persona_id}"
            logger.info(success_msg)
            status_message = gr.HTML(value=success_msg)
        else:
            error_msg = f"❌ Persona with ID {persona_id} not found"
            logger.warning(error_msg)
            status_message = gr.HTML(value=error_msg)

        # Return updated dataframe and status
        return (gr.Dataframe(value=load_personas_data()), status_message)

    except PersonaRepositoryError as e:
        error_msg = f"❌ Database error: {str(e)}"
        logger.error(error_msg)
        return (gr.Dataframe(value=load_personas_data()), gr.HTML(value=error_msg))
    except Exception as e:
        error_msg = f"❌ Unexpected error deleting persona: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (gr.Dataframe(value=load_personas_data()), gr.HTML(value=error_msg))


def refresh_personas() -> gr.Dataframe:
    """Refresh the personas list.

    Returns:
        Updated dataframe with current personas data
    """
    try:
        logger.info("🔄 Refreshing personas list")
        return gr.Dataframe(value=load_personas_data())
    except Exception as e:
        logger.error(f"❌ Failed to refresh personas: {e}")
        return gr.Dataframe(value=[])
