"""Persona management handlers for the UI interface."""

import logging
from typing import List, Tuple

import gradio as gr

from ..managers.persona_repository import (
    PersonaRepositoryError,
    create_persona,
    delete_persona_by_id,
    get_all_personas,
    update_persona,
)

logger = logging.getLogger(__name__)


def validate_persona_id_exists(persona_id: int) -> bool:
    """Check if persona ID exists in current database.

    Args:
        persona_id: The persona ID to validate

    Returns:
        bool: True if persona exists, False otherwise
    """
    try:
        from ..managers.persona_repository import get_persona_by_id

        persona = get_persona_by_id(persona_id)
        return persona is not None
    except Exception as e:
        logger.warning(f"⚠️ Error validating persona ID {persona_id}: {e}")
        return False


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


def submit_new_persona(persona_id: str, name: str, description: str) -> gr.Dataframe:
    """Handle persona creation or update form submission.

    Args:
        persona_id: Persona ID from the form (empty for new, populated for update)
        name: Persona name from the form
        description: Persona description from the form

    Returns:
        Updated dataframe
    """
    try:
        logger.info(
            f"💾 Processing persona submission: ID='{persona_id}', name='{name}'"
        )

        # Validate input
        if not name or not name.strip():
            logger.warning("❌ Empty persona name in submission")
            gr.Warning("Persona name cannot be empty ⚠️", duration=3)
            return gr.Dataframe(value=load_personas_data())

        # Determine if this is create or update based on persona_id
        is_update = persona_id and persona_id.strip()

        if is_update:
            # Update existing persona
            try:
                persona_id_int = int(persona_id.strip())
                logger.info(f"🔄 Updating existing persona ID: {persona_id_int}")

                updated_persona = update_persona(
                    persona_id=persona_id_int,
                    name=name.strip(),
                    description=description.strip() if description else None,
                )

                if updated_persona:
                    logger.info(
                        f"✅ Successfully updated persona: {updated_persona.name}"
                    )
                    gr.Info(
                        f"Persona '{updated_persona.name}' updated successfully! 🔄",
                        duration=3,
                    )
                else:
                    logger.warning(f"❌ Failed to update persona ID {persona_id_int}")
                    gr.Warning(
                        f"Persona with ID {persona_id_int} not found for update ⚠️",
                        duration=4,
                    )

            except ValueError:
                logger.warning(f"❌ Invalid persona ID format for update: {persona_id}")
                gr.Warning(f"Invalid persona ID format: {persona_id} ⚠️", duration=3)
        else:
            # Create new persona
            logger.info("➕ Creating new persona")
            persona = create_persona(
                name=name.strip(),
                description=description.strip() if description else None,
            )
            logger.info(f"✅ Successfully created persona: {persona.name}")
            gr.Info(f"Persona '{persona.name}' created successfully! ➕", duration=3)

        # Return updated UI components
        return gr.Dataframe(value=load_personas_data())

    except PersonaRepositoryError as e:
        logger.error(f"❌ Database error: {str(e)}")
        gr.Warning(f"Database error: {str(e)} 💥!", duration=5)
        return gr.Dataframe(value=load_personas_data())
    except Exception as e:
        logger.error(f"❌ Unexpected error processing persona: {str(e)}", exc_info=True)
        gr.Warning(f"Unexpected error processing persona: {str(e)} 💥!", duration=5)
        return gr.Dataframe(value=load_personas_data())


def delete_persona_by_id_input(persona_id_str: str) -> gr.Dataframe:
    """Handle persona deletion by ID from input field.

    Args:
        persona_id_str: Persona ID as string from input field

    Returns:
        Updated dataframe
    """
    try:
        logger.info(f"🗑️ Processing persona deletion request: {persona_id_str}")

        # Validate input
        if not persona_id_str or not persona_id_str.strip():
            logger.warning("❌ Empty persona ID input for delete operation")
            gr.Warning("Please enter a persona ID to delete ⚠️", duration=3)
            return gr.Dataframe(value=load_personas_data())

        # Parse persona ID
        try:
            persona_id = int(persona_id_str.strip())
        except ValueError:
            logger.warning(f"❌ Invalid persona ID format: {persona_id_str}")
            gr.Warning(f"Invalid persona ID format: {persona_id_str} ⚠️", duration=3)
            return gr.Dataframe(value=load_personas_data())

        # Validate persona ID exists before attempting deletion
        if not validate_persona_id_exists(persona_id):
            logger.warning(f"❌ Persona ID {persona_id} not found for deletion")
            gr.Warning(
                f"Persona with ID {persona_id} not found in the current list ⚠️",
                duration=4,
            )
            return gr.Dataframe(value=load_personas_data())

        # Attempt deletion
        success = delete_persona_by_id(persona_id)

        if success:
            logger.info(f"✅ Successfully deleted persona ID: {persona_id}")
        else:
            logger.warning(f"❌ Failed to delete persona with ID {persona_id}")
            gr.Warning(f"Failed to delete persona with ID {persona_id} ⚠️", duration=4)
            return gr.Dataframe(value=load_personas_data())

        # Return updated dataframe
        return gr.Dataframe(value=load_personas_data())

    except PersonaRepositoryError as e:
        logger.error(f"❌ Database error: {str(e)}")
        gr.Warning(f"Database error: {str(e)} 💥!", duration=5)
        return gr.Dataframe(value=load_personas_data())
    except Exception as e:
        logger.error(f"❌ Unexpected error deleting persona: {str(e)}", exc_info=True)
        gr.Warning(f"Unexpected error deleting persona: {str(e)} 💥!", duration=5)
        return gr.Dataframe(value=load_personas_data())


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


def create_persona_from_id(persona_id_str: str) -> Tuple[gr.Dataframe, str, str, str]:
    """Clear form fields to prepare for creating a new persona.

    Args:
        persona_id_str: Persona ID as string from input field (unused but kept for compatibility)

    Returns:
        Tuple of (updated_dataframe, empty_id, empty_name, empty_description)
    """
    try:
        logger.info("➕ Processing create new persona request - clearing form fields")
        logger.info("✅ Form fields cleared for new persona creation")

        # Return updated UI components with cleared form fields
        return (
            gr.Dataframe(value=load_personas_data()),  # Refresh the list
            "",  # Clear persona ID field (empty for new)
            "",  # Clear persona name field
            "",  # Clear persona description field
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error clearing form: {str(e)}", exc_info=True)
        raise gr.Error(f"Unexpected error clearing form: {str(e)} 💥!", duration=5)


def load_persona_by_id(persona_id_str: str) -> Tuple[gr.Dataframe, str, str, str]:
    """Load and populate form fields with a specific persona by ID.

    Args:
        persona_id_str: Persona ID as string from input field

    Returns:
        Tuple of (updated_dataframe, persona_id, persona_name, persona_description)
    """
    try:
        logger.info(f"📋 Processing load persona request: {persona_id_str}")

        # Validate input
        if not persona_id_str or not persona_id_str.strip():
            logger.warning("❌ Empty persona ID input for load operation")
            gr.Warning("Please enter a persona ID to load ⚠️", duration=3)
            return (gr.Dataframe(value=load_personas_data()), "", "", "")

        # Parse persona ID
        try:
            persona_id = int(persona_id_str.strip())
        except ValueError:
            logger.warning(f"❌ Invalid persona ID format: {persona_id_str}")
            gr.Warning(f"Invalid persona ID format: {persona_id_str} ⚠️", duration=3)
            return (gr.Dataframe(value=load_personas_data()), "", "", "")

        # Validate persona ID exists
        if not validate_persona_id_exists(persona_id):
            logger.warning(f"❌ Persona ID {persona_id} not found")
            gr.Warning(
                f"Persona with ID {persona_id} not found in the current list ⚠️",
                duration=4,
            )
            return (gr.Dataframe(value=load_personas_data()), "", "", "")

        # Get the persona
        from ..managers.persona_repository import get_persona_by_id

        persona = get_persona_by_id(persona_id)

        if persona:
            logger.info(f"✅ Successfully loaded persona: {persona.name}")

            # Return with form fields populated including the ID
            return (
                gr.Dataframe(value=load_personas_data()),
                str(persona.id),  # Populate ID field
                persona.name,  # Populate name field
                persona.description or "",  # Populate description field
            )
        else:
            logger.warning(f"❌ Persona with ID {persona_id} returned None")
            gr.Warning(
                f"Persona with ID {persona_id} could not be loaded ⚠️", duration=4
            )
            return (gr.Dataframe(value=load_personas_data()), "", "", "")

    except PersonaRepositoryError as e:
        logger.error(f"❌ Database error: {str(e)}")
        gr.Warning(f"Database error: {str(e)} 💥!", duration=5)
        return (gr.Dataframe(value=load_personas_data()), "", "", "")
    except Exception as e:
        logger.error(f"❌ Unexpected error loading persona: {str(e)}", exc_info=True)
        gr.Warning(f"Unexpected error loading persona: {str(e)} 💥!", duration=5)
        return (gr.Dataframe(value=load_personas_data()), "", "", "")


def get_persona_choices() -> List[Tuple[str, str]]:
    """Get persona choices for dropdown selection.

    Returns:
        List of tuples (display_name, persona_id) for dropdown choices
        Includes a "None" option for no persona selection
    """
    try:
        logger.info("🔍 Getting persona choices for dropdown")
        personas = get_all_personas()

        # Start with None option
        choices = [("None", "")]

        # Add persona choices in format "Name (ID: X)"
        for persona in personas:
            display_name = f"{persona.name} (ID: {persona.id})"
            choices.append((display_name, str(persona.id)))

        logger.info(f"✅ Generated {len(choices)} persona choices")
        return choices

    except Exception as e:
        logger.error(f"❌ Failed to get persona choices: {e}")
        return [("None", "")]


def handle_speaker_persona_change(selected_persona_id: str, speaker_label: str) -> str:
    """Handle persona selection change for a speaker.

    Args:
        selected_persona_id: The selected persona ID (empty string for None)
        speaker_label: "A" or "B" to identify the speaker

    Returns:
        Status message about the persona change
    """
    try:
        logger.info(
            f"🔄 Persona change for Speaker {speaker_label}: {selected_persona_id}"
        )

        if not selected_persona_id or selected_persona_id.strip() == "":
            logger.info(f"✅ Speaker {speaker_label} persona cleared")
            return f"✅ Speaker {speaker_label} persona cleared"

        # Validate persona exists
        persona_id = int(selected_persona_id.strip())
        if not validate_persona_id_exists(persona_id):
            logger.warning(
                f"❌ Invalid persona ID for Speaker {speaker_label}: {persona_id}"
            )
            return f"❌ Invalid persona selected for Speaker {speaker_label}"

        # Get persona details
        from ..managers.persona_repository import get_persona_by_id

        persona = get_persona_by_id(persona_id)

        if persona:
            logger.info(f"✅ Speaker {speaker_label} assigned persona: {persona.name}")
            return f"✅ Speaker {speaker_label} assigned to persona: {persona.name}"
        else:
            logger.warning(
                f"❌ Persona not found for Speaker {speaker_label}: {persona_id}"
            )
            return f"❌ Persona not found for Speaker {speaker_label}"

    except ValueError:
        logger.warning(
            f"❌ Invalid persona ID format for Speaker {speaker_label}: {selected_persona_id}"
        )
        return f"❌ Invalid persona format for Speaker {speaker_label}"
    except Exception as e:
        logger.error(f"❌ Error changing persona for Speaker {speaker_label}: {e}")
        return f"❌ Error changing persona for Speaker {speaker_label}: {str(e)}"


def refresh_persona_dropdowns() -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Refresh both speaker persona dropdowns with current persona choices.

    Returns:
        Tuple of updated dropdowns for (Speaker A, Speaker B)
    """
    try:
        logger.info("🔄 Refreshing persona dropdown choices")
        persona_choices = get_persona_choices()

        return (
            gr.Dropdown(choices=persona_choices, value=""),
            gr.Dropdown(choices=persona_choices, value=""),
        )

    except Exception as e:
        logger.error(f"❌ Failed to refresh persona dropdowns: {e}")
        return (
            gr.Dropdown(choices=[("None", "")], value=""),
            gr.Dropdown(choices=[("None", "")], value=""),
        )
