"""Persona repository for database operations."""

import logging
from typing import List, Optional

from ..core.models import Persona
from ..utils.database import get_postgresql_client
from ..utils.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class PersonaRepositoryError(AudioProcessingError):
    """Exception raised for persona repository errors."""


class PersonaRepository:
    """Repository for persona database operations."""

    def __init__(self):
        self.client = get_postgresql_client()
        self.table_name = "ymemo_persona"

    def get_all_personas(self) -> List[Persona]:
        """Fetch all personas from the database, ordered by created_at DESC."""
        try:
            logger.info("🔍 Fetching all personas from database")

            query = """
                SELECT id, name, description, created_at, updated_at
                FROM {}
                ORDER BY created_at DESC
            """.format(
                self.table_name
            )

            results = self.client.execute_query(query)

            if not results:
                logger.info("📝 No personas found in database")
                return []

            personas = []
            for row in results:
                try:
                    persona = Persona.from_dict(dict(row))
                    personas.append(persona)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to parse persona row {row.get('id')}: {e}"
                    )
                    continue

            logger.info(f"✅ Successfully fetched {len(personas)} personas")
            return personas

        except Exception as e:
            logger.error(f"❌ Failed to fetch personas: {e}")
            raise PersonaRepositoryError(f"Failed to fetch personas: {e}")

    def create_persona(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Persona:
        """Create a new persona in the database."""
        try:
            logger.info(f"💾 Creating new persona: {name}")

            # Validate input
            if not name or not name.strip():
                raise ValueError("Persona name cannot be empty")

            # Prepare parameters
            name = name.strip()
            description = description.strip() if description else None

            # Insert into database
            query = """
                INSERT INTO {} (name, description)
                VALUES (%s, %s)
                RETURNING id, name, description, created_at, updated_at
            """.format(
                self.table_name
            )

            result = self.client.execute_insert_returning(query, (name, description))

            # Convert response to Persona object
            persona = Persona.from_dict(result)

            logger.info(f"✅ Successfully created persona with ID: {persona.id}")
            return persona

        except ValueError as e:
            logger.error(f"❌ Invalid persona data: {e}")
            raise PersonaRepositoryError(f"Invalid persona data: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to create persona: {e}")
            raise PersonaRepositoryError(f"Failed to create persona: {e}")

    def update_persona(self, persona_id: int, **updates) -> Optional[Persona]:
        """Update an existing persona."""
        try:
            logger.info(f"🔧 Updating persona ID: {persona_id}")

            # Validate persona ID
            if not persona_id or persona_id <= 0:
                raise ValueError("Invalid persona ID")

            # Validate updates
            if not updates:
                raise ValueError("No updates provided")

            # Clean up the updates dictionary
            clean_updates = {}
            for key, value in updates.items():
                if key in ["name", "description"] and isinstance(value, str):
                    clean_updates[key] = value.strip()
                else:
                    clean_updates[key] = value

            # Build dynamic update query
            set_clauses = []
            params = []
            for key, value in clean_updates.items():
                if key in ["name", "description"]:
                    set_clauses.append(f"{key} = %s")
                    params.append(value)

            if not set_clauses:
                raise ValueError("No valid fields to update")

            params.append(persona_id)

            # Update in database
            query = """
                UPDATE {}
                SET {}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id, name, description, created_at, updated_at
            """.format(
                self.table_name, ", ".join(set_clauses)
            )

            results = self.client.execute_query(query, tuple(params))

            if not results:
                logger.warning(f"⚠️ No persona found with ID {persona_id}")
                return None

            # Convert response to Persona object
            persona = Persona.from_dict(dict(results[0]))
            logger.info(f"✅ Successfully updated persona: {persona.name}")
            return persona

        except ValueError as e:
            logger.error(f"❌ Invalid persona update: {e}")
            raise PersonaRepositoryError(f"Invalid persona update: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to update persona {persona_id}: {e}")
            raise PersonaRepositoryError(f"Failed to update persona: {e}")

    def get_persona_by_id(self, persona_id: int) -> Optional[Persona]:
        """Get a specific persona by ID."""
        try:
            logger.info(f"🔍 Fetching persona with ID: {persona_id}")

            query = """
                SELECT id, name, description, created_at, updated_at
                FROM {}
                WHERE id = %s
            """.format(
                self.table_name
            )

            results = self.client.execute_query(query, (persona_id,))

            if not results:
                logger.info(f"📝 No persona found with ID: {persona_id}")
                return None

            persona = Persona.from_dict(dict(results[0]))
            logger.info(f"✅ Successfully fetched persona: {persona.name}")
            return persona

        except Exception as e:
            logger.error(f"❌ Failed to fetch persona {persona_id}: {e}")
            raise PersonaRepositoryError(f"Failed to fetch persona {persona_id}: {e}")

    def delete_persona(self, persona_id: int) -> bool:
        """Delete a persona from the database by ID."""
        try:
            logger.info(f"🗑️ Deleting persona with ID: {persona_id}")

            # Validate persona ID
            if not persona_id or persona_id <= 0:
                raise ValueError("Invalid persona ID")

            # Delete from database
            query = "DELETE FROM {} WHERE id = %s".format(self.table_name)

            rows_affected = self.client.execute_update(query, (persona_id,))

            if rows_affected > 0:
                logger.info(f"✅ Successfully deleted persona {persona_id}")
                return True

            logger.warning(f"⚠️ No persona found with ID {persona_id}")
            return False

        except ValueError as e:
            logger.error(f"❌ Invalid persona ID {persona_id}: {e}")
            raise PersonaRepositoryError(f"Invalid persona operation: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to delete persona {persona_id}: {e}")
            raise PersonaRepositoryError(f"Failed to delete persona: {e}")

    def get_personas_count(self) -> int:
        """Get the total number of personas."""
        try:
            logger.info("🔢 Counting personas in database")

            query = "SELECT COUNT(*) as count FROM {}".format(self.table_name)

            results = self.client.execute_query(query)
            count = results[0]["count"] if results else 0

            logger.info(f"✅ Total personas count: {count}")
            return count

        except Exception as e:
            logger.error(f"❌ Failed to count personas: {e}")
            raise PersonaRepositoryError(f"Failed to count personas: {e}")

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            logger.info("🔌 Testing persona repository connection")

            # Try to perform a simple query
            query = "SELECT 1 FROM {} LIMIT 1".format(self.table_name)
            self.client.execute_query(query)

            logger.info("✅ Persona repository connection test successful")
            return True

        except Exception as e:
            logger.error(f"❌ Persona repository connection test failed: {e}")
            return False


# Global repository instance
_persona_repository = None


def get_persona_repository() -> PersonaRepository:
    """Get the global persona repository instance."""
    global _persona_repository
    if _persona_repository is None:
        _persona_repository = PersonaRepository()
    return _persona_repository


# Convenience functions
def get_all_personas() -> List[Persona]:
    """Get all personas."""
    return get_persona_repository().get_all_personas()


def create_persona(
    name: str,
    description: Optional[str] = None,
) -> Persona:
    """Create a new persona."""
    return get_persona_repository().create_persona(name, description)


def update_persona(persona_id: int, **updates) -> Optional[Persona]:
    """Update a persona."""
    return get_persona_repository().update_persona(persona_id, **updates)


def get_persona_by_id(persona_id: int) -> Optional[Persona]:
    """Get a persona by ID."""
    return get_persona_repository().get_persona_by_id(persona_id)


def delete_persona_by_id(persona_id: int) -> bool:
    """Delete a persona by ID."""
    return get_persona_repository().delete_persona(persona_id)


def test_persona_connection() -> bool:
    """Test persona repository connection."""
    return get_persona_repository().test_connection()
