"""Database utilities for Supabase integration."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Singleton Supabase client manager."""

    _instance: Optional["SupabaseClient"] = None
    _client: Client | None = None

    def __new__(cls) -> "SupabaseClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Supabase client."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError(
                    "Missing Supabase configuration. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file"
                )

            if (
                supabase_url == "your_supabase_url_here"
                or supabase_key == "your_supabase_anon_key_here"
            ):
                raise ValueError(
                    "Please update SUPABASE_URL and SUPABASE_ANON_KEY in .env file with your actual Supabase credentials"
                )

            self._client = create_client(supabase_url, supabase_key)
            logger.info("✅ Supabase client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            raise

    @property
    def client(self) -> Client:
        """Get the Supabase client."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            # Try to perform a simple query to test connection
            self.client.table("ymemo").select("id").limit(1).execute()
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

    def reset_client(self) -> None:
        """Reset the client (useful for testing)."""
        self._client = None
        self._initialize_client()


# Global instance
_supabase_client = SupabaseClient()


def get_supabase_client() -> Client:
    """Get the global Supabase client instance."""
    return _supabase_client.client


def test_database_connection() -> bool:
    """Test the database connection."""
    return _supabase_client.test_connection()


def reset_database_client() -> None:
    """Reset the database client (useful for testing)."""
    _supabase_client.reset_client()
