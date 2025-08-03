"""Database utilities for PostgreSQL integration."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PostgreSQLClient:
    """Singleton PostgreSQL client manager."""

    _instance: Optional["PostgreSQLClient"] = None
    _connection_params: Dict[str, Any] = {}

    def __new__(cls) -> "PostgreSQLClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._connection_params:
            self._initialize_connection_params()

    def _initialize_connection_params(self) -> None:
        """Initialize PostgreSQL connection parameters from POSTGRES_URL."""
        try:
            postgres_url = os.getenv("POSTGRES_URL")

            if not postgres_url:
                raise ValueError(
                    "Missing PostgreSQL configuration. Please set POSTGRES_URL in .env file"
                )

            if postgres_url == "your_postgres_url_here":
                raise ValueError(
                    "Please update POSTGRES_URL in .env file with your actual PostgreSQL connection string"
                )

            # Parse the PostgreSQL URL
            parsed = urlparse(postgres_url)

            self._connection_params = {
                "host": parsed.hostname,
                "port": parsed.port or 5432,
                "database": parsed.path[1:] if parsed.path else "postgres",
                "user": parsed.username,
                "password": parsed.password,
                "connect_timeout": 30,
            }

            logger.info("✅ PostgreSQL connection parameters initialized successfully")

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize PostgreSQL connection parameters: {e}"
            )
            raise

    @contextmanager
    def get_connection(self):
        """Get a PostgreSQL connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(**self._connection_params)
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """Get a database cursor with automatic cleanup."""
        with self.get_connection() as conn:
            cursor_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor, conn
            except Exception as e:
                conn.rollback()
                raise
            else:
                conn.commit()
            finally:
                cursor.close()

    def execute_query(
        self, query: str, params: tuple = None, fetch: str = "all"
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query, params)

            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                result = cursor.fetchone()
                return [result] if result else []
            elif fetch == "none":
                return []
            else:
                raise ValueError("fetch must be 'all', 'one', or 'none'")

    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_cursor(dict_cursor=False) as (cursor, conn):
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_insert_returning(
        self, query: str, params: tuple = None
    ) -> Dict[str, Any]:
        """Execute an INSERT query with RETURNING clause."""
        with self.get_cursor() as (cursor, conn):
            cursor.execute(query, params)
            result = cursor.fetchone()
            if not result:
                raise ValueError("INSERT query did not return any data")
            return dict(result)

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            logger.info("🔌 Testing PostgreSQL connection")

            # Try to perform a simple query
            result = self.execute_query("SELECT 1 as test", fetch="one")

            if result and result[0].get("test") == 1:
                logger.info("✅ PostgreSQL connection test successful")
                return True
            else:
                logger.error("❌ PostgreSQL connection test failed: Unexpected result")
                return False

        except Exception as e:
            logger.error(f"❌ PostgreSQL connection test failed: {e}")
            return False

    def reset_connection_params(self) -> None:
        """Reset connection parameters (useful for testing)."""
        self._connection_params = {}
        self._initialize_connection_params()


# Global instance
_postgresql_client = PostgreSQLClient()


def get_postgresql_client() -> PostgreSQLClient:
    """Get the global PostgreSQL client instance."""
    return _postgresql_client


def get_database_connection():
    """Get a database connection context manager."""
    return _postgresql_client.get_connection()


def get_database_cursor(dict_cursor: bool = True):
    """Get a database cursor context manager."""
    return _postgresql_client.get_cursor(dict_cursor=dict_cursor)


def test_database_connection() -> bool:
    """Test the database connection."""
    return _postgresql_client.test_connection()


def reset_database_client() -> None:
    """Reset the database client (useful for testing)."""
    _postgresql_client.reset_connection_params()


def get_migration_runner():
    """Get a migration runner instance for the current PostgreSQL client.

    Returns:
        PostgreSQLMigrationRunner instance
    """
    from db.tools.migration_runner import PostgreSQLMigrationRunner

    return PostgreSQLMigrationRunner(_postgresql_client)


def run_database_migrations(dry_run: bool = False):
    """Run pending database migrations.

    Args:
        dry_run: If True, validate but don't execute migrations

    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    runner = get_migration_runner()
    return runner.run_pending_migrations(dry_run=dry_run)


def get_migration_status():
    """Get current database migration status.

    Returns:
        Dictionary with migration status information
    """
    runner = get_migration_runner()
    return runner.get_migration_status()


# Legacy compatibility functions (maintain same API as before)
def get_supabase_client():
    """Legacy compatibility function - returns PostgreSQL client."""
    logger.warning(
        "get_supabase_client() is deprecated, use get_postgresql_client() instead"
    )
    return get_postgresql_client()
