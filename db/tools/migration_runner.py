"""Database migration runner for YMemo using PostgreSQL."""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised for migration-related errors."""


class PostgreSQLMigrationRunner:
    """Handles database migrations for YMemo using PostgreSQL."""

    def __init__(self, postgresql_client, migrations_dir: Optional[str] = None):
        """Initialize the migration runner.

        Args:
            postgresql_client: PostgreSQL client instance
            migrations_dir: Path to migrations directory (defaults to db/migrations)
        """
        self.client = postgresql_client
        self.migrations_table = "schema_migrations"

        # Set migrations directory
        if migrations_dir is None:
            # Default to db/migrations relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.migrations_dir = project_root / "db" / "migrations"
        else:
            self.migrations_dir = Path(migrations_dir)

        logger.info(
            f"🔧 Migration runner initialized with directory: {self.migrations_dir}"
        )

    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check
            schema: Database schema (default: public)

        Returns:
            True if table exists, False otherwise
        """
        try:
            logger.debug(f"🔍 Checking if table {schema}.{table_name} exists")

            # Query information_schema to check table existence
            query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            """

            results = self.client.execute_query(query, (schema, table_name))

            exists = len(results) > 0
            logger.debug(f"📋 Table {schema}.{table_name} exists: {exists}")

            return exists

        except Exception as e:
            logger.warning(f"⚠️ Could not check table existence for {table_name}: {e}")
            return False

    def get_table_schema(
        self, table_name: str, schema: str = "public"
    ) -> Optional[Dict]:
        """Get detailed schema information for a table.

        Args:
            table_name: Name of the table
            schema: Database schema (default: public)

        Returns:
            Dictionary with table schema details or None if table doesn't exist
        """
        try:
            logger.debug(f"🔍 Getting schema for table {schema}.{table_name}")

            # Query information_schema for column details
            query = """
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """

            results = self.client.execute_query(query, (schema, table_name))

            if not results:
                logger.debug(f"📋 No schema found for table {schema}.{table_name}")
                return None

            schema_info = {
                "table_name": table_name,
                "schema": schema,
                "columns": [dict(row) for row in results],
                "column_count": len(results),
            }

            logger.debug(
                f"📋 Retrieved schema for {schema}.{table_name}: {len(results)} columns"
            )
            return schema_info

        except Exception as e:
            logger.error(f"❌ Error getting table schema for {table_name}: {e}")
            return None

    def detect_database_state(self) -> Dict[str, Any]:
        """Detect the current state of the database.

        Returns:
            Dictionary with database state information
        """
        try:
            logger.info("🔍 Detecting database state...")

            state = {
                "ymemo_table_exists": self.table_exists("ymemo"),
                "ymemo_persona_table_exists": self.table_exists("ymemo_persona"),
                "migration_table_exists": self.table_exists(self.migrations_table),
                "is_fresh_database": False,
                "is_partial_deployment": False,
                "detected_issues": [],
            }

            # Get table schemas if they exist
            if state["ymemo_table_exists"]:
                state["ymemo_schema"] = self.get_table_schema("ymemo")

            if state["ymemo_persona_table_exists"]:
                state["ymemo_persona_schema"] = self.get_table_schema("ymemo_persona")

            # Determine database state
            if (
                not state["ymemo_table_exists"]
                and not state["ymemo_persona_table_exists"]
            ):
                state["is_fresh_database"] = True
                logger.info("🌱 Fresh database detected (no application tables)")
            elif state["ymemo_table_exists"] and not state["migration_table_exists"]:
                state["is_partial_deployment"] = True
                state["detected_issues"].append(
                    "ymemo table exists but no migration tracking"
                )
                logger.warning(
                    "⚠️ Partial deployment detected: ymemo exists without migration tracking"
                )
            elif not state["ymemo_table_exists"] and state["migration_table_exists"]:
                state["is_partial_deployment"] = True
                state["detected_issues"].append(
                    "migration table exists but no ymemo table"
                )
                logger.warning(
                    "⚠️ Inconsistent state: migration tracking without ymemo table"
                )

            logger.info(
                f"📊 Database state: Fresh={state['is_fresh_database']}, "
                f"Partial={state['is_partial_deployment']}, "
                f"Issues={len(state['detected_issues'])}"
            )

            return state

        except Exception as e:
            logger.error(f"❌ Error detecting database state: {e}")
            return {"error": str(e), "detected_issues": [f"Detection failed: {e}"]}

    def validate_expected_schema(
        self, table_name: str, expected_columns: List[str]
    ) -> Dict[str, Any]:
        """Validate that a table has the expected schema.

        Args:
            table_name: Name of the table to validate
            expected_columns: List of expected column names

        Returns:
            Dictionary with validation results
        """
        try:
            logger.info(f"🔍 Validating schema for table {table_name}")

            schema_info = self.get_table_schema(table_name)

            if not schema_info:
                return {
                    "valid": False,
                    "error": f"Table {table_name} does not exist",
                    "missing_columns": expected_columns,
                    "extra_columns": [],
                }

            actual_columns = [col["column_name"] for col in schema_info["columns"]]
            missing_columns = [
                col for col in expected_columns if col not in actual_columns
            ]
            extra_columns = [
                col for col in actual_columns if col not in expected_columns
            ]

            validation_result = {
                "valid": len(missing_columns) == 0,  # Valid if no missing columns
                "table_exists": True,
                "expected_columns": expected_columns,
                "actual_columns": actual_columns,
                "missing_columns": missing_columns,
                "extra_columns": extra_columns,
                "column_count_expected": len(expected_columns),
                "column_count_actual": len(actual_columns),
            }

            if validation_result["valid"]:
                logger.info(f"✅ Schema validation passed for {table_name}")
            else:
                logger.warning(
                    f"⚠️ Schema validation failed for {table_name}: "
                    f"missing {missing_columns}, extra {extra_columns}"
                )

            return validation_result

        except Exception as e:
            logger.error(f"❌ Error validating schema for {table_name}: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_columns": [],
                "extra_columns": [],
            }

    def initialize_migration_tracking(self) -> bool:
        """Create schema_migrations table if it doesn't exist.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🏗️ Initializing migration tracking table...")

            create_table_sql = """
            CREATE TABLE IF NOT EXISTS public.schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                sql_hash TEXT,
                status TEXT NOT NULL DEFAULT 'applied'
            )
            """

            # Execute using PostgreSQL client
            with self.client.get_cursor(dict_cursor=False) as (cursor, conn):
                cursor.execute(create_table_sql)

            logger.info("✅ Migration tracking table initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize migration tracking: {e}")
            return False

    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations.

        Returns:
            List of migration names that have been applied
        """
        try:
            logger.info("🔍 Fetching applied migrations...")

            query = """
                SELECT migration_name
                FROM {}
                WHERE status = 'applied'
                ORDER BY applied_at ASC
            """.format(
                self.migrations_table
            )

            results = self.client.execute_query(query)

            applied_migrations = [row["migration_name"] for row in results]
            logger.info(f"📝 Found {len(applied_migrations)} applied migrations")

            return applied_migrations

        except Exception as e:
            logger.warning(
                f"⚠️ Could not fetch applied migrations (table may not exist): {e}"
            )
            return []

    def get_migration_files(self) -> List[str]:
        """Get list of migration files from the migrations directory.

        Returns:
            Sorted list of migration file names
        """
        try:
            if not self.migrations_dir.exists():
                logger.warning(
                    f"⚠️ Migrations directory does not exist: {self.migrations_dir}"
                )
                return []

            migration_files = []
            for file_path in self.migrations_dir.glob("*.sql"):
                if file_path.is_file():
                    migration_files.append(file_path.stem)  # filename without extension

            # Sort migrations by name (assumes numeric prefix like 001_, 002_)
            migration_files.sort()

            logger.info(f"📂 Found {len(migration_files)} migration files")
            return migration_files

        except Exception as e:
            logger.error(f"❌ Error reading migration files: {e}")
            return []

    def get_pending_migrations(self) -> List[str]:
        """Get list of migrations that need to be applied.

        Returns:
            List of migration names that haven't been applied yet
        """
        all_migrations = self.get_migration_files()
        applied_migrations = self.get_applied_migrations()

        pending = [m for m in all_migrations if m not in applied_migrations]

        logger.info(f"⏳ Found {len(pending)} pending migrations: {pending}")
        return pending

    def calculate_sql_hash(self, sql_content: str) -> str:
        """Calculate SHA-256 hash of SQL content for integrity checking.

        Args:
            sql_content: SQL content to hash

        Returns:
            SHA-256 hash string
        """
        return hashlib.sha256(sql_content.encode('utf-8')).hexdigest()

    def read_migration_file(self, migration_name: str) -> str:
        """Read SQL content from a migration file.

        Args:
            migration_name: Name of the migration (without .sql extension)

        Returns:
            SQL content of the migration file

        Raises:
            MigrationError: If file cannot be read
        """
        try:
            file_path = self.migrations_dir / f"{migration_name}.sql"

            if not file_path.exists():
                raise MigrationError(f"Migration file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                raise MigrationError(f"Migration file is empty: {file_path}")

            logger.debug(
                f"📖 Read migration file: {migration_name} ({len(content)} characters)"
            )
            return content

        except Exception as e:
            raise MigrationError(f"Failed to read migration {migration_name}: {e}")

    def apply_migration(self, migration_name: str, dry_run: bool = False) -> bool:
        """Apply a single migration.

        Args:
            migration_name: Name of the migration to apply
            dry_run: If True, validate but don't actually execute

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🚀 Applying migration: {migration_name}")

            # Read migration SQL
            sql_content = self.read_migration_file(migration_name)
            sql_hash = self.calculate_sql_hash(sql_content)

            # Handle baseline migration with smart detection
            if "baseline" in migration_name.lower() or "001_baseline" in migration_name:
                return self._handle_baseline_migration(
                    migration_name, sql_content, sql_hash, dry_run
                )

            if dry_run:
                logger.info(f"🧪 DRY RUN: Would apply migration {migration_name}")
                return True

            # Execute SQL using PostgreSQL client
            statements = self._split_sql_statements(sql_content)

            with self.client.get_cursor(dict_cursor=False) as (cursor, conn):
                for statement in statements:
                    if statement.strip():
                        logger.debug(f"📝 Executing: {statement[:100]}...")
                        cursor.execute(statement)

            # Record successful migration
            self._record_migration(migration_name, sql_hash, "applied")

            logger.info(f"✅ Successfully applied migration: {migration_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to apply migration {migration_name}: {e}")
            # Record failed migration
            try:
                self._record_migration(migration_name, "", "failed")
            except:
                pass  # Don't fail if we can't record the failure
            return False

    def _handle_baseline_migration(
        self,
        migration_name: str,
        sql_content: str,
        sql_hash: str,
        dry_run: bool = False,
    ) -> bool:
        """Handle baseline migration with intelligence about existing schema.

        Args:
            migration_name: Name of the baseline migration
            sql_content: SQL content of the migration
            sql_hash: Hash of the SQL content
            dry_run: If True, don't actually execute

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📋 Processing baseline migration: {migration_name}")

            # Detect current database state
            db_state = self.detect_database_state()

            if db_state.get("error"):
                logger.error(f"❌ Could not detect database state: {db_state['error']}")
                return False

            # Check if ymemo table already exists
            ymemo_exists = db_state.get("ymemo_table_exists", False)

            if ymemo_exists:
                logger.info("📋 ymemo table already exists - validating schema")

                # Validate existing schema matches expectations
                expected_columns = [
                    "id",
                    "name",
                    "duration",
                    "transcription",
                    "created_at",
                    "audio_file_path",
                ]
                validation = self.validate_expected_schema("ymemo", expected_columns)

                if validation["valid"]:
                    logger.info("✅ Existing ymemo table schema is valid")
                    status = "skipped_existing"
                else:
                    logger.warning(
                        f"⚠️ Existing ymemo table schema issues: {validation}"
                    )
                    if dry_run:
                        logger.info("🧪 DRY RUN: Would need schema remediation")
                        return True
                    status = "applied_with_issues"
            else:
                logger.info("🌱 ymemo table does not exist - will create from baseline")

                if dry_run:
                    logger.info("🧪 DRY RUN: Would create ymemo table")
                    return True

                logger.info("🏗️ Creating ymemo table from baseline migration")
                statements = self._split_sql_statements(sql_content)

                with self.client.get_cursor(dict_cursor=False) as (cursor, conn):
                    for statement in statements:
                        if statement.strip():
                            logger.debug(f"📝 Executing: {statement[:100]}...")
                            cursor.execute(statement)

                status = "applied"

            # Record the migration
            if not dry_run:
                self._record_migration(migration_name, sql_hash, status)

            logger.info(f"✅ Baseline migration handled successfully: {status}")
            return True

        except Exception as e:
            logger.error(
                f"❌ Failed to handle baseline migration {migration_name}: {e}"
            )
            return False

    def _split_sql_statements(self, sql_content: str) -> List[str]:
        """Split SQL content into individual statements.

        Args:
            sql_content: Raw SQL content

        Returns:
            List of individual SQL statements
        """
        # If the content contains dollar quotes (functions, procedures), execute as single block
        if '$$' in sql_content:
            # Remove comments and return as single statement
            lines = []
            for line in sql_content.split('\n'):
                if line.strip() and not line.strip().startswith('--'):
                    lines.append(line)
            return ["\n".join(lines)] if lines else []

        # Otherwise, split by semicolon
        statements = []
        current_statement = ""

        lines = sql_content.split('\n')
        for line in lines:
            line = line.strip()

            # Skip comment lines
            if line.startswith('--') or not line:
                continue

            current_statement += line + "\n"

            # Check if statement ends with semicolon
            if line.endswith(';'):
                statements.append(current_statement.strip())
                current_statement = ""

        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())

        return [s for s in statements if s.strip()]

    def _record_migration(
        self, migration_name: str, sql_hash: str, status: str
    ) -> None:
        """Record migration in the tracking table.

        Args:
            migration_name: Name of the migration
            sql_hash: Hash of the SQL content
            status: Migration status (applied, failed, skipped_existing)
        """
        try:
            query = """
                INSERT INTO {} (migration_name, sql_hash, status, applied_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (migration_name)
                DO UPDATE SET
                    sql_hash = EXCLUDED.sql_hash,
                    status = EXCLUDED.status,
                    applied_at = EXCLUDED.applied_at
            """.format(
                self.migrations_table
            )

            self.client.execute_update(
                query, (migration_name, sql_hash, status, datetime.now())
            )

        except Exception as e:
            logger.error(f"❌ Failed to record migration {migration_name}: {e}")
            raise

    def run_pending_migrations(
        self, dry_run: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Run all pending migrations.

        Args:
            dry_run: If True, validate but don't actually execute

        Returns:
            Tuple of (successful_migrations, failed_migrations)
        """
        logger.info("🏁 Starting migration run...")

        # Initialize migration tracking if needed
        if not dry_run:
            self.initialize_migration_tracking()

        # Get pending migrations
        pending_migrations = self.get_pending_migrations()

        if not pending_migrations:
            logger.info("✨ No pending migrations found")
            return ([], [])

        successful = []
        failed = []

        for migration_name in pending_migrations:
            if self.apply_migration(migration_name, dry_run=dry_run):
                successful.append(migration_name)
            else:
                failed.append(migration_name)
                # Stop on first failure to maintain consistency
                logger.error(
                    f"🛑 Stopping migration run due to failure in: {migration_name}"
                )
                break

        logger.info(
            f"📊 Migration run complete: {len(successful)} successful, {len(failed)} failed"
        )
        return (successful, failed)

    def get_migration_status(self) -> Dict:
        """Get detailed migration status information.

        Returns:
            Dictionary with migration status details
        """
        try:
            all_migrations = self.get_migration_files()
            applied_migrations = self.get_applied_migrations()
            pending_migrations = self.get_pending_migrations()

            # Get migration records for additional details
            migration_records = {}
            try:
                query = f"SELECT * FROM {self.migrations_table} ORDER BY applied_at ASC"
                results = self.client.execute_query(query)
                for record in results:
                    migration_records[record["migration_name"]] = dict(record)
            except:
                pass  # Migration table might not exist yet

            status = {
                "total_migrations": len(all_migrations),
                "applied_count": len(applied_migrations),
                "pending_count": len(pending_migrations),
                "migrations": {
                    "applied": applied_migrations,
                    "pending": pending_migrations,
                },
                "details": migration_records,
                "migrations_directory": str(self.migrations_dir),
                "tracking_table": self.migrations_table,
            }

            return status

        except Exception as e:
            logger.error(f"❌ Error getting migration status: {e}")
            return {"error": str(e)}
