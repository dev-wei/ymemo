#!/usr/bin/env python3
"""
Gradio App - A simple Gradio application skeleton.
"""

import argparse
import atexit
import logging
import os
import signal

from dotenv import load_dotenv

from src.ui.interface import THEMES, create_interface

logger = logging.getLogger(__name__)

# Global flag to prevent multiple signal handlers
_shutdown_in_progress = False


def handle_migration_commands(args) -> int:
    """Handle database migration commands.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        from src.utils.database import get_migration_status, run_database_migrations

        if args.migration_status:
            logger.info("📊 Checking database migration status...")
            status = get_migration_status()

            print("\n" + "=" * 50)
            print("📊 DATABASE MIGRATION STATUS")
            print("=" * 50)
            print(f"📂 Migrations directory: {status['migrations_directory']}")
            print(f"📋 Total migrations: {status['total_migrations']}")
            print(f"✅ Applied: {status['applied_count']}")
            print(f"⏳ Pending: {status['pending_count']}")

            if status['migrations']['pending']:
                print(f"\n⏳ Pending migrations:")
                for migration in status['migrations']['pending']:
                    print(f"   • {migration}")

            if status['migrations']['applied']:
                print(f"\n✅ Applied migrations:")
                for migration in status['migrations']['applied']:
                    print(f"   • {migration}")

            print("=" * 50)
            return 0

        elif args.migration_dry_run:
            logger.info("🧪 Running migration dry run...")
            successful, failed = run_database_migrations(dry_run=True)

            print("\n" + "=" * 50)
            print("🧪 MIGRATION DRY RUN RESULTS")
            print("=" * 50)

            if successful:
                print(f"✅ Would apply {len(successful)} migrations:")
                for migration in successful:
                    print(f"   • {migration}")
            else:
                print("ℹ️ No migrations to apply")

            if failed:
                print(f"\n❌ Would fail {len(failed)} migrations:")
                for migration in failed:
                    print(f"   • {migration}")

            print("=" * 50)
            return 0

        elif args.migrate:
            logger.info("🚀 Running database migrations...")
            successful, failed = run_database_migrations(dry_run=False)

            print("\n" + "=" * 50)
            print("🚀 MIGRATION EXECUTION RESULTS")
            print("=" * 50)

            if successful:
                print(f"✅ Successfully applied {len(successful)} migrations:")
                for migration in successful:
                    print(f"   • {migration}")

            if failed:
                print(f"\n❌ Failed {len(failed)} migrations:")
                for migration in failed:
                    print(f"   • {migration}")
                print("=" * 50)
                return 1

            if not successful and not failed:
                print("ℹ️ No migrations to apply")

            print("=" * 50)
            return 0

        elif args.schema_validate:
            logger.info("🔍 Validating database schema...")

            from src.utils.database import get_migration_runner

            runner = get_migration_runner()

            # Validate ymemo table
            ymemo_expected = [
                "id",
                "name",
                "duration",
                "transcription",
                "created_at",
                "audio_file_path",
            ]
            ymemo_validation = runner.validate_expected_schema("ymemo", ymemo_expected)

            # Validate ymemo_persona table if it exists
            persona_expected = ["id", "name", "description", "created_at", "updated_at"]
            persona_validation = runner.validate_expected_schema(
                "ymemo_persona", persona_expected
            )

            print("\n" + "=" * 50)
            print("🔍 SCHEMA VALIDATION RESULTS")
            print("=" * 50)

            print(f"📋 ymemo table:")
            print(f"   ✅ Valid: {ymemo_validation['valid']}")
            if ymemo_validation.get('missing_columns'):
                print(f"   ❌ Missing columns: {ymemo_validation['missing_columns']}")
            if ymemo_validation.get('extra_columns'):
                print(f"   ⚠️ Extra columns: {ymemo_validation['extra_columns']}")

            print(f"\n📋 ymemo_persona table:")
            if persona_validation.get('table_exists'):
                print(f"   ✅ Valid: {persona_validation['valid']}")
                if persona_validation.get('missing_columns'):
                    print(
                        f"   ❌ Missing columns: {persona_validation['missing_columns']}"
                    )
                if persona_validation.get('extra_columns'):
                    print(f"   ⚠️ Extra columns: {persona_validation['extra_columns']}")
            else:
                print(
                    "   ℹ️ Table does not exist (expected for pre-persona deployments)"
                )

            overall_valid = ymemo_validation['valid'] and (
                not persona_validation.get('table_exists')
                or persona_validation['valid']
            )
            print(
                f"\n🎯 Overall schema validity: {'✅ VALID' if overall_valid else '❌ ISSUES FOUND'}"
            )

            print("=" * 50)
            return 0 if overall_valid else 1

        elif args.database_state:
            logger.info("🔍 Analyzing database state...")

            from src.utils.database import get_migration_runner

            runner = get_migration_runner()
            db_state = runner.detect_database_state()

            print("\n" + "=" * 50)
            print("🔍 DATABASE STATE ANALYSIS")
            print("=" * 50)

            if db_state.get("error"):
                print(f"❌ Error detecting state: {db_state['error']}")
                print("=" * 50)
                return 1

            print(
                f"📊 Database Type: {'🌱 Fresh' if db_state['is_fresh_database'] else '🏗️ Existing'}"
            )
            print(
                f"📋 ymemo table: {'✅ Exists' if db_state['ymemo_table_exists'] else '❌ Missing'}"
            )
            print(
                f"👤 ymemo_persona table: {'✅ Exists' if db_state['ymemo_persona_table_exists'] else '❌ Missing'}"
            )
            print(
                f"🗂️ Migration tracking: {'✅ Exists' if db_state['migration_table_exists'] else '❌ Missing'}"
            )

            if db_state['is_partial_deployment']:
                print(f"\n⚠️ PARTIAL DEPLOYMENT DETECTED")
                print("   This database is in an inconsistent state.")

            if db_state['detected_issues']:
                print(f"\n❌ DETECTED ISSUES:")
                for issue in db_state['detected_issues']:
                    print(f"   • {issue}")

            if db_state.get('ymemo_schema'):
                print(f"\n📋 ymemo table schema:")
                print(f"   Columns: {db_state['ymemo_schema']['column_count']}")

            if db_state.get('ymemo_persona_schema'):
                print(f"\n👤 ymemo_persona table schema:")
                print(f"   Columns: {db_state['ymemo_persona_schema']['column_count']}")

            print("=" * 50)
            return 0

    except Exception as e:
        logger.error(f"❌ Migration command failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


def cleanup_on_exit():
    """Clean up resources on exit."""
    logger.info("🧹 Cleaning up resources on exit...")
    try:
        from src.managers.session_manager import get_audio_session

        session = get_audio_session()
        if session.is_recording():
            logger.info("🛑 Stopping recording on exit...")
            # Use threading with timeout to prevent cleanup from hanging
            import threading

            stop_result = [False]  # Use list to make it mutable

            def stop_recording_thread():
                try:
                    success = session.stop_recording()
                    stop_result[0] = success
                    logger.info(f"🛑 Recording stopped: {success}")
                except Exception as e:
                    logger.error(f"❌ Error stopping recording: {e}")

            # Start stop operation in a separate thread
            stop_thread = threading.Thread(target=stop_recording_thread)
            stop_thread.daemon = True
            stop_thread.start()

            # Wait for up to 1 second for cleanup to complete
            stop_thread.join(timeout=1.0)

            if stop_thread.is_alive():
                logger.warning("⚠️ Recording stop timed out - abandoning cleanup")
            else:
                logger.info(f"✅ Recording cleanup completed: {stop_result[0]}")

        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")


def signal_handler(signum, frame):
    """Handle signals like SIGINT, SIGTERM."""
    global _shutdown_in_progress

    if _shutdown_in_progress:
        logger.info("🛑 Shutdown already in progress, forcing exit...")
        import os

        os._exit(1)

    _shutdown_in_progress = True
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")

    try:
        cleanup_on_exit()
    except Exception as e:
        logger.error(f"❌ Error during signal cleanup: {e}")
    finally:
        logger.info("🛑 Exiting application...")
        # Use os._exit to force immediate termination
        import os

        os._exit(0)


def main():
    """Main entry point for the Gradio application."""
    load_dotenv()

    # Get log level from environment variable
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Set specific logger levels based on environment
    audio_log_level = getattr(logging, log_level, logging.INFO)
    logging.getLogger('src.audio').setLevel(audio_log_level)
    logging.getLogger('src.audio.providers.aws_transcribe').setLevel(audio_log_level)
    logging.getLogger('src.audio.providers.pyaudio_capture').setLevel(audio_log_level)
    logging.getLogger('src.core.processor').setLevel(audio_log_level)
    logging.getLogger('src.managers.session_manager').setLevel(audio_log_level)
    logging.getLogger('src.managers.meeting_repository').setLevel(audio_log_level)

    parser = argparse.ArgumentParser(
        description="Gradio App - Simple Gradio application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ip",
        type=str,
        default="127.0.0.1",
        help="IP address to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port", type=int, default=7860, help="Port to listen on (default: 7860)"
    )

    parser.add_argument(
        "--theme",
        type=str,
        default="Ocean",
        choices=list(THEMES.keys()),
        help=f"UI theme to use (default: Ocean). Available: {', '.join(THEMES.keys())}",
    )

    parser.add_argument(
        "--share", action="store_true", help="Create a public shareable link"
    )

    # Database migration commands
    parser.add_argument(
        "--migrate", action="store_true", help="Run pending database migrations"
    )

    parser.add_argument(
        "--migration-status", action="store_true", help="Show database migration status"
    )

    parser.add_argument(
        "--migration-dry-run",
        action="store_true",
        help="Show what migrations would be applied (dry run)",
    )

    parser.add_argument(
        "--schema-validate",
        action="store_true",
        help="Validate current database schema against expected structure",
    )

    parser.add_argument(
        "--database-state",
        action="store_true",
        help="Show detailed database state information",
    )

    args = parser.parse_args()

    # Handle migration commands first (before starting the app)
    if (
        args.migration_status
        or args.migration_dry_run
        or args.migrate
        or args.schema_validate
        or args.database_state
    ):
        return handle_migration_commands(args)

    logger.info("🚀 Starting Gradio App...")
    logger.info(f"📍 Server: http://{args.ip}:{args.port}")
    logger.info(f"🎨 Theme: {args.theme}")
    logger.info(f"🎤 Audio logging: {log_level}")

    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and launch the interface
    demo = create_interface(theme_name=args.theme)

    try:
        demo.queue(
            max_size=20,  # Maximum number of requests in queue
            api_open=False,  # Don't expose API endpoints
        ).launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
            show_error=True,
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down Gradio App...")
        cleanup_on_exit()
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}", exc_info=True)
        cleanup_on_exit()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
