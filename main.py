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

    args = parser.parse_args()

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
