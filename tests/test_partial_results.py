#!/usr/bin/env python3
"""Test partial result handling logic."""

import sys
sys.path.append('/Users/mweiwei/src/ymemo')

from src.core.interfaces import TranscriptionResult
from src.managers.session_manager import get_audio_session

def test_partial_result_handling():
    """Test that partial results are properly handled."""
    
    # Get session manager
    session = get_audio_session()
    
    # Clear any existing transcriptions
    session.current_transcriptions.clear()
    session.active_partial_results.clear()
    
    # Test case: Progressive partial results
    print("Testing progressive partial results...")
    
    # First partial result
    result1 = TranscriptionResult(
        text="Hello, hello",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_1",
        sequence_number=1,
        result_id="result_1"
    )
    
    session._on_transcription_received(result1)
    print(f"After first partial: {len(session.current_transcriptions)} transcriptions")
    print(f"Content: '{session.current_transcriptions[0]['content']}'")
    
    # Second partial result (should replace first)
    result2 = TranscriptionResult(
        text="Hello, hello, hello",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_1",
        sequence_number=2,
        result_id="result_1"
    )
    
    session._on_transcription_received(result2)
    print(f"After second partial: {len(session.current_transcriptions)} transcriptions")
    print(f"Content: '{session.current_transcriptions[0]['content']}'")
    
    # Final result (should replace partial)
    result3 = TranscriptionResult(
        text="Hello, hello, hello, hello.",
        confidence=0.9,
        is_partial=False,
        utterance_id="utterance_1",
        sequence_number=3,
        result_id="result_1"
    )
    
    session._on_transcription_received(result3)
    print(f"After final result: {len(session.current_transcriptions)} transcriptions")
    print(f"Content: '{session.current_transcriptions[0]['content']}'")
    print(f"Is partial: {session.current_transcriptions[0]['is_partial']}")
    print(f"Active partials: {len(session.active_partial_results)}")
    
    # Test successful if we have exactly 1 transcription and no active partials
    assert len(session.current_transcriptions) == 1, f"Expected 1 transcription, got {len(session.current_transcriptions)}"
    assert len(session.active_partial_results) == 0, f"Expected 0 active partials, got {len(session.active_partial_results)}"
    assert session.current_transcriptions[0]['content'] == "Hello, hello, hello, hello."
    assert session.current_transcriptions[0]['is_partial'] == False
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_partial_result_handling()