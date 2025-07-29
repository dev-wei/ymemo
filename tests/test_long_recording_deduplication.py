#!/usr/bin/env python3
"""Test long recording deduplication bug - reproduces the list truncation issue."""

import sys
sys.path.append('/Users/mweiwei/src/ymemo')

from src.core.interfaces import TranscriptionResult
from src.managers.session_manager import get_audio_session

def test_long_recording_deduplication():
    """Test that partial results work correctly during long recordings with list truncation."""
    
    # Get session manager
    session = get_audio_session()
    
    # Clear any existing transcriptions
    session.current_transcriptions.clear()
    session.active_partial_results.clear()
    
    print("🧪 Testing long recording deduplication bug...")
    print(f"Initial state: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    
    # Phase 1: Add 95 completed transcriptions to approach the truncation limit
    print("\n📝 Phase 1: Adding 95 completed transcriptions...")
    for i in range(95):
        result = TranscriptionResult(
            text=f"Completed utterance {i+1}.",
            confidence=0.9,
            is_partial=False,
            utterance_id=f"utterance_{i+1}",
            sequence_number=1,
            result_id=f"result_{i+1}"
        )
        session._on_transcription_received(result)
    
    print(f"After 95 completed: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    
    # Phase 2: Add partial results that will span the truncation boundary
    print("\n🔄 Phase 2: Adding partial results near truncation boundary...")
    
    # Add 3 more completed transcriptions (will be at 98 total)
    for i in range(95, 98):
        result = TranscriptionResult(
            text=f"Completed utterance {i+1}.",
            confidence=0.9,
            is_partial=False,
            utterance_id=f"utterance_{i+1}",
            sequence_number=1,
            result_id=f"result_{i+1}"
        )
        session._on_transcription_received(result)
    
    print(f"After 98 completed: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    
    # Add first partial result for utterance 99 (will be at 99 total)
    result_99_1 = TranscriptionResult(
        text="This is the start of utterance 99",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_99",
        sequence_number=1,
        result_id="result_99"
    )
    session._on_transcription_received(result_99_1)
    
    print(f"After partial 99-1: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    print(f"Active partials state: {session.active_partial_results}")
    
    # Add first partial result for utterance 100 (will be at 100 total)
    result_100_1 = TranscriptionResult(
        text="This is the start of utterance 100",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_100",
        sequence_number=1,
        result_id="result_100"
    )
    session._on_transcription_received(result_100_1)
    
    print(f"After partial 100-1: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    print(f"Active partials state: {session.active_partial_results}")
    
    # Phase 3: Add multiple transcriptions to trigger truncation and force index adjustment
    print("\n⚠️  Phase 3: Triggering truncation by adding 5 more transcriptions...")
    
    # Add 5 more transcriptions to go from 100 to 105, which will truncate back to 100
    # This will remove the first 5 transcriptions, shifting all indices down by 5
    for i in range(101, 106):
        result = TranscriptionResult(
            text=f"This will trigger truncation - utterance {i}.",
            confidence=0.9,
            is_partial=False,
            utterance_id=f"utterance_{i}",
            sequence_number=1,
            result_id=f"result_{i}"
        )
        session._on_transcription_received(result)
    
    print(f"After truncation trigger: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    print(f"Active partials state after truncation: {session.active_partial_results}")
    
    # Check if the partial results' indices are still valid
    print("🔍 Validating partial result indices after truncation:")
    for utterance_id, index in session.active_partial_results.items():
        if index < len(session.current_transcriptions):
            actual_content = session.current_transcriptions[index].get('content', '')
            expected_utterance = utterance_id.replace('utterance_', 'utterance ')
            content_matches = expected_utterance in actual_content
            print(f"  {utterance_id} at index {index}: {'✅' if content_matches else '❌'} '{actual_content[:50]}...'")
        else:
            print(f"  {utterance_id} at index {index}: ❌ INDEX OUT OF BOUNDS!")
    
    # Phase 4: Try to update the partial results (this is where the bug manifests)
    print("\n🐛 Phase 4: Attempting to update partial results (bug reproduction)...")
    
    # Try to update utterance 99 (should update existing, but will likely add new due to bug)
    result_99_2 = TranscriptionResult(
        text="This is the start of utterance 99 with more text",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_99",
        sequence_number=2,
        result_id="result_99"
    )
    
    # Count occurrences of utterance_99 before update
    utterance_99_count_before = sum(1 for msg in session.current_transcriptions 
                                    if "utterance 99" in msg.get('content', ''))
    
    session._on_transcription_received(result_99_2)
    
    # Count occurrences of utterance_99 after update
    utterance_99_count_after = sum(1 for msg in session.current_transcriptions 
                                   if "utterance 99" in msg.get('content', ''))
    
    print(f"Utterance 99 occurrences - Before: {utterance_99_count_before}, After: {utterance_99_count_after}")
    print(f"After partial 99-2: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    
    # Try to update utterance 100 
    result_100_2 = TranscriptionResult(
        text="This is the start of utterance 100 with more text",
        confidence=0.8,
        is_partial=True,
        utterance_id="utterance_100",
        sequence_number=2,
        result_id="result_100"
    )
    
    utterance_100_count_before = sum(1 for msg in session.current_transcriptions 
                                     if "utterance 100" in msg.get('content', ''))
    
    session._on_transcription_received(result_100_2)
    
    utterance_100_count_after = sum(1 for msg in session.current_transcriptions 
                                    if "utterance 100" in msg.get('content', ''))
    
    print(f"Utterance 100 occurrences - Before: {utterance_100_count_before}, After: {utterance_100_count_after}")
    print(f"After partial 100-2: {len(session.current_transcriptions)} transcriptions, {len(session.active_partial_results)} active partials")
    
    # Phase 5: Analysis and Validation
    print("\n🔍 Phase 5: Bug Analysis...")
    
    # Check for duplicates
    content_counts = {}
    for msg in session.current_transcriptions:
        content = msg.get('content', '')
        if 'utterance 99' in content or 'utterance 100' in content:
            if content in content_counts:
                content_counts[content] += 1
            else:
                content_counts[content] = 1
    
    print("Content analysis for utterances 99 and 100:")
    for content, count in content_counts.items():
        if count > 1:
            print(f"  🚨 DUPLICATE FOUND: '{content}' appears {count} times")
        else:
            print(f"  ✅ UNIQUE: '{content}' appears {count} time")
    
    # Test expectations
    print("\n📊 Test Results:")
    
    # Should have exactly 100 transcriptions after truncation
    transcription_count_ok = len(session.current_transcriptions) == 100
    print(f"  Transcription count (should be 100): {len(session.current_transcriptions)} {'✅' if transcription_count_ok else '❌'}")
    
    # Should have exactly 2 active partials (utterance_99 and utterance_100)
    active_partials_ok = len(session.active_partial_results) == 2
    print(f"  Active partials (should be 2): {len(session.active_partial_results)} {'✅' if active_partials_ok else '❌'}")
    
    # Should have no duplicates
    no_duplicates = all(count == 1 for count in content_counts.values())
    print(f"  No duplicates: {'✅' if no_duplicates else '❌'}")
    
    # Should have updated content, not duplicated
    utterance_99_updated = any("with more text" in msg.get('content', '') 
                               for msg in session.current_transcriptions 
                               if "utterance 99" in msg.get('content', ''))
    utterance_100_updated = any("with more text" in msg.get('content', '') 
                                for msg in session.current_transcriptions 
                                if "utterance 100" in msg.get('content', ''))
    
    print(f"  Utterance 99 updated: {'✅' if utterance_99_updated else '❌'}")
    print(f"  Utterance 100 updated: {'✅' if utterance_100_updated else '❌'}")
    
    # Overall test result
    all_tests_passed = (transcription_count_ok and active_partials_ok and 
                        no_duplicates and utterance_99_updated and utterance_100_updated)
    
    print(f"\n🎯 Overall Test Result: {'✅ PASSED' if all_tests_passed else '❌ FAILED (Bug Reproduced)'}")
    
    if not all_tests_passed:
        print("\n🐛 Bug confirmed: List truncation breaks partial result deduplication!")
        print("   The index adjustment logic in session_manager.py lines 127-131 is incorrect.")
    
    return all_tests_passed

def test_simple_partial_results():
    """Test that basic partial results work correctly (baseline test)."""
    
    print("\n🧪 Running baseline partial results test...")
    
    # Get fresh session
    session = get_audio_session()
    session.current_transcriptions.clear()
    session.active_partial_results.clear()
    
    # Test simple partial result sequence
    results = [
        TranscriptionResult(
            text="Hello",
            confidence=0.8,
            is_partial=True,
            utterance_id="test_utterance",
            sequence_number=1,
            result_id="test_result"
        ),
        TranscriptionResult(
            text="Hello world",
            confidence=0.8,
            is_partial=True,
            utterance_id="test_utterance",
            sequence_number=2,
            result_id="test_result"
        ),
        TranscriptionResult(
            text="Hello world how are you?",
            confidence=0.9,
            is_partial=False,
            utterance_id="test_utterance",
            sequence_number=3,
            result_id="test_result"
        )
    ]
    
    for i, result in enumerate(results):
        session._on_transcription_received(result)
        print(f"  After result {i+1}: {len(session.current_transcriptions)} transcriptions, content: '{session.current_transcriptions[0]['content'] if session.current_transcriptions else 'None'}'")
    
    # Validation
    success = (len(session.current_transcriptions) == 1 and 
              len(session.active_partial_results) == 0 and
              session.current_transcriptions[0]['content'] == "Hello world how are you?")
    
    print(f"  Baseline test: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

if __name__ == "__main__":
    print("🚀 Starting AWS Transcribe Deduplication Bug Tests")
    print("=" * 60)
    
    # Test basic functionality first
    baseline_passed = test_simple_partial_results()
    
    if baseline_passed:
        print("\n" + "=" * 60)
        # Test the long recording bug
        bug_test_passed = test_long_recording_deduplication()
        
        if not bug_test_passed:
            print("\n🎯 Bug successfully reproduced! Now we can fix it.")
            sys.exit(1)  # Exit with error to indicate bug found
        else:
            print("\n🎉 All tests passed! No bug found.")
            sys.exit(0)
    else:
        print("\n❌ Baseline test failed - basic functionality is broken")
        sys.exit(1)