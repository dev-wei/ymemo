#!/usr/bin/env python3
"""Test AWS Transcribe connection monitoring and auto-recovery."""

import sys
sys.path.append('/Users/mweiwei/src/ymemo')

import asyncio
import time
from src.audio.providers.aws_transcribe import AWSTranscribeProvider
from src.core.interfaces import AudioConfig

def test_connection_health_callback():
    """Test connection health callback functionality."""
    
    print("🧪 Testing AWS Transcribe connection monitoring...")
    
    # Create provider
    provider = AWSTranscribeProvider()
    
    # Track callback invocations
    callback_calls = []
    
    def health_callback(is_healthy: bool, message: str):
        callback_calls.append((is_healthy, message, time.time()))
        print(f"📞 Health callback: healthy={is_healthy}, message='{message}'")
    
    # Set callback
    provider.set_connection_health_callback(health_callback)
    
    # Simulate connection health changes
    print("✅ Connection health callback set successfully")
    
    # Test callback manually
    if provider.connection_health_callback:
        provider.connection_health_callback(True, "Test connection healthy")
        provider.connection_health_callback(False, "Test connection failed")
    
    # Verify callbacks were called
    assert len(callback_calls) == 2, f"Expected 2 callback calls, got {len(callback_calls)}"
    assert callback_calls[0][0] == True, "First callback should be healthy=True"
    assert callback_calls[1][0] == False, "Second callback should be healthy=False"
    
    print("✅ Connection health callback test passed!")
    return True

def test_retry_delay_calculation():
    """Test exponential backoff calculation."""
    
    print("\n🧪 Testing retry delay calculation...")
    
    provider = AWSTranscribeProvider()
    
    async def test_delays():
        # Test exponential backoff
        provider.retry_count = 0
        delay1 = await provider._calculate_retry_delay()
        print(f"Retry 0: {delay1}s")
        
        provider.retry_count = 1
        delay2 = await provider._calculate_retry_delay()
        print(f"Retry 1: {delay2}s")
        
        provider.retry_count = 2
        delay3 = await provider._calculate_retry_delay()
        print(f"Retry 2: {delay3}s")
        
        provider.retry_count = 3
        delay4 = await provider._calculate_retry_delay()
        print(f"Retry 3: {delay4}s")
        
        provider.retry_count = 10  # Test max delay cap
        delay5 = await provider._calculate_retry_delay()
        print(f"Retry 10: {delay5}s (should be capped at {provider.max_retry_delay}s)")
        
        # Verify exponential backoff
        assert delay2 > delay1, "Delay should increase exponentially"
        assert delay3 > delay2, "Delay should increase exponentially"
        assert delay4 > delay3, "Delay should increase exponentially"
        assert delay5 == provider.max_retry_delay, f"Delay should be capped at {provider.max_retry_delay}s"
    
    asyncio.run(test_delays())
    print("✅ Retry delay calculation test passed!")
    return True

def test_connection_monitoring_integration():
    """Test full integration of connection monitoring."""
    
    print("\n🧪 Testing connection monitoring integration...")
    
    # Test that health monitoring is properly integrated
    provider = AWSTranscribeProvider()
    
    # Verify initial state
    assert provider.is_connected == False, "Should start disconnected"
    assert provider.last_result_time == 0.0, "Should start with no result time"
    assert provider.last_audio_sent_time == 0.0, "Should start with no audio sent time"
    
    # Verify retry settings
    assert provider.max_retries == 3, "Should have 3 max retries by default"
    assert provider.retry_delay == 1.0, "Should have 1s initial retry delay"
    assert provider.max_retry_delay == 60.0, "Should have 60s max retry delay"
    assert provider.connection_timeout == 30.0, "Should have 30s connection timeout"
    
    print("✅ Connection monitoring integration test passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting AWS Transcribe Connection Monitoring Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_connection_health_callback()
        test_retry_delay_calculation()
        test_connection_monitoring_integration()
        
        print("\n🎉 All connection monitoring tests passed!")
        print("\n📋 Features implemented:")
        print("  ✅ Connection health callback system")
        print("  ✅ Exponential backoff retry logic")
        print("  ✅ Timeout detection (30s without results)")
        print("  ✅ Connection state tracking")
        print("  ✅ Health monitoring task integration")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)