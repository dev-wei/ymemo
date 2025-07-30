#!/usr/bin/env python3
"""Test script for Azure Speech Service provider integration."""

import os
import sys
import logging
from pathlib import Path

# Add src to path so we can import modules
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from config.audio_config import get_config, AudioSystemConfig
from src.core.interfaces import AudioConfig, TranscriptionResult
from src.core.factory import AudioProcessorFactory, create_azure_speech_provider
from src.utils.exceptions import AzureSpeechError, AzureSpeechConfigurationError

def test_azure_configuration():
    """Test Azure Speech Service configuration."""
    print("🧪 Testing Azure Speech Service Configuration")
    print("=" * 60)
    
    # Test 1: Default configuration
    print("\n1. Testing default Azure configuration:")
    default_config = AudioSystemConfig()
    print(f"   azure_speech_key: {default_config.azure_speech_key}")
    print(f"   azure_speech_region: {default_config.azure_speech_region}")
    print(f"   azure_speech_language: {default_config.azure_speech_language}")
    print(f"   azure_enable_speaker_diarization: {default_config.azure_enable_speaker_diarization}")
    print(f"   azure_max_speakers: {default_config.azure_max_speakers}")
    print(f"   azure_speech_timeout: {default_config.azure_speech_timeout}")
    
    # Test 2: Environment variable configuration
    print("\n2. Testing environment variable configuration:")
    
    # Set environment variables for testing
    os.environ['AZURE_SPEECH_KEY'] = 'test_key_12345'
    os.environ['AZURE_SPEECH_REGION'] = 'westus2'
    os.environ['AZURE_SPEECH_LANGUAGE'] = 'es-ES'
    os.environ['AZURE_ENABLE_SPEAKER_DIARIZATION'] = 'true'
    os.environ['AZURE_MAX_SPEAKERS'] = '6'
    os.environ['AZURE_SPEECH_TIMEOUT'] = '45'
    
    env_config = AudioSystemConfig.from_env()
    print(f"   azure_speech_key: {env_config.azure_speech_key}")
    print(f"   azure_speech_region: {env_config.azure_speech_region}")
    print(f"   azure_speech_language: {env_config.azure_speech_language}")
    print(f"   azure_enable_speaker_diarization: {env_config.azure_enable_speaker_diarization}")
    print(f"   azure_max_speakers: {env_config.azure_max_speakers}")
    print(f"   azure_speech_timeout: {env_config.azure_speech_timeout}")
    
    # Test 3: Transcription config for Azure
    print("\n3. Testing transcription config generation:")
    env_config.transcription_provider = 'azure'
    transcription_config = env_config.get_transcription_config()
    print(f"   Transcription config: {transcription_config}")
    
    # Clean up environment variables
    for key in ['AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION', 'AZURE_SPEECH_LANGUAGE', 
                'AZURE_ENABLE_SPEAKER_DIARIZATION', 'AZURE_MAX_SPEAKERS', 'AZURE_SPEECH_TIMEOUT']:
        if key in os.environ:
            del os.environ[key]
    
    print("\n✅ Azure configuration test completed successfully!")
    return True

def test_azure_provider_creation():
    """Test Azure Speech Service provider creation."""
    print("\n🧪 Testing Azure Speech Provider Creation")
    print("=" * 60)
    
    try:
        # Test 1: Import Azure provider
        print("\n1. Testing Azure provider import:")
        from src.audio.providers.azure_speech import AzureSpeechProvider
        print("   ✅ AzureSpeechProvider imported successfully")
        
        # Test 2: Test factory registration
        print("\n2. Testing factory registration:")
        available_providers = AudioProcessorFactory.list_transcription_providers()
        print(f"   Available providers: {available_providers}")
        
        if 'azure' in available_providers:
            print("   ✅ Azure provider registered in factory")
        else:
            print("   ❌ Azure provider NOT registered in factory")
            return False
        
        # Test 3: Direct provider creation (without Azure SDK)
        print("\n3. Testing direct provider creation:")
        try:
            provider = AzureSpeechProvider(
                speech_key='test_key',
                region='eastus',
                language_code='en-US',
                enable_speaker_diarization=True,
                max_speakers=4
            )
            print(f"   ✅ Provider created: region={provider.region}, language={provider.language_code}")
            print(f"   ✅ Speaker diarization: {provider.enable_speaker_diarization}")
        except ImportError as e:
            print(f"   ⚠️  Azure SDK not installed - this is expected: {e}")
            print("   ℹ️  To install: pip install azure-cognitiveservices-speech>=1.45.0")
        except Exception as e:
            print(f"   ❌ Unexpected error creating provider: {e}")
            return False
        
        # Test 4: Factory creation
        print("\n4. Testing factory creation:")
        try:
            factory_provider = AudioProcessorFactory.create_transcription_provider(
                'azure',
                speech_key='test_factory_key',
                region='westus',
                language_code='en-US',
                enable_speaker_diarization=False,
                max_speakers=2,
                timeout=20
            )
            print(f"   ✅ Factory provider created: region={factory_provider.region}")
            print(f"   ✅ Speaker diarization: {factory_provider.enable_speaker_diarization}")
        except ImportError as e:
            print(f"   ⚠️  Azure SDK not installed - this is expected: {e}")
        except Exception as e:
            print(f"   ❌ Factory creation error: {e}")
            return False
        
        # Test 5: Convenience function
        print("\n5. Testing convenience function:")
        try:
            convenience_provider = create_azure_speech_provider(
                speech_key='test_convenience_key',
                region='eastus2',
                enable_speaker_diarization=True
            )
            print(f"   ✅ Convenience provider created: region={convenience_provider.region}")
        except ImportError as e:
            print(f"   ⚠️  Azure SDK not installed - this is expected: {e}")
        except Exception as e:
            print(f"   ❌ Convenience function error: {e}")
            return False
        
        print("\n✅ Azure provider creation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Azure provider creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transcription_result_compatibility():
    """Test TranscriptionResult compatibility with Azure provider."""
    print("\n🧪 Testing TranscriptionResult Compatibility")
    print("=" * 60)
    
    try:
        # Test TranscriptionResult with Azure-specific data
        azure_result = TranscriptionResult(
            text="Hello from Azure Speech Service",
            speaker_id="Speaker 1",
            confidence=0.92,
            start_time=1.5,
            end_time=3.2,
            is_partial=False,
            result_id="azure_result_001",
            utterance_id="azure_utterance_1",
            sequence_number=1
        )
        
        print(f"   Text: {azure_result.text}")
        print(f"   Speaker ID: {azure_result.speaker_id}")
        print(f"   Confidence: {azure_result.confidence}")
        print(f"   Is Partial: {azure_result.is_partial}")
        print(f"   Result ID: {azure_result.result_id}")
        print(f"   Utterance ID: {azure_result.utterance_id}")
        
        print("\n✅ TranscriptionResult compatibility test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ TranscriptionResult compatibility test failed: {e}")
        return False

def test_error_handling():
    """Test Azure-specific error handling."""
    print("\n🧪 Testing Azure Error Handling")
    print("=" * 60)
    
    try:
        from src.utils.exceptions import (
            AzureSpeechError, AzureSpeechConnectionError,
            AzureSpeechAuthenticationError, AzureSpeechConfigurationError
        )
        
        print("   ✅ All Azure exception classes imported successfully")
        
        # Test exception hierarchy
        try:
            raise AzureSpeechAuthenticationError("Test authentication error")
        except AzureSpeechError:
            print("   ✅ Exception hierarchy working correctly")
        except Exception:
            print("   ❌ Exception hierarchy broken")
            return False
        
        print("\n✅ Error handling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        return False

def print_azure_setup_instructions():
    """Print setup instructions for Azure Speech Service."""
    print("\n📋 Azure Speech Service Setup Instructions")
    print("=" * 60)
    print("To use Azure Speech Service with YMemo:")
    print()
    print("1. Create Azure Account and Speech Service Resource:")
    print("   - Go to https://portal.azure.com")
    print("   - Create a new 'Speech Service' resource")
    print("   - Note the API key and region")
    print()
    print("2. Set Environment Variables:")
    print("   export AZURE_SPEECH_KEY='your_speech_service_key'")
    print("   export AZURE_SPEECH_REGION='eastus'  # or your chosen region")
    print("   export AZURE_SPEECH_LANGUAGE='en-US'")
    print("   export AZURE_ENABLE_SPEAKER_DIARIZATION='true'  # optional")
    print("   export TRANSCRIPTION_PROVIDER='azure'")
    print()
    print("3. Install Dependencies:")
    print("   pip install azure-cognitiveservices-speech>=1.45.0")
    print()
    print("4. Run YMemo:")
    print("   python main.py")
    print()
    print("Available Azure regions: eastus, westus, westus2, eastus2, centralus,")
    print("                        northeurope, westeurope, southeastasia, etc.")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        success = True
        
        # Run all tests
        success &= test_azure_configuration()
        success &= test_azure_provider_creation()
        success &= test_transcription_result_compatibility()
        success &= test_error_handling()
        
        if success:
            print("\n🎉 All Azure Speech Service tests passed!")
            print_azure_setup_instructions()
        else:
            print("\n❌ Some Azure Speech Service tests failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)