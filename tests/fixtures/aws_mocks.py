"""AWS-specific mocking utilities for testing AWS Transcribe integration.

This module provides specialized mock objects for AWS services, particularly
focused on AWS Transcribe streaming functionality.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any, Optional

from src.core.interfaces import TranscriptionResult
from .async_mocks import AsyncIteratorMock


class MockAWSTranscribeProvider:
    """Comprehensive mock for AWS Transcribe provider."""
    
    def __init__(self, region: str = "us-east-1", language_code: str = "en-US"):
        """Initialize AWS provider mock."""
        self.region = region
        self.language_code = language_code
        self.profile_name = None
        
        # State tracking
        self.client = None
        self.stream = None
        self.result_queue = None
        self._current_event_loop = None
        self.is_connected = False
        self._streaming_task = None
        self._health_check_task = None
        
        # Connection health
        self.last_result_time = 0.0
        self.last_audio_sent_time = 0.0
        self.connection_timeout = 30.0
        self.retry_count = 0
        self.max_retries = 3
        self.connection_health_callback = None
        
        # Mock methods
        self.start_stream = AsyncMock()
        self.stop_stream = AsyncMock()
        self.send_audio = AsyncMock()
        self.get_transcription = AsyncMock()
        self.set_connection_health_callback = Mock()
        
        # Configure realistic behavior
        self._configure_realistic_behavior()
    
    def _configure_realistic_behavior(self):
        """Configure realistic behavior for mocks."""
        
        async def mock_start_stream(audio_config):
            """Mock start stream with realistic setup."""
            self.is_connected = True
            self.result_queue = asyncio.Queue()
            self._current_event_loop = asyncio.get_event_loop()
            
        async def mock_stop_stream():
            """Mock stop stream with cleanup."""
            self.is_connected = False
            if self.result_queue:
                # Clear queue
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
            
        async def mock_send_audio(audio_chunk: bytes):
            """Mock send audio with connection health tracking."""
            self.last_audio_sent_time = asyncio.get_event_loop().time()
            
        async def mock_get_transcription():
            """Mock transcription generator."""
            if self.result_queue:
                while True:
                    try:
                        result = await asyncio.wait_for(
                            self.result_queue.get(), 
                            timeout=0.1
                        )
                        yield result
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
        
        # Apply realistic behavior
        self.start_stream.side_effect = mock_start_stream
        self.stop_stream.side_effect = mock_stop_stream
        self.send_audio.side_effect = mock_send_audio
        self.get_transcription.side_effect = mock_get_transcription
    
    def simulate_transcription_result(self, result: TranscriptionResult):
        """Simulate receiving a transcription result."""
        if self.result_queue:
            asyncio.create_task(self.result_queue.put(result))
    
    def simulate_connection_loss(self):
        """Simulate connection loss."""
        self.is_connected = False
        if self.connection_health_callback:
            self.connection_health_callback(False, "Connection lost")
    
    def simulate_connection_recovery(self):
        """Simulate connection recovery."""
        self.is_connected = True
        self.retry_count = 0
        if self.connection_health_callback:
            self.connection_health_callback(True, "Connection recovered")


class MockBoto3Session:
    """Mock for boto3.Session with credential handling."""
    
    def __init__(self, has_credentials: bool = True, region: str = "us-east-1"):
        """Initialize boto3 session mock."""
        self.has_credentials = has_credentials
        self.region_name = region
        self.profile_name = None
        
        # Mock credentials
        if has_credentials:
            self.credentials = Mock()
            self.credentials.access_key = "AKIAIOSFODNN7EXAMPLE"
            self.credentials.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        else:
            self.credentials = None
    
    def get_credentials(self):
        """Return mock credentials."""
        return self.credentials
    
    def client(self, service_name: str, region_name: str = None):
        """Return mock AWS client."""
        if service_name == 'transcribe':
            return MockTranscribeClient()
        return Mock()


class MockTranscribeClient:
    """Mock for AWS Transcribe client."""
    
    def __init__(self):
        """Initialize Transcribe client mock."""
        self.start_stream_transcription_calls = []
        
    def start_stream_transcription(self, **kwargs):
        """Mock start stream transcription."""
        self.start_stream_transcription_calls.append(kwargs)
        return Mock()


class MockTranscribeStreamingClient:
    """Mock for AWS Transcribe streaming client."""
    
    def __init__(self):
        """Initialize streaming client mock."""
        self.start_stream_calls = []
        self.mock_stream = MockTranscriptStream()
        
        # Configure async method
        self.start_stream_transcription = AsyncMock(return_value=self.mock_stream)
    
    def configure_stream_data(self, events: List[Any]):
        """Configure mock stream to return specific events."""
        self.mock_stream.configure_events(events)


class MockTranscriptStream:
    """Mock for AWS Transcribe stream with realistic event handling."""
    
    def __init__(self):
        """Initialize transcript stream mock."""
        self.input_stream = MockInputStream()
        self.output_stream = MockOutputStream()
        self.events = []
        self.closed = False
    
    def configure_events(self, events: List[Any]):
        """Configure events for output stream."""
        self.output_stream.configure_events(events)


class MockInputStream:
    """Mock for AWS Transcribe input stream."""
    
    def __init__(self):
        """Initialize input stream mock."""
        self.sent_audio = []
        self.ended = False
        
        # Async methods
        self.send_audio_event = AsyncMock()
        self.end_stream = AsyncMock()
    
    def configure_behavior(self):
        """Configure realistic behavior."""
        
        async def mock_send_audio(audio_chunk: bytes):
            """Mock send audio with tracking."""
            self.sent_audio.append(audio_chunk)
            
        async def mock_end_stream():
            """Mock end stream."""
            self.ended = True
            
        self.send_audio_event.side_effect = mock_send_audio
        self.end_stream.side_effect = mock_end_stream


class MockOutputStream:
    """Mock for AWS Transcribe output stream."""
    
    def __init__(self):
        """Initialize output stream mock."""
        self.events = []
        self.closed = False
    
    def configure_events(self, events: List[Any]):
        """Configure events to yield."""
        self.events = events
    
    def __aiter__(self):
        """Return async iterator over events."""
        return AsyncIteratorMock(self.events)


class MockTranscriptEvent:
    """Mock for AWS Transcribe transcript event."""
    
    def __init__(self, transcript_data: Dict[str, Any]):
        """Initialize with transcript data."""
        self.transcript = MockTranscript(transcript_data)


class MockTranscript:
    """Mock for AWS Transcribe transcript."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with transcript data."""
        self.results = [MockResult(result_data) for result_data in data.get('results', [])]


class MockResult:
    """Mock for AWS Transcribe result."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with result data."""
        self.alternatives = [MockAlternative(alt) for alt in data.get('alternatives', [])]
        self.is_partial = data.get('is_partial', False)
        self.result_id = data.get('result_id', 'mock_result_001')
        self.start_time = data.get('start_time', 0.0)
        self.end_time = data.get('end_time', 1.0)


class MockAlternative:
    """Mock for AWS Transcribe alternative."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with alternative data."""
        self.transcript = data.get('transcript', 'Mock transcript')
        self.confidence = data.get('confidence', 0.95)
        self.items = [MockItem(item) for item in data.get('items', [])]


class MockItem:
    """Mock for AWS Transcribe item."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with item data."""
        self.content = data.get('content', 'word')
        self.start_time = data.get('start_time', 0.0)
        self.end_time = data.get('end_time', 1.0)
        self.speaker = data.get('speaker', 'spk_0')


class AWSMockFactory:
    """Factory for creating complete AWS mock setups."""
    
    @staticmethod
    def create_full_transcribe_setup(with_credentials: bool = True):
        """Create complete AWS Transcribe mock setup."""
        
        # Create session mock
        session_mock = MockBoto3Session(
            has_credentials=with_credentials,
            region="us-east-1"
        )
        
        # Create streaming client mock
        streaming_client = MockTranscribeStreamingClient()
        
        # Create provider mock
        provider = MockAWSTranscribeProvider()
        
        # Create sample transcript events
        sample_events = [
            MockTranscriptEvent({
                'results': [
                    {
                        'alternatives': [
                            {
                                'transcript': 'Hello world',
                                'confidence': 0.95,
                                'items': [
                                    {'content': 'Hello', 'start_time': 0.0, 'end_time': 0.5},
                                    {'content': 'world', 'start_time': 0.6, 'end_time': 1.0}
                                ]
                            }
                        ],
                        'is_partial': False,
                        'result_id': 'result_001'
                    }
                ]
            })
        ]
        
        streaming_client.configure_stream_data(sample_events)
        
        return {
            'session': session_mock,
            'streaming_client': streaming_client,
            'provider': provider,
            'events': sample_events
        }
    
    @staticmethod
    def create_error_scenario_setup():
        """Create AWS mock setup that simulates errors."""
        
        session_mock = MockBoto3Session(has_credentials=False)
        
        streaming_client = Mock()
        streaming_client.start_stream_transcription = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        
        provider = MockAWSTranscribeProvider()
        provider.start_stream.side_effect = Exception("AWS connection error")
        
        return {
            'session': session_mock,
            'streaming_client': streaming_client,
            'provider': provider
        }
    
    @staticmethod
    def create_partial_results_scenario():
        """Create AWS mock setup with partial results sequence."""
        
        setup = AWSMockFactory.create_full_transcribe_setup()
        
        # Create sequence of partial results followed by final
        partial_events = [
            MockTranscriptEvent({
                'results': [{
                    'alternatives': [{'transcript': 'Hello'}],
                    'is_partial': True,
                    'result_id': 'result_001'
                }]
            }),
            MockTranscriptEvent({
                'results': [{
                    'alternatives': [{'transcript': 'Hello there'}],
                    'is_partial': True,
                    'result_id': 'result_001'
                }]
            }),
            MockTranscriptEvent({
                'results': [{
                    'alternatives': [{'transcript': 'Hello there, how are you?'}],
                    'is_partial': False,
                    'result_id': 'result_001'
                }]
            })
        ]
        
        setup['streaming_client'].configure_stream_data(partial_events)
        setup['events'] = partial_events
        
        return setup