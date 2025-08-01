"""Advanced result synchronization and merging system for dual-channel transcription."""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from ..core.interfaces import TranscriptionResult


logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategy for merging overlapping results."""
    TIMESTAMP_ORDER = "timestamp_order"  # Order by timestamp
    CONFIDENCE_PRIORITY = "confidence_priority"  # Higher confidence wins
    CHANNEL_PRIORITY = "channel_priority"  # Specific channel has priority
    INTERLEAVE = "interleave"  # Interleave results from both channels


@dataclass
class BufferedResult:
    """A transcription result with buffering metadata."""
    result: TranscriptionResult
    channel: str  # 'left' or 'right'
    arrival_time: float = field(default_factory=time.time)
    processed: bool = False


@dataclass 
class MergeStatistics:
    """Statistics for result merging operations."""
    total_left_results: int = 0
    total_right_results: int = 0
    merged_results: int = 0
    conflicting_results: int = 0
    dropped_results: int = 0
    average_merge_delay: float = 0.0
    timestamp_corrections: int = 0


class DualChannelResultMerger:
    """
    Advanced result merger for dual-channel transcription.
    
    This class handles sophisticated merging of transcription results from
    two separate channels, ensuring proper ordering, handling conflicts,
    and maintaining speaker attribution.
    """
    
    def __init__(
        self,
        merge_strategy: MergeStrategy = MergeStrategy.TIMESTAMP_ORDER,
        buffer_window: float = 2.0,
        max_buffer_size: int = 100,
        confidence_threshold: float = 0.0,
        priority_channel: Optional[str] = None
    ):
        """
        Initialize the result merger.
        
        Args:
            merge_strategy: Strategy for handling overlapping results
            buffer_window: Time window in seconds for result buffering
            max_buffer_size: Maximum number of results to buffer per channel
            confidence_threshold: Minimum confidence to include results
            priority_channel: Channel to prioritize ('left' or 'right')
        """
        self.merge_strategy = merge_strategy
        self.buffer_window = buffer_window
        self.max_buffer_size = max_buffer_size
        self.confidence_threshold = confidence_threshold
        self.priority_channel = priority_channel
        
        # Result buffers for each channel
        self.left_buffer: deque[BufferedResult] = deque(maxlen=max_buffer_size)
        self.right_buffer: deque[BufferedResult] = deque(maxlen=max_buffer_size)
        
        # Merged result queue
        self.output_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()
        
        # State management
        self.is_active = False
        self.merge_task: Optional[asyncio.Task] = None
        self.last_output_timestamp = 0.0
        
        # Statistics
        self.stats = MergeStatistics()
        self.start_time = 0.0
        
        # Channel tracking for speaker labeling
        self.left_speaker_label = "Speaker A"
        self.right_speaker_label = "Speaker B"
        
        logger.info(f"🔄 DualChannelResultMerger initialized:")
        logger.info(f"   📋 Strategy: {merge_strategy.value}")
        logger.info(f"   ⏰ Buffer window: {buffer_window}s")
        logger.info(f"   📊 Confidence threshold: {confidence_threshold}")
        logger.info(f"   🏆 Priority channel: {priority_channel}")
    
    async def start(self) -> None:
        """Start the result merger."""
        logger.info("🚀 Result Merger: Starting")
        
        self.is_active = True
        self.start_time = time.time()
        
        # Start the merge processing task
        self.merge_task = asyncio.create_task(self._process_merge_queue())
        
        logger.info("✅ Result Merger: Started successfully")
    
    async def add_left_result(self, result: TranscriptionResult) -> None:
        """Add a result from the left channel."""
        if not self.is_active:
            return
            
        # Filter by confidence
        if result.confidence < self.confidence_threshold:
            logger.debug(f"🚫 Left channel: Dropped low confidence result ({result.confidence:.2f})")
            self.stats.dropped_results += 1
            return
        
        # Update speaker labeling
        enhanced_result = TranscriptionResult(
            text=result.text,
            speaker_id=self.left_speaker_label,
            confidence=result.confidence,
            start_time=result.start_time,
            end_time=result.end_time,
            is_partial=result.is_partial,
            result_id=result.result_id,
            utterance_id=result.utterance_id,
            sequence_number=result.sequence_number
        )
        
        buffered_result = BufferedResult(
            result=enhanced_result,
            channel="left"
        )
        
        self.left_buffer.append(buffered_result)
        self.stats.total_left_results += 1
        
        logger.debug(f"📝 Left channel: Added result '{result.text}' (confidence: {result.confidence:.2f})")
    
    async def add_right_result(self, result: TranscriptionResult) -> None:
        """Add a result from the right channel."""
        if not self.is_active:
            return
            
        # Filter by confidence
        if result.confidence < self.confidence_threshold:
            logger.debug(f"🚫 Right channel: Dropped low confidence result ({result.confidence:.2f})")
            self.stats.dropped_results += 1
            return
        
        # Update speaker labeling
        enhanced_result = TranscriptionResult(
            text=result.text,
            speaker_id=self.right_speaker_label,
            confidence=result.confidence,
            start_time=result.start_time,
            end_time=result.end_time,
            is_partial=result.is_partial,
            result_id=result.result_id,
            utterance_id=result.utterance_id,
            sequence_number=result.sequence_number
        )
        
        buffered_result = BufferedResult(
            result=enhanced_result,
            channel="right"
        )
        
        self.right_buffer.append(buffered_result)
        self.stats.total_right_results += 1
        
        logger.debug(f"📝 Right channel: Added result '{result.text}' (confidence: {result.confidence:.2f})")
    
    async def get_merged_results(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Get merged results as they become available."""
        if not self.is_active:
            logger.warning("⚠️ Result Merger: Not active, cannot get results")
            return
        
        logger.info("🔊 Result Merger: Starting result output stream")
        
        try:
            while self.is_active or not self.output_queue.empty():
                try:
                    result = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                    yield result
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info("🛑 Result Merger: Result output cancelled")
        finally:
            logger.info("🛑 Result Merger: Result output stream stopped")
    
    async def _process_merge_queue(self) -> None:
        """Main merge processing loop."""
        try:
            logger.info("🔄 Result Merger: Starting merge processing")
            
            while self.is_active:
                current_time = time.time()
                
                # Process results based on strategy
                await self._process_buffered_results(current_time)
                
                # Clean up old results
                self._cleanup_old_results(current_time)
                
                # Log periodic statistics
                if int(current_time - self.start_time) % 30 == 0:  # Every 30 seconds
                    self._log_merge_statistics()
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
        except asyncio.CancelledError:
            logger.info("🛑 Result Merger: Merge processing cancelled")
        except Exception as e:
            logger.error(f"❌ Result Merger: Merge processing error: {e}")
        finally:
            logger.info("🔄 Result Merger: Merge processing stopped")
    
    async def _process_buffered_results(self, current_time: float) -> None:
        """Process buffered results according to merge strategy."""
        
        if self.merge_strategy == MergeStrategy.TIMESTAMP_ORDER:
            await self._process_timestamp_order(current_time)
        elif self.merge_strategy == MergeStrategy.CONFIDENCE_PRIORITY:
            await self._process_confidence_priority(current_time)
        elif self.merge_strategy == MergeStrategy.CHANNEL_PRIORITY:
            await self._process_channel_priority(current_time)
        elif self.merge_strategy == MergeStrategy.INTERLEAVE:
            await self._process_interleave(current_time)
    
    async def _process_timestamp_order(self, current_time: float) -> None:
        """Process results in timestamp order."""
        ready_results: List[Tuple[BufferedResult, str]] = []
        
        # Collect results ready for processing (outside buffer window)
        for result in self.left_buffer:
            if not result.processed and current_time - result.arrival_time > self.buffer_window:
                ready_results.append((result, "left"))
        
        for result in self.right_buffer:
            if not result.processed and current_time - result.arrival_time > self.buffer_window:
                ready_results.append((result, "right"))
        
        # Sort by timestamp
        ready_results.sort(key=lambda x: x[0].result.start_time)
        
        # Output results in timestamp order
        for buffered_result, channel in ready_results:
            await self._output_result(buffered_result.result)
            buffered_result.processed = True
    
    async def _process_confidence_priority(self, current_time: float) -> None:
        """Process results prioritizing higher confidence."""
        # Find overlapping time windows
        overlapping_groups = self._find_overlapping_results(current_time)
        
        for group in overlapping_groups:
            if len(group) == 1:
                # No conflict, output directly
                await self._output_result(group[0][0].result)
                group[0][0].processed = True
            else:
                # Multiple results, choose highest confidence
                best_result = max(group, key=lambda x: x[0].result.confidence)
                await self._output_result(best_result[0].result)
                
                # Mark all in group as processed
                for buffered_result, _ in group:
                    buffered_result.processed = True
                
                self.stats.conflicting_results += len(group) - 1
    
    async def _process_channel_priority(self, current_time: float) -> None:
        """Process results with channel priority."""
        if not self.priority_channel:
            # Fall back to timestamp order if no priority set
            await self._process_timestamp_order(current_time)
            return
        
        overlapping_groups = self._find_overlapping_results(current_time)
        
        for group in overlapping_groups:
            if len(group) == 1:
                await self._output_result(group[0][0].result)
                group[0][0].processed = True
            else:
                # Check if priority channel has a result in this group
                priority_results = [x for x in group if x[1] == self.priority_channel]
                
                if priority_results:
                    # Use priority channel result
                    await self._output_result(priority_results[0][0].result)
                else:
                    # Use highest confidence from available results
                    best_result = max(group, key=lambda x: x[0].result.confidence)
                    await self._output_result(best_result[0].result)
                
                # Mark all as processed
                for buffered_result, _ in group:
                    buffered_result.processed = True
                
                self.stats.conflicting_results += len(group) - 1
    
    async def _process_interleave(self, current_time: float) -> None:
        """Process results by interleaving channels."""
        # Simple interleaving: alternate between channels when both have results
        left_ready = [r for r in self.left_buffer 
                     if not r.processed and current_time - r.arrival_time > self.buffer_window]
        right_ready = [r for r in self.right_buffer 
                      if not r.processed and current_time - r.arrival_time > self.buffer_window]
        
        # Sort each channel by timestamp
        left_ready.sort(key=lambda x: x.result.start_time)
        right_ready.sort(key=lambda x: x.result.start_time)
        
        # Interleave results
        left_idx, right_idx = 0, 0
        while left_idx < len(left_ready) or right_idx < len(right_ready):
            
            if left_idx >= len(left_ready):
                # Only right results remaining
                await self._output_result(right_ready[right_idx].result)
                right_ready[right_idx].processed = True
                right_idx += 1
            elif right_idx >= len(right_ready):
                # Only left results remaining
                await self._output_result(left_ready[left_idx].result)
                left_ready[left_idx].processed = True
                left_idx += 1
            else:
                # Both have results, alternate
                if (left_idx + right_idx) % 2 == 0:
                    await self._output_result(left_ready[left_idx].result)
                    left_ready[left_idx].processed = True
                    left_idx += 1
                else:
                    await self._output_result(right_ready[right_idx].result)
                    right_ready[right_idx].processed = True
                    right_idx += 1
    
    def _find_overlapping_results(self, current_time: float) -> List[List[Tuple[BufferedResult, str]]]:
        """Find groups of overlapping results from both channels."""
        ready_results: List[Tuple[BufferedResult, str]] = []
        
        # Collect ready results
        for result in self.left_buffer:
            if not result.processed and current_time - result.arrival_time > self.buffer_window:
                ready_results.append((result, "left"))
        
        for result in self.right_buffer:
            if not result.processed and current_time - result.arrival_time > self.buffer_window:
                ready_results.append((result, "right"))
        
        if not ready_results:
            return []
        
        # Sort by start time
        ready_results.sort(key=lambda x: x[0].result.start_time)
        
        # Group overlapping results
        groups = []
        current_group = [ready_results[0]]
        
        for i in range(1, len(ready_results)):
            current_result = ready_results[i][0].result
            last_group_result = current_group[-1][0].result
            
            # Check if results overlap (allowing small gap tolerance)
            gap_tolerance = 0.5  # seconds
            if (current_result.start_time <= last_group_result.end_time + gap_tolerance):
                current_group.append(ready_results[i])
            else:
                # No overlap, start new group
                groups.append(current_group)
                current_group = [ready_results[i]]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _output_result(self, result: TranscriptionResult) -> None:
        """Output a merged result."""
        # Update timestamp tracking
        if result.start_time > self.last_output_timestamp:
            self.last_output_timestamp = result.start_time
        else:
            # Timestamp correction needed
            result.start_time = self.last_output_timestamp + 0.01
            self.stats.timestamp_corrections += 1
        
        await self.output_queue.put(result)
        self.stats.merged_results += 1
        
        logger.debug(f"🔄 Merged result: {result.speaker_id}: '{result.text}' "
                    f"(confidence: {result.confidence:.2f}, time: {result.start_time:.2f}s)")
    
    def _cleanup_old_results(self, current_time: float) -> None:
        """Clean up processed results from buffers."""
        # Clean left buffer
        while (self.left_buffer and 
               self.left_buffer[0].processed and 
               current_time - self.left_buffer[0].arrival_time > self.buffer_window * 2):
            self.left_buffer.popleft()
        
        # Clean right buffer  
        while (self.right_buffer and 
               self.right_buffer[0].processed and 
               current_time - self.right_buffer[0].arrival_time > self.buffer_window * 2):
            self.right_buffer.popleft()
    
    def _log_merge_statistics(self) -> None:
        """Log periodic merge statistics."""
        runtime = time.time() - self.start_time
        
        logger.info(f"🔄 Merge Statistics (runtime: {runtime:.1f}s):")
        logger.info(f"   📊 Input: Left={self.stats.total_left_results}, Right={self.stats.total_right_results}")
        logger.info(f"   📤 Output: Merged={self.stats.merged_results}, Dropped={self.stats.dropped_results}")
        logger.info(f"   ⚔️  Conflicts: {self.stats.conflicting_results}")
        logger.info(f"   📋 Buffer Status: Left={len(self.left_buffer)}, Right={len(self.right_buffer)}")
        logger.info(f"   🔧 Timestamp Corrections: {self.stats.timestamp_corrections}")
    
    async def stop(self) -> None:
        """Stop the result merger."""
        logger.info("🛑 Result Merger: Stopping")
        
        try:
            self.is_active = False
            
            # Cancel merge task
            if self.merge_task and not self.merge_task.done():
                self.merge_task.cancel()
                try:
                    await self.merge_task
                except asyncio.CancelledError:
                    pass
            
            # Process any remaining buffered results
            await self._flush_remaining_results()
            
            # Log final statistics
            self._log_final_statistics()
            
            logger.info("✅ Result Merger: Stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Result Merger: Error during stop: {e}")
    
    async def _flush_remaining_results(self) -> None:
        """Flush any remaining results in buffers."""
        logger.info("🔄 Result Merger: Flushing remaining results")
        
        remaining_results: List[Tuple[BufferedResult, str]] = []
        
        # Collect all unprocessed results
        for result in self.left_buffer:
            if not result.processed:
                remaining_results.append((result, "left"))
        
        for result in self.right_buffer:
            if not result.processed:
                remaining_results.append((result, "right"))
        
        # Sort by timestamp and output
        remaining_results.sort(key=lambda x: x[0].result.start_time)
        
        for buffered_result, channel in remaining_results:
            await self._output_result(buffered_result.result)
            buffered_result.processed = True
        
        if remaining_results:
            logger.info(f"🔄 Result Merger: Flushed {len(remaining_results)} remaining results")
    
    def _log_final_statistics(self) -> None:
        """Log final merge statistics."""
        runtime = time.time() - self.start_time
        
        logger.info(f"📊 Final Merge Statistics:")
        logger.info(f"   ⏱️  Total Runtime: {runtime:.1f}s")
        logger.info(f"   📊 Total Input: {self.stats.total_left_results + self.stats.total_right_results} results")
        logger.info(f"   📤 Total Output: {self.stats.merged_results} merged results")
        logger.info(f"   🚫 Dropped Results: {self.stats.dropped_results}")
        logger.info(f"   ⚔️  Conflicting Results: {self.stats.conflicting_results}")
        logger.info(f"   🔧 Timestamp Corrections: {self.stats.timestamp_corrections}")
        logger.info(f"   📋 Final Strategy: {self.merge_strategy.value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current merger statistics."""
        return {
            'total_left_results': self.stats.total_left_results,
            'total_right_results': self.stats.total_right_results,
            'merged_results': self.stats.merged_results,
            'conflicting_results': self.stats.conflicting_results,
            'dropped_results': self.stats.dropped_results,
            'timestamp_corrections': self.stats.timestamp_corrections,
            'left_buffer_size': len(self.left_buffer),
            'right_buffer_size': len(self.right_buffer),
            'merge_strategy': self.merge_strategy.value,
            'buffer_window': self.buffer_window,
            'is_active': self.is_active
        }