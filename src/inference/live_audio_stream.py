"""Real-time audio streaming and processing for live multimodal inference."""

from typing import Optional, Deque
from collections import deque
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LiveAudioStream:
    """Manages real-time audio streaming and embedding extraction."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        chunk_duration: float = 3.0,  # seconds per chunk
        history_duration: float = 10.0,  # seconds of history to maintain
        device: Optional[torch.device] = None,
    ):
        """
        Initialize live audio stream processor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            chunk_duration: Duration of each audio chunk in seconds
            history_duration: Duration of audio history to maintain
            device: Device to process audio on
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.chunk_duration = chunk_duration
        self.history_duration = history_duration
        self.device = device if device is not None else torch.device("cpu")
        
        # Audio history: store MFCC features
        self.mfcc_history: Deque[torch.Tensor] = deque(maxlen=int(history_duration / chunk_duration))
        
        # MFCC transform
        self.mfcc_transform = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
        
        # Resampler cache (will be created on first use with specific sample rate)
        self._resamplers = {}
    
    def process_audio_chunk(
        self,
        audio_waveform: torch.Tensor,
        input_sample_rate: int,
    ) -> Optional[torch.Tensor]:
        """
        Process a single audio chunk and extract MFCC features.
        
        Args:
            audio_waveform: Audio waveform tensor of shape (channels, samples) or (samples,)
            input_sample_rate: Sample rate of the input audio
            
        Returns:
            MFCC features tensor of shape (n_mfcc, time) or None if processing fails
        """
        try:
            # Ensure waveform is on CPU for processing
            if audio_waveform.is_cuda:
                audio_waveform = audio_waveform.cpu()
            
            # Handle shape: ensure (channels, samples)
            if audio_waveform.dim() == 1:
                audio_waveform = audio_waveform.unsqueeze(0)
            
            # Convert to mono if stereo
            if audio_waveform.shape[0] > 1:
                audio_waveform = audio_waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if input_sample_rate != self.sample_rate:
                if input_sample_rate not in self._resamplers:
                    self._resamplers[input_sample_rate] = torchaudio.transforms.Resample(
                        input_sample_rate,
                        self.sample_rate,
                    )
                resampler = self._resamplers[input_sample_rate]
                audio_waveform = resampler(audio_waveform)
            
            # Extract MFCC
            mfcc = self.mfcc_transform(audio_waveform)
            
            # Remove channel dimension if present
            if mfcc.dim() == 3:
                mfcc = mfcc.squeeze(0)
            
            # Add to history
            self.mfcc_history.append(mfcc)
            
            return mfcc
            
        except Exception as e:
            logger.warning(f"Failed to process audio chunk: {e}")
            return None
    
    def get_recent_mfcc(
        self,
        max_time_frames: int = 200,
    ) -> Optional[torch.Tensor]:
        """
        Get recent MFCC features from audio history.
        
        Args:
            max_time_frames: Maximum time frames to use (for padding/truncation)
            
        Returns:
            MFCC features tensor of shape (n_mfcc, time) or None if no audio available
        """
        if not self.mfcc_history:
            return None
        
        try:
            # Concatenate recent MFCC features
            mfcc_list = list(self.mfcc_history)
            
            # Concatenate along time dimension
            if len(mfcc_list) == 1:
                mfcc_combined = mfcc_list[0]
            else:
                mfcc_combined = torch.cat(mfcc_list, dim=1)
            
            # Truncate or pad to max_time_frames
            current_time = mfcc_combined.shape[1]
            if current_time > max_time_frames:
                mfcc_combined = mfcc_combined[:, :max_time_frames]
            elif current_time < max_time_frames:
                padding = max_time_frames - current_time
                mfcc_combined = torch.nn.functional.pad(mfcc_combined, (0, padding))
            
            return mfcc_combined
            
        except Exception as e:
            logger.warning(f"Failed to get recent MFCC: {e}")
            return None
    
    def clear_history(self):
        """Clear audio history."""
        self.mfcc_history.clear()
    
    def has_audio(self) -> bool:
        """Check if there is any audio history."""
        return len(self.mfcc_history) > 0


__all__ = ["LiveAudioStream"]

