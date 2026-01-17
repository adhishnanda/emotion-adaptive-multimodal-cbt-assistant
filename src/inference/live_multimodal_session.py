"""Live multimodal session manager for real-time emotion detection and CBT responses."""

from typing import Optional, Dict, List, Tuple
from datetime import datetime
import torch

from src.inference.live_audio_stream import LiveAudioStream
from src.inference.live_video_stream import LiveVideoStream
from src.inference.multimodal_iemocap_inference import IEMOCAPMultimodalInference
from src.cbt.cbt_rules import generate_cbt_response
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LiveMultimodalSession:
    """Manages a live multimodal session with real-time emotion detection."""
    
    def __init__(
        self,
        fusion_model: IEMOCAPMultimodalInference,
        audio_encoder=None,  # Audio encoder from fusion model
        video_encoder=None,  # Video encoder from fusion model
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        image_size: int = 224,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize live multimodal session.
        
        Args:
            fusion_model: IEMOCAP multimodal fusion model
            audio_encoder: Audio encoder (from fusion model)
            video_encoder: Video encoder (from fusion model)
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            image_size: Video image size
            device: Device to run inference on
        """
        self.fusion_model = fusion_model
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.device = device if device is not None else torch.device("cpu")
        
        # Initialize stream processors
        self.audio_stream = LiveAudioStream(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            device=device,
        )
        
        self.video_stream = LiveVideoStream(
            image_size=image_size,
            device=device,
        )
        
        # Session state
        self.is_active = False
        self.current_text = ""
        self.history: List[Dict] = []
        
        # Extract encoders from fusion model if not provided
        if self.audio_encoder is None and hasattr(fusion_model, 'fusion_model'):
            self.audio_encoder = fusion_model.fusion_model.audio_model
        
        if self.video_encoder is None and hasattr(fusion_model, 'fusion_model'):
            self.video_encoder = fusion_model.fusion_model.video_model
    
    def start_session(self):
        """Start the live session."""
        self.is_active = True
        self.audio_stream.clear_history()
        self.video_stream.clear_history()
        self.history = []
        logger.info("Live multimodal session started")
    
    def stop_session(self):
        """Stop the live session."""
        self.is_active = False
        self.audio_stream.clear_history()
        self.video_stream.clear_history()
        logger.info("Live multimodal session stopped")
    
    def update_text(self, text: str):
        """Update the current text input."""
        self.current_text = text
    
    def process_audio_chunk(
        self,
        audio_waveform: torch.Tensor,
        input_sample_rate: int,
    ) -> bool:
        """
        Process an audio chunk.
        
        Args:
            audio_waveform: Audio waveform tensor
            input_sample_rate: Sample rate of input audio
            
        Returns:
            True if processing succeeded, False otherwise
        """
        if not self.is_active:
            return False
        
        mfcc = self.audio_stream.process_audio_chunk(audio_waveform, input_sample_rate)
        return mfcc is not None
    
    def process_video_frame(self, image) -> bool:
        """
        Process a video frame.
        
        Args:
            image: PIL Image
            
        Returns:
            True if processing succeeded, False otherwise
        """
        if not self.is_active:
            return False
        
        frame = self.video_stream.process_frame(image)
        return frame is not None
    
    def get_current_emotion(
        self,
        use_fusion: bool = True,
    ) -> Optional[Dict]:
        """
        Get current emotion prediction from available modalities.
        
        Args:
            use_fusion: Whether to use fusion model (True) or individual models (False)
            
        Returns:
            Dictionary with emotion prediction results or None if no modalities available
        """
        if not self.is_active:
            return None
        
        # Get available modalities
        has_text = bool(self.current_text and self.current_text.strip())
        has_audio = self.audio_stream.has_audio()
        has_video = self.video_stream.has_video()
        
        if not has_text and not has_audio and not has_video:
            return None
        
        try:
            if use_fusion and self.fusion_model and hasattr(self.fusion_model, 'fusion_model'):
                # Use fusion model directly with embeddings
                fusion_model = self.fusion_model.fusion_model
                
                # Prepare inputs
                text_input_ids = None
                text_attention_mask = None
                if has_text:
                    text_input_ids, text_attention_mask = self.fusion_model.preprocess_text(
                        self.current_text
                    )
                    text_input_ids = text_input_ids.to(self.device)
                    text_attention_mask = text_attention_mask.to(self.device)
                
                # Get audio MFCC if available
                audio_mfcc = None
                if has_audio:
                    # Get recent MFCC features
                    mfcc = self.audio_stream.get_recent_mfcc(max_time_frames=200)
                    if mfcc is not None:
                        audio_mfcc = mfcc.unsqueeze(0).to(self.device)
                
                # Get video frame if available
                video_frames = None
                if has_video and self.video_stream.frame_history:
                    # Get the most recent frame
                    recent_frame = self.video_stream.frame_history[-1]
                    video_frames = recent_frame.unsqueeze(0).to(self.device)
                
                # Forward pass through fusion model
                with torch.no_grad():
                    loss, logits = fusion_model(
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        audio_mfcc=audio_mfcc,
                        video_frames=video_frames,
                        labels=None,
                    )
                
                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get prediction
                pred_id = logits.argmax(dim=-1).item()
                pred_emotion = self.fusion_model.id_to_emotion[pred_id]
                pred_confidence = probs[0, pred_id].item()
                
                return {
                    "emotion": pred_emotion,
                    "confidence": pred_confidence,
                    "logits": logits.cpu(),
                    "probs": probs.cpu(),
                    "available_modalities": {
                        "text": has_text,
                        "audio": has_audio,
                        "video": has_video,
                    },
                    "method": "fusion",
                }
            else:
                # Fallback: return None if fusion model not available
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get current emotion: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def get_cbt_response(
        self,
        emotion_label: str,
        confidence: float,
        user_text: str,
        depression_prob: Optional[float] = None,
    ) -> Dict:
        """
        Get CBT response based on detected emotion.
        
        Args:
            emotion_label: Detected emotion label
            confidence: Confidence score
            user_text: User's text input
            depression_prob: Optional depression probability
            
        Returns:
            Dictionary with CBT response
        """
        try:
            cbt_result = generate_cbt_response(
                user_text=user_text,
                emotion_label=emotion_label,
                depression_risk=depression_prob,
            )
            return cbt_result
        except Exception as e:
            logger.warning(f"Failed to generate CBT response: {e}")
            return {"response": "", "steps": []}
    
    def update_history(
        self,
        emotion_result: Dict,
        cbt_response: Dict,
    ):
        """
        Update session history with new results.
        
        Args:
            emotion_result: Emotion prediction result
            cbt_response: CBT response result
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_result.get("emotion"),
            "confidence": emotion_result.get("confidence"),
            "available_modalities": emotion_result.get("available_modalities", {}),
            "cbt_response": cbt_response.get("response", ""),
            "cbt_steps": cbt_response.get("steps", []),
        }
        self.history.append(entry)
        
        # Keep only recent history (e.g., last 100 entries)
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_recent_history(self, n: int = 10) -> List[Dict]:
        """
        Get recent history entries.
        
        Args:
            n: Number of recent entries to return
            
        Returns:
            List of recent history entries
        """
        return self.history[-n:] if len(self.history) > n else self.history


__all__ = ["LiveMultimodalSession"]

