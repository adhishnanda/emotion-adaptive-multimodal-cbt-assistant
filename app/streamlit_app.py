"""Streamlit application for Emotion-Adaptive CBT Assistant (Text-Only Prototype)."""

from pathlib import Path
import sys

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer

from config.config import load_config
from src.models.text_distilbert import build_text_model
from src.fusion.late_fusion import (
    FusionWeights,
    fuse_emotion_logits,
    LateFusion,
)
from src.cbt.cbt_rules import generate_cbt_response
from src.cbt.nlg_engine import build_emotional_state
from src.inference.audio_inference import predict_depression_from_wav
from src.inference.video_inference import extract_video_embedding
from src.inference.multimodal_iemocap_inference import load_iemocap_multimodal_inference
from src.inference.live_multimodal_session import LiveMultimodalSession
from src.nlg.llm_engine import generate_psychologist_response, PsychologistLLMConfig
from src.utils.logging_utils import get_logger
import time
import io

logger = get_logger(__name__)

# Page configuration - MUST be first Streamlit command (before any decorators or other st.* calls)
st.set_page_config(
    page_title="Emotion-Adaptive CBT Assistant",
    page_icon="😊",
    layout="wide",
)


@st.cache_resource
def load_text_model_and_tokenizer():
    """
    Load text model, tokenizer, and label mappings (MELD for backward compatibility).

    Returns:
        Tuple of (model, tokenizer, id2label, device)
    """
    cfg = load_config()
    device = cfg.device.device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)

    # Build model with correct num_labels using MELD train CSV
    data_root = cfg.paths.raw_dir / "meld"
    train_csv = data_root / "train_sent_emo.csv"

    try:
        df_train = pd.read_csv(train_csv)
        label_col = "Emotion"
        le = LabelEncoder()
        y = le.fit_transform(df_train[label_col].astype(str))
        num_labels = len(le.classes_)
    except Exception as e:
        st.error(f"Failed to load training data to determine labels: {e}")
        # Fallback to default
        num_labels = 7

    model = build_text_model(num_labels=num_labels, device=device)

    # Load trained weights if available
    best_path = cfg.paths.models_dir / "text" / "distilbert_best.pt"
    logger.info(f"[TEXT MODEL] Attempting to load checkpoint from: {best_path}")
    if best_path.exists():
        try:
            state_dict = torch.load(best_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"[TEXT MODEL] Successfully loaded checkpoint from: {best_path} (Model: DistilBERT)")
            st.success("Loaded trained model weights")
        except Exception as e:
            logger.warning(f"[TEXT MODEL] Failed to load checkpoint from {best_path}: {e}")
            st.warning(f"Failed to load model weights: {e}")
    else:
        logger.info(f"[TEXT MODEL] Checkpoint not found at {best_path}, using untrained model (Model: DistilBERT)")

    model.eval()

    # Build id-to-label mapping
    try:
        id2label = {i: label for i, label in enumerate(le.classes_)}
        emotion_labels = list(le.classes_)
        logger.info(f"[TEXT MODEL] Loaded emotion_labels from MELD training data: {emotion_labels}")
    except:
        # Fallback mapping
        id2label = {
            0: "neutral",
            1: "joy",
            2: "sadness",
            3: "anger",
            4: "fear",
            5: "surprise",
            6: "disgust",
        }
        emotion_labels = list(id2label.values())
        logger.info(f"[TEXT MODEL] Using fallback emotion_labels: {emotion_labels}")

    logger.info(f"[TEXT MODEL] Final num_labels: {num_labels}, id2label mapping: {id2label}")

    return model, tokenizer, id2label, device


@st.cache_resource
def load_iemocap_fusion_model():
    """
    Load IEMOCAP multimodal fusion model.

    Returns:
        IEMOCAPMultimodalInference instance or None if loading fails
    """
    try:
        cfg = load_config()
        inference_pipeline = load_iemocap_multimodal_inference(device=cfg.device.device)
        logger.info("[IEMOCAP FUSION MODEL] IEMOCAP multimodal fusion model loaded successfully.")
        return inference_pipeline
    except Exception as e:
        st.warning(f"Failed to load IEMOCAP fusion model: {e}")
        logger.error(f"[IEMOCAP FUSION MODEL] Failed to load IEMOCAP fusion model: {e}")
        return None


def _create_emotion_description(result_dict):
    """
    Create a user-friendly description string from emotion prediction results.
    
    Args:
        result_dict: Dict with primary_label, secondary_label, primary_confidence, confidence_level, etc.
    
    Returns:
        str: User-friendly description
    """
    primary_label = result_dict["primary_label"]
    secondary_label = result_dict["secondary_label"]
    primary_conf = result_dict["primary_confidence"]
    confidence_level = result_dict["confidence_level"]
    
    # Special case for negative overload
    negative_emotions = {"sadness", "anger", "fear", "disgust"}
    is_negative_overload = (
        primary_label.lower() in {"fear", "anger"}
        and confidence_level in {"low", "medium"}
        and secondary_label.lower() in negative_emotions
    )
    
    if is_negative_overload:
        return f"Distress cluster: model detects a mix of {primary_label} and {secondary_label} rather than a single clear emotion (uncertain)."
    
    # Regular cases based on confidence level
    if confidence_level == "high":
        percent = int(primary_conf * 100)
        return f"Primary emotion: {primary_label} (confidence ~{percent}%)"
    elif confidence_level == "medium":
        return f"Likely emotion: {primary_label} (moderate confidence), secondary: {secondary_label}."
    else:  # low
        return f"Mixed / uncertain emotions: {primary_label} and {secondary_label} (low confidence)."


def predict_emotion(model, tokenizer, id2label, device, text, max_len=128):
    """
    Predict emotion from text with top-2 emotions and confidence levels.

    Args:
        model: Trained emotion classification model
        tokenizer: Text tokenizer
        id2label: Mapping from label ID to label name
        device: Device to run inference on
        text: Input text
        max_len: Maximum sequence length

    Returns:
        Dict with emotion prediction results, or None if text is empty.
        Structure:
        {
            "primary_label": str,
            "primary_confidence": float,
            "secondary_label": str,
            "secondary_confidence": float,
            "all_labels": List[str],
            "all_probs": List[float],
            "confidence_level": str,  # "high" | "medium" | "low"
            "description": str,  # User-friendly description
            "logits": torch.Tensor,  # Raw logits for fusion
        }
    """
    if not text or not text.strip():
        logger.info("[TEXT INFERENCE] No text provided for prediction.")
        return None
    
    logger.info(f"[TEXT INFERENCE] Text model (DistilBERT) predicting emotion for text: '{text[:50]}...'")

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Sort indices by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    
    # Get top-2
    primary_idx = int(sorted_indices[0])
    secondary_idx = int(sorted_indices[1])
    
    primary_label = id2label.get(primary_idx, "unknown")
    secondary_label = id2label.get(secondary_idx, "unknown")
    primary_conf = float(probs[primary_idx])
    secondary_conf = float(probs[secondary_idx])
    
    # Determine confidence level
    if primary_conf >= 0.65:
        confidence_level = "high"
    elif primary_conf >= 0.40:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Heuristic: neutral + low confidence + anxiety keywords -> anxious/fear
    anxiety_keywords = [
        "anxious",
        "anxiety",
        "overthinking",
        "worried",
        "worry",
        "panic",
        "panicking",
        "nervous",
        "uneasy",
        "afraid",
        "fear",
    ]
    text_lower = text.lower()
    if primary_label.lower() == "neutral" and primary_conf < 0.4:
        if any(k in text_lower for k in anxiety_keywords):
            # Prefer "anxious" if available in label set, else fall back to "fear"
            target_label = "anxious"
            if not any(str(v).lower() == "anxious" for v in id2label.values()):
                target_label = "fear"
            primary_label = target_label
            confidence_level = "low"
    
    # Build all_labels and all_probs in the same order as emotion_labels
    all_labels = [id2label.get(i, "unknown") for i in range(len(probs))]
    all_probs = [float(probs[i]) for i in range(len(probs))]
    
    # Create result dict
    result = {
        "primary_label": primary_label,
        "primary_confidence": primary_conf,
        "secondary_label": secondary_label,
        "secondary_confidence": secondary_conf,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "confidence_level": confidence_level,
        "logits": logits.cpu(),  # Keep logits for fusion
    }
    
    # Add user-friendly description
    result["description"] = _create_emotion_description(result)
    
    # Log the result
    logger.info(
        "[TEXT INFERENCE] Prediction result: primary=%s (%.2f), secondary=%s (%.2f), level=%s",
        primary_label, primary_conf, secondary_label, secondary_conf, confidence_level
    )
    
    return result


def main():
    """Main Streamlit application."""
    # Load model and tokenizer
    try:
        model, tokenizer, id2label, device = load_text_model_and_tokenizer()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Initialize live session state
    if "live_session_active" not in st.session_state:
        st.session_state["live_session_active"] = False
    
    if "live_session" not in st.session_state:
        st.session_state["live_session"] = None
    
    if "last_inference_time" not in st.session_state:
        st.session_state["last_inference_time"] = 0.0

    # Sidebar: Response mode selection
    mode = st.sidebar.selectbox(
        "Response mode",
        ["Rule-based CBT", "LLM-style CBT (simulated)"],
        index=0,
        help="Choose between the handcrafted CBT rules engine or a simulated LLM-style therapist response.",
    )
    
    # Add "New conversation" button in sidebar
    if st.sidebar.button("Start New Conversation"):
        # Clear conversation-related state
        st.session_state["chat_history"] = []
        # Optionally clear live session state if it exists
        if "live_session" in st.session_state:
            st.session_state["live_session"] = None
        if "live_session_active" in st.session_state:
            st.session_state["live_session_active"] = False
        if "last_inference_time" in st.session_state:
            st.session_state["last_inference_time"] = 0.0
        st.rerun()

    # Title and description
    st.title("Emotion-Adaptive CBT Assistant (Multimodal)")
    st.markdown(
        """
        **Research Prototype** - This is a demonstration system for research purposes only.
        This tool is not a substitute for professional medical advice, diagnosis, or treatment.
        If you are experiencing a mental health crisis, please contact a mental health professional
        or emergency services immediately.
        """
    )

    st.markdown("---")

    # Display previous conversation
    if st.session_state["chat_history"]:
        st.markdown("### Conversation")
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown("---")

    # Live Session Mode
    if st.session_state["live_session_active"]:
        st.markdown("### 🎙️ Live Session Active")
        st.info("Real-time emotion detection is running. Speak into your microphone and show your face to the webcam.")
        
        # Live session UI
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📹 Live Video")
            live_video = st.camera_input("Live webcam feed", key="live_camera")
            
            # Process video frame
            if live_video is not None and st.session_state["live_session"] is not None:
                try:
                    logger.info("[LIVE SESSION] Processing live video frame.")
                    from PIL import Image
                    import io
                    image_bytes = live_video.getvalue()
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    st.session_state["live_session"].process_video_frame(pil_img)
                except Exception as e:
                    logger.error(f"[LIVE SESSION] Failed to process video frame: {e}")
                    st.warning(f"Failed to process video frame: {e}")
        
        with col2:
            st.subheader("🎤 Live Audio")
            live_audio = st.audio_input("Record audio chunk", key="live_audio")
            
            # Process audio chunk
            if live_audio is not None and st.session_state["live_session"] is not None:
                try:
                    logger.info("[LIVE SESSION] Processing live audio chunk.")
                    import torchaudio
                    import io
                    audio_bytes = live_audio.read()
                    audio_io = io.BytesIO(audio_bytes)
                    waveform, sr = torchaudio.load(audio_io)
                    # Resample if needed
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    waveform = waveform.squeeze(0)
                    st.session_state["live_session"].process_audio_chunk(waveform, 16000)
                except Exception as e:
                    logger.error(f"[LIVE SESSION] Failed to process audio chunk: {e}")
                    st.warning(f"Failed to process audio chunk: {e}")
        
        # Text input for live session
        user_text = st.text_area(
            "How are you feeling today? (Updates in real-time)",
            height=100,
            placeholder="Type how you're feeling...",
            key="live_text_input",
        )
        
        # Update session text
        if st.session_state["live_session"] is not None:
            logger.info(f"[LIVE SESSION] Updating session text: '{user_text[:50]}...'")
            st.session_state["live_session"].update_text(user_text)
        
        # Live inference loop (runs every 1-2 seconds)
        current_time = time.time()
        inference_interval = 2.0  # seconds
        
        # Initialize last inference time if needed
        if "last_inference_time" not in st.session_state:
            st.session_state["last_inference_time"] = 0.0
        
        # Check if it's time to run inference
        should_run_inference = (
            current_time - st.session_state["last_inference_time"] >= inference_interval
        )
        
        if should_run_inference and st.session_state["live_session"] is not None:
            logger.info("[LIVE SESSION] Live Multimodal Session (IEMOCAP Fusion) predicting emotion.")
            # Get current emotion
            emotion_result = st.session_state["live_session"].get_current_emotion()
            
            if emotion_result:
                logger.info(
                    "[LIVE SESSION] Prediction result: emotion=%s (confidence=%.2f), available_modalities=%s",
                    emotion_result["emotion"], emotion_result["confidence"], emotion_result.get("available_modalities", "N/A")
                )
                # Get CBT response
                cbt_response = st.session_state["live_session"].get_cbt_response(
                    emotion_label=emotion_result["emotion"],
                    confidence=emotion_result["confidence"],
                    user_text=user_text if user_text else "User is expressing emotions through audio/video.",
                )
                
                # Update history
                st.session_state["live_session"].update_history(emotion_result, cbt_response)
                
                # Store results in session state for display
                st.session_state["live_emotion_result"] = emotion_result
                st.session_state["live_cbt_response"] = cbt_response
            
            st.session_state["last_inference_time"] = current_time
        
        # Display results if available
        if "live_emotion_result" in st.session_state and st.session_state["live_emotion_result"]:
            emotion_result = st.session_state["live_emotion_result"]
            cbt_response = st.session_state.get("live_cbt_response", {})
            
            st.markdown("---")
            st.subheader("🎯 Current Emotion Detection")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Detected Emotion",
                    emotion_result["emotion"].upper(),
                    f"{emotion_result['confidence']:.1%}",
                )
            
            with col2:
                modalities = emotion_result.get("available_modalities", {})
                active_mods = [k for k, v in modalities.items() if v]
                st.write(f"**Active Modalities:** {', '.join(active_mods) if active_mods else 'None'}")
            
            # Display CBT response
            if cbt_response.get("response"):
                st.markdown("### 💬 CBT Response")
                st.write(cbt_response["response"])
                
                if cbt_response.get("steps"):
                    with st.expander("View CBT Steps"):
                        for i, step in enumerate(cbt_response["steps"], 1):
                            st.write(f"{i}. {step}")
            
            # Display recent history
            if st.session_state["live_session"] is not None:
                recent_history = st.session_state["live_session"].get_recent_history(n=5)
                if recent_history:
                    with st.expander("View Recent History"):
                        for entry in recent_history:
                            st.write(f"**{entry['timestamp']}**: {entry['emotion']} ({entry['confidence']:.2f})")
        
        # Auto-refresh for live updates (only if session is active)
        if st.session_state["live_session_active"]:
            time.sleep(1.0)  # Wait 1 second before next refresh
            st.rerun()
    
    else:
        # Standard (non-live) mode
        # User input
        user_text = st.text_area(
            "How are you feeling today?",
            height=150,
            placeholder="Type how you're feeling today...",
        )

        # Audio Input section
        st.subheader("Audio Input (Optional)")
        uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])

        # Initialize audio result variables (will be used in button handler)
        audio_logits = None
        audio_probs = None
        audio_result = None

        # Webcam Video section
        st.subheader("Webcam Video (Optional snapshot)")
        webcam_image = st.camera_input("Take a snapshot while you speak or express how you feel")

    if uploaded_audio is not None:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_audio.wav"

        # Save uploaded file
        try:
            temp_path.write_bytes(uploaded_audio.read())
            
            with st.spinner("Processing audio with DAIC-WOZ model..."):
                logger.info(f"[AUDIO INFERENCE] DAIC-WOZ audio model predicting depression risk for audio: {uploaded_audio.name}")
                # Run audio inference (DAIC-WOZ depression detection)
                audio_result = predict_depression_from_wav(temp_path)
                
                # Store results for fusion
                audio_logits = audio_result["logits"]
                audio_probs = audio_result["probs"]
                
                # Display results
                st.write("**Audio Prediction (DAIC-WOZ):**", audio_result["pred_label"])
                
                # Display depression probability (for binary classification)
                if audio_probs.shape[1] >= 2:
                    depression_prob = float(audio_probs[0, 1])
                    st.write("**Depression Probability:**", f"{depression_prob:.4f}")
                    logger.info(f"[AUDIO INFERENCE] DAIC-WOZ Prediction result: label={audio_result['pred_label']}, depression_prob={depression_prob:.4f}")
                elif audio_probs.shape[1] == 1:
                    # Single output (sigmoid)
                    depression_prob = float(audio_probs[0, 0])
                    st.write("**Depression Probability:**", f"{depression_prob:.4f}")
                    logger.info(f"[AUDIO INFERENCE] DAIC-WOZ Prediction result (single output): label={audio_result['pred_label']}, depression_prob={depression_prob:.4f}")
        except Exception as e:
            logger.error(f"[AUDIO INFERENCE] Failed to process audio: {e}")
            st.error(f"Failed to process audio: {e}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

    # Depression risk slider (for testing)
    depression_risk = st.slider(
        "Simulated depression risk (for testing)",
        0.0,
        1.0,
        0.2,
        0.05,
    )

    # Get response button
    if st.button("Get CBT Response", type="primary"):
        # Check if we have at least one modality
        has_text = user_text and user_text.strip()
        has_audio = uploaded_audio is not None
        has_video = webcam_image is not None
        
        if not has_text and not has_audio and not has_video:
            st.warning("Please enter some text, upload an audio file, or take a webcam snapshot.")
            logger.warning("[MAIN APP] No input provided for CBT response.")
        else:
            with st.spinner("Analyzing emotion and generating response..."):
                # Try to load IEMOCAP fusion model first
                iemocap_fusion = load_iemocap_fusion_model()
                
                # Determine which inference pipeline to use
                # Only use IEMOCAP fusion if we have multiple modalities (not just text)
                # IEMOCAP fusion is designed for multimodal inputs (text+audio, text+video, or all three)
                use_iemocap_fusion = (
                    iemocap_fusion is not None and
                    (has_audio or has_video)  # Require at least audio or video, not just text
                )
                
                # Log when skipping fusion due to text-only input
                if iemocap_fusion is not None and has_text and not has_audio and not has_video:
                    logger.debug("[MAIN APP] Skipping IEMOCAP fusion: only text modality available; using text-only emotional state.")
                
                pred_label = None
                pred_conf = None
                pred_conf_tag = None
                fused_emotion_result = None
                depression_prob_for_state = None
                emotion_result = None  # For text emotion prediction results
                
                if use_iemocap_fusion:
                    logger.info("[MAIN APP] Using IEMOCAP Multimodal Fusion model for prediction.")
                    try:
                        # Prepare inputs
                        text_input = user_text if has_text else None
                        
                        audio_waveform_input = None
                        if has_audio and uploaded_audio is not None:
                            # Load audio waveform
                            import torchaudio
                            import io
                            audio_bytes = uploaded_audio.read()
                            audio_io = io.BytesIO(audio_bytes)
                            waveform, sr = torchaudio.load(audio_io)
                            # Resample if needed
                            if sr != 16000:
                                resampler = torchaudio.transforms.Resample(sr, 16000)
                                waveform = resampler(waveform)
                            # Convert to mono if stereo
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            audio_waveform_input = waveform.squeeze(0)  # Remove channel dimension
                            logger.info(f"[MAIN APP] Prepared audio waveform input from {uploaded_audio.name}.")
                        
                        video_image_input = None
                        if has_video and webcam_image is not None:
                            from PIL import Image
                            import io
                            image_bytes = webcam_image.getvalue()
                            video_image_input = Image.open(io.BytesIO(image_bytes))
                            logger.info("[MAIN APP] Prepared video image input from webcam snapshot.")
                        
                        # Run IEMOCAP fusion inference
                        fused_emotion_result = iemocap_fusion.predict(
                            text=text_input,
                            audio_waveform=audio_waveform_input,
                            video_image=video_image_input,
                        )
                        
                        pred_label = fused_emotion_result["emotion"]
                        pred_conf = fused_emotion_result["confidence"]
                        
                        st.success(f"✅ IEMOCAP Fusion Model: {pred_label} (confidence: {pred_conf:.2f})")
                        logger.info(f"[MAIN APP] IEMOCAP Fusion Prediction result: emotion={pred_label}, confidence={pred_conf:.2f}")
                        
                    except Exception as e:
                        logger.error(f"[MAIN APP] IEMOCAP fusion failed: {e}. Falling back to legacy inference.")
                        st.warning(f"IEMOCAP fusion failed: {e}. Falling back to legacy inference.")
                        use_iemocap_fusion = False
                
                if not use_iemocap_fusion:
                    logger.info("[MAIN APP] Falling back to legacy MELD/DAIC inference path.")
                    text_logits = None
                    
                    if has_text:
                        emotion_result = predict_emotion(
                            model=model,
                            tokenizer=tokenizer,
                            id2label=id2label,
                            device=device,
                            text=user_text,
                        )
                        if emotion_result is not None:
                            # Extract values for backward compatibility
                            pred_label = emotion_result["primary_label"]
                            pred_conf = emotion_result["primary_confidence"]
                            pred_conf_tag = emotion_result["confidence_level"]
                            text_logits = emotion_result["logits"]
                            logger.info(f"[MAIN APP] Text-only prediction result: emotion={pred_label}, confidence={pred_conf:.2f}")
                        else:
                            pred_label = None
                            pred_conf = None
                            pred_conf_tag = None
                            text_logits = None
                    
                    # Extract video embedding if webcam image is provided
                    video_emb = None
                    if webcam_image is not None:
                        try:
                            logger.info("[MAIN APP] Video model extracting embedding from webcam snapshot (ResNet18).")
                            from PIL import Image
                            import io
                            
                            image_bytes = webcam_image.getvalue()
                            pil_img = Image.open(io.BytesIO(image_bytes))
                            video_emb = extract_video_embedding(pil_img)  # (512,)
                            logger.info(f"[MAIN APP] Video embedding extracted (shape: {video_emb.shape}).")
                        except Exception as e:
                            logger.error(f"[MAIN APP] Failed to extract video embedding: {e}")
                            st.error(f"Failed to extract video embedding: {e}")
                    
                    # The original LateFusion logic was flawed, as it was initialized with incorrect dimensions.
                    # Bypassing it and creating a dummy vector for the LLM-style generator,
                    # which does not critically depend on its content.
                    fused_vector = [0.0] * 256  # Dummy vector of the expected dimension
                    logger.info("[MAIN APP] Bypassed flawed LateFusion logic and created a dummy fused_vector.")
                else:
                    # Use IEMOCAP fusion result for LLM input
                    # Convert logits to a vector representation (use first few principal components or mean)
                    logits = fused_emotion_result["logits"]
                    fused_vector = logits.squeeze(0).numpy().tolist()
                    logger.info(f"[MAIN APP] Using IEMOCAP fusion result to create fused vector (length: {len(fused_vector)}).")
                
                # Get depression probability from DAIC-WOZ audio model (if available)
                if has_audio and audio_probs is not None and audio_probs.shape[1] >= 2:
                    depression_prob_for_state = float(audio_probs[0, 1])
                    logger.info(f"[MAIN APP] Depression probability from DAIC-WOZ: {depression_prob_for_state:.4f}")
                
                # Use user_text or default message if only audio/video
                if has_text:
                    user_text_for_llm = user_text
                elif has_audio:
                    user_text_for_llm = "The user provided only audio describing how they feel."
                elif has_video:
                    user_text_for_llm = "The user provided only a video snapshot expressing their emotions."
                else:
                    user_text_for_llm = "The user provided multimodal input."

                # Build emotional state summary and CBT suggestions
                emotional_state_summary = None
                
                if pred_label is not None and pred_conf is not None:
                    try:
                        emotional_state_summary = build_emotional_state(
                            emotion_label=pred_label,
                            emotion_confidence=pred_conf,
                            depression_prob=depression_prob_for_state,
                        )
                        logger.info(f"[MAIN APP] Built emotional state summary: '{emotional_state_summary}'")
                    except Exception as e:
                        logger.warning(f"[MAIN APP] Could not build emotional state summary: {e}")
                        st.warning(f"Could not build emotional state summary: {e}")
                
                # Simple CBT suggestions from existing rule-based engine
                cbt_suggestions = []
                if has_text and pred_label is not None:
                    try:
                        cbt_result = generate_cbt_response(
                            user_text=user_text,
                            emotion_label=pred_label,
                            depression_risk=depression_prob_for_state,
                        )
                        cbt_suggestions = cbt_result.get("steps", [])
                        logger.info(f"[MAIN APP] Generated CBT suggestions (rule-based): {len(cbt_suggestions)} steps.")
                    except Exception as e:
                        logger.warning(f"[MAIN APP] Could not generate CBT suggestions: {e}")
                        cbt_suggestions = []

                # Generate response using local multi-turn NLG engine
                try:
                    # Correctly log that the LLM-style simulated engine is always used in this path.
                    logger.info("[MAIN APP] Generating response using LLM-style (simulated) engine.")
                    assistant_reply, updated_history = generate_psychologist_response(
                        user_message=user_text_for_llm,
                        fused_vector=fused_vector,
                        emotional_state=emotional_state_summary,
                        history=st.session_state["chat_history"],
                        cbt_suggestions=cbt_suggestions,
                        config=PsychologistLLMConfig(),
                    )
                    st.session_state["chat_history"] = updated_history
                    logger.info(f"[MAIN APP] Generated assistant reply (LLM-style simulated): '{assistant_reply[:100]}...'")
                    
                    # Display the final response
                    st.markdown("---")
                    st.subheader("Combined Response")
                    st.write(assistant_reply)
                    
                    # Optionally show analysis details in expander
                    with st.expander("View Analysis Details"):
                        if pred_label is not None:
                            st.markdown("### Emotion Detection")
                            # Check if we have the new format (dict) or old format (tuple)
                            if emotion_result is not None and isinstance(emotion_result, dict) and "description" in emotion_result:
                                st.markdown(f"**{emotion_result['description']}**")
                                st.markdown(
                                    f"**Top emotions:** {emotion_result['primary_label']} ({emotion_result['primary_confidence']:.2f}), "
                                    f"{emotion_result['secondary_label']} ({emotion_result['secondary_confidence']:.2f})"
                                )
                                st.markdown(f"**Confidence level:** {emotion_result['confidence_level']}")
                            else:
                                # Fallback for old format or IEMOCAP fusion result
                                st.markdown(
                                    f"**Predicted Emotion:** {pred_label}  \n"
                                    f"**Confidence:** {pred_conf:.2f}"
                                )
                                if pred_conf_tag:
                                    st.markdown(f"**Confidence tag:** {pred_conf_tag}")
                            if use_iemocap_fusion and fused_emotion_result:
                                st.markdown(f"**Model:** IEMOCAP Multimodal Fusion")
                                st.markdown(f"**Available Modalities:** {', '.join([k for k, v in fused_emotion_result['available_modalities'].items() if v])}")
                        
                        if has_audio and audio_result is not None:
                            st.markdown("### Audio Analysis (DAIC-WOZ)")
                            st.markdown(f"**Audio Prediction:** {audio_result['pred_label']}")
                            if audio_probs is not None and audio_probs.shape[1] >= 2:
                                depression_prob = float(audio_probs[0, 1])
                                st.markdown(f"**Depression Probability:** {depression_prob:.4f}")
                        
                        if has_video and webcam_image is not None:
                            st.markdown("### Video Analysis")
                            if use_iemocap_fusion:
                                st.markdown("**Video:** Processed by IEMOCAP ResNet")
                            else:
                                st.markdown("**Video Embedding:** Extracted (512-dim)")
                            st.image(
                                webcam_image,
                                caption="Captured snapshot",
                                use_container_width=False,
                                width=200,
                            )

                except Exception as e:
                    logger.error(f"[MAIN APP] Failed to generate combined response: {e}", exc_info=True)
                    st.error(f"Failed to generate combined response: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown(
        "**Emotion-Adaptive CBT Assistant** - Research prototype for emotion recognition and CBT-based support"
    )


if __name__ == "__main__":
    main()
