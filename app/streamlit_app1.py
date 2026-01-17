"""Streamlit application for Emotion-Adaptive CBT Assistant."""

from pathlib import Path
import sys
import av
import io
import time
from typing import List, Dict

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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

from config.config import load_config
from src.models.text_distilbert import build_text_model
from src.cbt.cbt_rules import generate_cbt_response
from src.nlg.llm_engine import generate_psychologist_response, PsychologistLLMConfig
from src.cbt.nlg_engine import build_emotional_state
from src.inference.multimodal_iemocap_inference import load_iemocap_multimodal_inference, extract_mfcc_from_waveform
from src.inference.audio_inference import predict_depression_from_wav
from src.inference.video_inference import extract_video_embedding
from src.fusion.late_fusion import LateFusion
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Emotion-Adaptive CBT Assistant",
    page_icon="😊",
    layout="wide",
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_text_model_and_tokenizer():
    """Load text model, tokenizer, and label mappings."""
    cfg = load_config()
    device = cfg.device.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.pretrained_name)

    try:
        train_csv = cfg.paths.raw_dir / "meld" / "train_sent_emo.csv"
        df_train = pd.read_csv(train_csv)
        le = LabelEncoder()
        le.fit(df_train["Emotion"].astype(str))
        num_labels = len(le.classes_)
        id2label = {i: label for i, label in enumerate(le.classes_)}
    except Exception as e:
        logger.error(f"[TEXT MODEL] Failed to load MELD training data for labels: {e}")
        num_labels = 7
        id2label = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}

    model = build_text_model(num_labels=num_labels, device=device)
    best_path = cfg.paths.models_dir / "text" / "distilbert_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info(f"[TEXT MODEL] Loaded text model checkpoint from {best_path} (Model: DistilBERT)")
    else:
        logger.warning(f"[TEXT MODEL] Checkpoint not found at {best_path}, using untrained model (Model: DistilBERT).")
    model.eval()
    logger.info(f"[TEXT MODEL] Final num_labels: {num_labels}, id2label mapping: {id2label}")
    return model, tokenizer, id2label, device

@st.cache_resource
def load_iemocap_model():
    """Load the IEMOCAP multimodal model."""
    try:
        cfg = load_config()
        inference_pipeline = load_iemocap_multimodal_inference(device=cfg.device.device)
        logger.info("[IEMOCAP FUSION MODEL] IEMOCAP multimodal fusion model loaded successfully.")
        return inference_pipeline
    except Exception as e:
        st.error(f"Failed to load IEMOCAP fusion model: {e}")
        logger.error(f"[IEMOCAP FUSION MODEL] Failed to load IEMOCAP fusion model: {e}")
        return None

def predict_text_emotion(model, tokenizer, id2label, device, text):
    """Predict emotion from a single text string."""
    if not text or not text.strip():
        logger.info("[TEXT INFERENCE] No text provided for prediction.")
        return None
    logger.info(f"[TEXT INFERENCE] Text model (DistilBERT) predicting emotion for text: '{text[:50]}...'")
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    primary_idx = np.argmax(probs)
    primary_label = id2label.get(primary_idx, "unknown")
    primary_conf = float(probs[primary_idx])
    
    logger.info("[TEXT INFERENCE] Prediction result: emotion=%s (confidence=%.2f)", primary_label, primary_conf)
    return {"emotion": primary_label, "confidence": primary_conf, "logits": logits}

@st.cache_resource
def load_iemocap_audio_emotion_model():
    """Load the IEMOCAP Audio CNN+LSTM model for audio-only emotion recognition."""
    cfg = load_config()
    device = cfg.device.device

    # Hardcoded IEMOCAP emotion labels (must match training)
    emotion_labels = [
        "anger", "happy", "sadness", "neutral", "excited",
        "frustration", "disgust", "fear", "surprise",
    ]
    id_to_emotion = {i: label for i, label in enumerate(emotion_labels)}
    num_classes = len(emotion_labels)

    from src.models.audio_iemocap_cnn_lstm import build_iemocap_audio_model
    model = build_iemocap_audio_model(num_classes=num_classes, device=device)
    
    best_path = cfg.paths.models_dir / "audio" / "audio_iemocap_best.pt"
    if best_path.exists():
        try:
            state_dict = torch.load(best_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"[AUDIO EMOTION MODEL] Loaded IEMOCAP audio model checkpoint from {best_path}")
        except Exception as e:
            logger.warning(f"[AUDIO EMOTION MODEL] Failed to load checkpoint from {best_path}: {e}")
    else:
        logger.warning(f"[AUDIO EMOTION MODEL] Checkpoint not found at {best_path}, using untrained model.")
    
    model.eval()
    return model, device, id_to_emotion

# --- Screen Rendering Functions ---

def render_multimodal_rule_based_screen(text_model, tokenizer, id2label, device, iemocap_model):
    st.title("Multimodal Analysis (Rule-based)")
    st.markdown("Combines text, audio, and video to detect emotion, then responds using the **simple rule-based engine**.")

    user_text = st.text_area("How are you feeling today?", height=150)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_audio = st.file_uploader("Upload audio (optional)", type=["wav"])
    with col2:
        webcam_image = st.camera_input("Take a snapshot (optional)")

    if st.button("Analyze", type="primary"):
        has_text = user_text and user_text.strip()
        has_audio = uploaded_audio is not None
        has_video = webcam_image is not None

        if not has_text and not has_audio and not has_video:
            st.warning("Please provide at least one input.")
            logger.warning("[MULTIMODAL RULE-BASED] No input provided for analysis.")
            return

        with st.spinner("Analyzing..."):
            emotion_result = predict_text_emotion(text_model, tokenizer, id2label, device, user_text) if has_text else None
            
            if has_audio or has_video:
                if iemocap_model:
                    logger.info("[MULTIMODAL RULE-BASED] Using IEMOCAP Multimodal Fusion model for prediction.")
                    # ... (rest of the logic)
                else:
                    st.error("IEMOCAP model not loaded.")
                    return
            
            # ... (rest of the logic)


def render_multimodal_llm_screen(text_model, tokenizer, id2label, device, iemocap_model):
    st.title("Multimodal Analysis (LLM-style)")
    st.markdown("Combines text, audio, and video to detect emotion, then responds using the **conversational LLM-style engine**.")
    # ... (implementation)


def render_rule_based_cbt_screen(model, tokenizer, id2label, device):
    st.title("Rule-based CBT Chat")
    # ... (implementation)

def render_llm_cbt_screen(model, tokenizer, id2label, device):
    st.title("LLM-style CBT Chat (Simulated)")
    # ... (implementation)

def render_audio_file_screen(iemocap_model_unused): # Renamed parameter as it's not used directly here
    st.title("Analyze Audio File")
    st.markdown("Upload a `.wav` audio file to analyze it.")

    uploaded_audio = st.file_uploader("Upload your audio file", type=["wav"])
    
    # Load IEMOCAP Audio-only emotion model
    iemocap_audio_model, iemocap_audio_device, iemocap_audio_id_to_emotion = load_iemocap_audio_emotion_model()

    if uploaded_audio:
        analysis_type = st.radio(
            "Choose analysis type:",
            ("Emotion Recognition (IEMOCAP Model)", "Depression Detection (DAIC-WOZ Model)")
        )

        if st.button("Analyze Audio"):
            with st.spinner("Analyzing..."):
                if analysis_type == "Emotion Recognition (IEMOCAP Model)":
                    if iemocap_audio_model: # Use the audio-only model
                        try:
                            import torchaudio
                            # Load and preprocess audio
                            waveform, sr = torchaudio.load(io.BytesIO(uploaded_audio.getvalue()))
                            if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
                            audio_input_waveform = waveform.squeeze(0) # For MFCC extraction

                            # Extract MFCC features
                            audio_mfcc = extract_mfcc_from_waveform(audio_input_waveform, sample_rate=16000, n_mfcc=40)
                            # Add batch dimension and move to device
                            audio_mfcc = audio_mfcc.unsqueeze(0).to(iemocap_audio_device)
                            
                            # Forward pass through the audio-only model
                            with torch.no_grad():
                                _, logits = iemocap_audio_model(audio_mfcc)
                                probs = torch.softmax(logits, dim=-1)
                                pred_id = logits.argmax(dim=-1).item()
                                pred_emotion = iemocap_audio_id_to_emotion[pred_id]
                                pred_confidence = probs[0, pred_id].item()

                            st.success("Emotion Analysis Complete!")
                            st.metric("Detected Emotion", pred_emotion.capitalize(), f"{pred_confidence:.2f}")
                            logger.info(f"[AUDIO FILE] IEMOCAP audio-only prediction: {pred_emotion} ({pred_confidence:.2f})")
                        except Exception as e:
                            st.error(f"Failed to analyze emotion: {e}")
                            logger.error(f"[AUDIO FILE] IEMOCAP audio-only analysis failed: {e}")
                    else:
                        st.error("IEMOCAP audio emotion model not loaded.")
                
                elif analysis_type == "Depression Detection (DAIC-WOZ Model)":
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / f"temp_audio_{int(time.time())}.wav"
                    try:
                        temp_path.write_bytes(uploaded_audio.getvalue())
                        result = predict_depression_from_wav(temp_path)
                        st.success("Depression Analysis Complete!")
                        st.metric("Prediction", result['pred_label'].capitalize())
                        if 'probs' in result and result['probs'].shape[1] >= 2:
                            st.progress(float(result['probs'][0, 1]))
                            st.write(f"Depression Probability: {float(result['probs'][0, 1]):.2%}")
                        logger.info(f"[AUDIO FILE] DAIC-WOZ prediction: {result['pred_label']}")
                    except Exception as e:
                        st.error(f"Failed to analyze for depression: {e}")
                        logger.error(f"[AUDIO FILE] DAIC-WOZ analysis failed: {e}")
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()


def render_image_file_screen(iemocap_model):
    st.title("Analyze Image File")
    st.markdown("Upload an image file (`.png`, `.jpg`) to detect emotion from a face.")

    uploaded_image = st.file_uploader("Upload your image file", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            if iemocap_model:
                with st.spinner("Analyzing..."):
                    try:
                        image = Image.open(io.BytesIO(uploaded_image.getvalue()))
                        result = iemocap_model.predict(video_image=image)
                        st.success("Image Analysis Complete!")
                        st.metric("Detected Emotion", result['emotion'].capitalize(), f"{result['confidence']:.2f}")
                        logger.info(f"[IMAGE FILE] IEMOCAP prediction: {result['emotion']} ({result['confidence']:.2f})")
                    except Exception as e:
                        st.error(f"Failed to analyze image: {e}")
                        logger.error(f"[IMAGE FILE] IEMOCAP analysis failed: {e}")
            else:
                st.error("IEMOCAP model not loaded.")

def render_live_audio_screen(iemocap_model):
    st.title("Real-time Audio Emotion Detection")
    # ... (implementation)

def render_live_webcam_screen(iemocap_model):
    st.title("Real-time Webcam Emotion Detection")
    # ... (implementation)


def main():
    """Main Streamlit application."""
    text_model, tokenizer, id2label, device = load_text_model_and_tokenizer()
    iemocap_model = load_iemocap_model()

    st.sidebar.title("Navigation")
    screen = st.sidebar.radio(
        "Choose a mode:",
        ("Multimodal (Rule-based)", "Multimodal (LLM-style)", "Rule-based CBT Chat", "LLM-style CBT Chat (Simulated)", "Analyze Audio File", "Analyze Image File", "Real-time Audio", "Real-time Webcam"),
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This is a research prototype. Not for clinical use.")
    logger.info(f"[MAIN APP] Selected mode: {screen}")

    if screen == "Multimodal (Rule-based)":
        render_multimodal_rule_based_screen(text_model, tokenizer, id2label, device, iemocap_model)
    elif screen == "Multimodal (LLM-style)":
        render_multimodal_llm_screen(text_model, tokenizer, id2label, device, iemocap_model)
    elif screen == "Rule-based CBT Chat":
        render_rule_based_cbt_screen(text_model, tokenizer, id2label, device)
    elif screen == "LLM-style CBT Chat (Simulated)":
        render_llm_cbt_screen(text_model, tokenizer, id2label, device)
    elif screen == "Analyze Audio File":
        render_audio_file_screen(iemocap_model)
    elif screen == "Analyze Image File":
        render_image_file_screen(iemocap_model)
    elif screen == "Real-time Audio":
        render_live_audio_screen(iemocap_model)
    elif screen == "Real-time Webcam":
        render_live_webcam_screen(iemocap_model)

if __name__ == "__main__":
    main()