# Live Multimodal Streaming Implementation

## Overview

This implementation provides real-time multimodal emotion detection using audio and video streams integrated with the IEMOCAP fusion model.

## Architecture

### Components

1. **`live_audio_stream.py`**: Manages real-time audio streaming and MFCC feature extraction
2. **`live_video_stream.py`**: Manages real-time video frame processing and preprocessing
3. **`live_multimodal_session.py`**: Orchestrates the live session, fusion inference, and CBT response generation

### How It Works

1. **Audio Processing**:
   - Captures audio chunks via Streamlit's `st.audio_input`
   - Extracts MFCC features from each chunk
   - Maintains a rolling history of MFCC features (last 10 seconds)
   - Processes chunks at the same sample rate as training (16kHz)

2. **Video Processing**:
   - Captures frames via Streamlit's `st.camera_input`
   - Applies ImageNet normalization transforms
   - Maintains a rolling history of recent frames (last 5 frames)
   - Processes at low FPS (1-3 FPS) to keep computation light

3. **Live Fusion**:
   - Periodically (every 1-2 seconds) fuses:
     - Latest text input
     - Recent audio MFCC features
     - Recent video frames
   - Runs through IEMOCAP fusion model
   - Generates emotion predictions and CBT responses
   - Updates UI in near real-time

## Usage

1. **Start Live Session**: Click "🎙️ Start Live Session" in the sidebar
2. **Provide Inputs**:
   - Type text in the text area
   - Record audio chunks using the microphone input
   - Show your face to the webcam
3. **View Results**: Emotion predictions and CBT responses update automatically every 1-2 seconds
4. **Stop Session**: Click "⏹️ Stop Live Session" when done

## Limitations & Notes

### Streamlit Limitations

Streamlit's `st.audio_input` and `st.camera_input` are not true continuous streams:
- `st.audio_input` captures discrete audio recordings (user must click to record each chunk)
- `st.camera_input` captures discrete snapshots (user must click to capture each frame)

For true continuous streaming, consider:
- Using `streamlit-webrtc` for WebRTC-based streaming
- Using a custom WebSocket-based solution
- Using a separate backend service with WebSocket support

### Current Implementation

The current implementation works with Streamlit's built-in components:
- Audio: User records short chunks (2-5 seconds) which are processed immediately
- Video: User captures snapshots which are processed immediately
- The app auto-refreshes every 1-2 seconds to update results

This provides a "near real-time" experience suitable for demonstration purposes.

### Performance Considerations

- Models are kept loaded in memory (cached with `@st.cache_resource`)
- Processing happens on GPU if available
- Audio/video history is limited to recent frames/chunks to manage memory
- Inference runs every 1-2 seconds to balance responsiveness and computation

## Future Improvements

1. **True Continuous Streaming**: Integrate `streamlit-webrtc` for continuous audio/video streams
2. **Adaptive Frame Rate**: Adjust video processing rate based on available compute
3. **Batch Processing**: Process multiple frames/chunks in batches for efficiency
4. **Streaming Backend**: Separate backend service for heavy processing with WebSocket communication

## Dependencies

- `torch`: For model inference
- `torchaudio`: For audio processing and MFCC extraction
- `torchvision`: For video frame preprocessing
- `PIL`: For image handling
- `streamlit`: For UI and input capture

