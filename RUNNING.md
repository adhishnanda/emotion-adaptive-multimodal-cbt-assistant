# How to Run the Emotion CBT Assistant

This guide provides step-by-step instructions for running the entire pipeline.

## Prerequisites

1. **Data Setup**: Ensure IEMOCAP dataset is extracted to `data/raw/iemocap/IEMOCAP_full_release/`
2. **Index File**: The multimodal index should be generated at `data/processed/iemocap_multimodal_index.csv`
3. **Dependencies**: Install all requirements (see main README)

## Quick Start: Smoke Test

Before running anything else, verify the pipeline works:

```bash
python -m src.tests.smoke_test_pipeline
```

This will:
- Load all models (text, audio, video, fusion)
- Test forward passes on small batches
- Verify shapes and device placement
- Report any errors

**If smoke test fails, fix those issues first before proceeding.**

## Training Scripts

### 1. Train Text Model (IEMOCAP)

```bash
python -m src.training.train_text_iemocap
```

**Output**: `models/text/distilbert_iemocap_best.pt`

**Debug Mode**: Set `debug_mode: true` or `fast_dev_run: true` in `config/config.yaml` under `project:` to run on a small subset with 1-2 epochs.

### 2. Train Audio Model (IEMOCAP)

```bash
python -m src.training.train_audio_iemocap
```

**Output**: `models/audio/audio_iemocap_best.pt`

**Note**: Requires IEMOCAP multimodal index with audio paths.

### 3. Train Video Model (IEMOCAP)

```bash
python -m src.training.train_video_iemocap
```

**Output**: `models/video/video_iemocap_resnet_best.pt`

**Note**: Requires IEMOCAP multimodal index with video paths.

### 4. Train Fusion Model

**Prerequisites**: All three unimodal models must be trained first.

```bash
python -m src.training.train_iemocap_multimodal_fusion
```

**Output**: `models/fusion/iemocap_multimodal_fusion_best.pt`

**Note**: This combines text, audio, and video encoders into a single fusion model.

## Evaluation

### Run Full Evaluation

```bash
python -m src.evaluation.eval_iemocap_multimodal
```

This evaluates all modality combinations:
- Text-only
- Audio-only
- Video-only
- Text+Audio
- Text+Video
- Audio+Video
- Text+Audio+Video (full fusion)

**Outputs**:
- `reports/metrics/metrics_YYYYMMDD_HHMMSS.json` - Complete metrics
- `reports/metrics/summary_YYYYMMDD_HHMMSS.csv` - Summary table
- `reports/metrics/per_class_metrics_YYYYMMDD_HHMMSS.csv` - Per-class details
- `reports/metrics/confusion_matrices/` - Confusion matrix arrays

### Generate Plots

```bash
python -m src.evaluation.plot_iemocap_results
```

**Outputs** (in `reports/figures/`):
- Confusion matrices for all configurations
- Bar charts comparing metrics
- Per-class F1 heatmaps

## Streamlit Application

### Run the App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser (typically `http://localhost:8501`).

### Testing Modalities

#### Text-Only Input
1. Type text in the text area
2. Click "Get CBT Response"
3. Verify emotion prediction and CBT response appear

#### Text + Audio
1. Type text
2. Upload a WAV file
3. Click "Get CBT Response"
4. Verify:
   - Audio prediction is shown
   - Fusion result uses both text and audio
   - CBT response is generated

#### Text + Video
1. Type text
2. Take a webcam snapshot
3. Click "Get CBT Response"
4. Verify:
   - Video is processed
   - Fusion result uses both text and video
   - CBT response is generated

#### Full Multimodal (Text + Audio + Video)
1. Type text
2. Upload audio file
3. Take webcam snapshot
4. Click "Get CBT Response"
5. Verify:
   - All three modalities are processed
   - Fusion model combines all three
   - CBT response is generated
   - No crashes or errors

#### Live Session Mode
1. Click "🎙️ Start Live Session" in sidebar
2. Provide text, audio chunks, and webcam frames
3. Verify real-time updates (every 1-2 seconds)
4. Click "⏹️ Stop Live Session" when done

## Debug Mode

For quick testing, set in `config/config.yaml`:

```yaml
project:
  debug_mode: true      # Use small dataset subset
  fast_dev_run: true    # Run only 1-2 epochs
```

This is useful for:
- Verifying scripts run without errors
- Testing on limited data
- Quick iteration during development

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure models are trained first
   - Check paths in `config/config.yaml`
   - Verify model files exist in `models/` directory

2. **Dataset Not Found**
   - Verify IEMOCAP data is extracted
   - Check `data/processed/iemocap_multimodal_index.csv` exists
   - Run index builder if needed: `python -m src.data.build_iemocap_multimodal_index`

3. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use CPU: set `device: "cpu"` in config
   - Enable debug mode to use smaller datasets

4. **Import Errors**
   - Ensure you're in the project root directory
   - Check `PYTHONPATH` includes project root
   - Verify all dependencies are installed

5. **Streamlit Errors**
   - Check model paths in config
   - Verify all models are trained
   - Check browser console for JavaScript errors
   - Ensure webcam/microphone permissions are granted

### Verification Checklist

Before deployment or thesis writing, verify:

- [ ] Smoke test passes: `python -m src.tests.smoke_test_pipeline`
- [ ] All training scripts run (at least in debug mode)
- [ ] Evaluation script produces metrics
- [ ] Streamlit app loads without errors
- [ ] Text-only input works
- [ ] Text + audio works
- [ ] Text + video works
- [ ] Full multimodal (text + audio + video) works
- [ ] Live session mode works (if implemented)
- [ ] No crashes or runtime errors
- [ ] All model paths are correct
- [ ] Device placement (CPU/GPU) is correct

## File Structure

```
emotion_cbt_assistant/
├── config/
│   └── config.yaml          # Main configuration
├── models/                  # Trained models
│   ├── text/
│   ├── audio/
│   ├── video/
│   └── fusion/
├── data/
│   ├── raw/                 # Raw datasets
│   └── processed/            # Processed indices
├── reports/
│   ├── metrics/             # Evaluation metrics
│   └── figures/              # Evaluation plots
├── src/
│   ├── training/             # Training scripts
│   ├── evaluation/          # Evaluation scripts
│   ├── tests/                # Smoke tests
│   └── ...
└── app/
    └── streamlit_app.py      # Streamlit application
```

## Next Steps

After verifying everything works:

1. **Full Training**: Disable debug mode and train on full datasets
2. **Evaluation**: Run full evaluation to get final metrics
3. **Documentation**: Update thesis with results
4. **Deployment**: Prepare for deployment (if needed)

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run smoke test to identify specific failures
3. Verify config paths and model files
4. Check that all dependencies are installed

