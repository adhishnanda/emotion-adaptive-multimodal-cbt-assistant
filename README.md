# Emotion CBT Assistant

A multi-modal emotion recognition system with Cognitive Behavioral Therapy (CBT) rules for emotion validation and analysis.

## Features

- **Multi-modal Emotion Recognition**: Support for text, audio, and video modalities
- **CBT Rules Integration**: Cognitive Behavioral Therapy rules for emotion validation
- **Late Fusion**: Combine predictions from multiple modalities
- **Streamlit Interface**: Interactive web application for emotion analysis
- **Multiple Datasets**: Support for MELD, IEMOCAP, and DAIC-WOZ datasets

## Project Structure

```
emotion_cbt_assistant/
├── config/              # Configuration files
├── data/                # Data directories (raw and processed)
├── models/              # Saved model checkpoints
├── cache/               # Cache directory
├── logs/                # Log files
├── src/                 # Source code
│   ├── utils/           # Utility functions
│   ├── data/            # Data loading modules
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── fusion/          # Multi-modal fusion
│   └── cbt/             # CBT rules
├── app/                 # Streamlit application
├── scripts/             # Setup and training scripts
└── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_cbt_assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the environment:
```bash
python scripts/setup_env.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Training settings
- Fusion weights
- CBT rules configuration

## Usage

### Training

Train a text-based emotion recognition model:

```bash
python scripts/run_text_training.py
```

With custom arguments:

```bash
python scripts/run_text_training.py --epochs 20 --batch_size 64 --learning_rate 3e-5
```

### Running the Application

Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Datasets

The system supports the following datasets:

- **MELD**: Multimodal EmotionLines Dataset
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture Database
- **DAIC-WOZ**: Distress Analysis Interview Corpus - Wizard of Oz

Place dataset files in the respective directories under `data/raw/`.

## Models

### Text Model
- **Architecture**: DistilBERT-based classifier
- **Input**: Text utterances
- **Output**: Emotion probabilities (7 classes)

### Audio Model (Future)
- **Architecture**: Wav2Vec2-based classifier
- **Input**: Audio features
- **Output**: Emotion probabilities

### Video Model (Future)
- **Architecture**: ResNet50-based classifier
- **Input**: Video frames
- **Output**: Emotion probabilities

## CBT Rules

The system includes CBT rules for:
- Keyword consistency checking
- Contradiction detection
- Confidence adjustment based on text content

## Multi-modal Fusion

Late fusion strategy combines predictions from multiple modalities with configurable weights.

## License

[Add your license here]

## Citation

If you use this code, please cite:

```bibtex
[Add citation information]
```

## Contact

[Add contact information]


