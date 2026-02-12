# Emotion-Adaptive Multimodal CBT Assistant

A research-oriented multimodal emotion recognition system that generates adaptive Cognitive Behavioral Therapy (CBT)-style responses based on emotional signals extracted from text, speech, and visual input.

This project was developed as part of an MSc-level applied AI research study focusing on multimodal affective computing, modular system design, and ethical AI for digital mental health support.

---

## Project Overview

Mental health support systems often rely on static questionnaires or text-only interaction. This project explores how multimodal emotion recognition (text, audio, visual) can be integrated into a modular and interpretable pipeline to generate emotion-aware CBT-style responses.

Instead of focusing only on classification accuracy, this system emphasizes:

- Interpretability
- Modular design
- Reproducibility
- Ethical caution
- Real-world deployment constraints

---

## Key Features

### Multimodal Emotion Recognition
- Text-based emotion classification using fine-tuned DistilBERT
- Audio-based emotion and depression risk analysis
- Visual feature extraction using pretrained ResNet
- Late fusion of embeddings across modalities
- Robust handling of missing modalities

### CBT-Oriented Response Generation
- Rule-based CBT response generation
- Simulated LLM-style structured responses
- Optional integration with external LLM APIs
- Emotion-conditioned response modulation
- Non-clinical, safety-framed outputs

### Modular Architecture
- Clear separation of:
  - Pretrained encoders
  - Task-trained fusion components
  - Response generation module
- Fusion head trained independently
- Frozen unimodal encoders for controlled experimentation

### Research-Focused Design
- Reproducible training pipelines
- Controlled dataset splits
- Clear distinction between pretrained and fine-tuned components
- Transparent methodological limitations

---

## System Architecture

The system follows a late-fusion multimodal pipeline:

1. Input processing (Text / Audio / Image)
2. Unimodal encoding
3. Embedding fusion via MLP
4. Emotion probability prediction (Top-2 output)
5. Optional depression risk estimation
6. CBT-style response generation

See `docs/architecture.svg` for full diagram.

---

## Repository Structure

```
emotion-adaptive-multimodal-cbt-assistant/
│
├── config/ # Configuration files
├── data/ # Dataset directories (not included)
├── models/ # Saved checkpoints
├── logs/ # Training logs
├── docs/ # Architecture diagrams and paper
├── src/
│ ├── data/ # Dataset loaders
│ ├── models/ # Model definitions
│ ├── fusion/ # Multimodal fusion logic
│ ├── cbt/ # CBT strategy modules
│ ├── training/ # Training utilities
│ └── utils/ # Helper functions
├── scripts/ # Training & evaluation scripts
├── requirements.txt
└── README.md
```

---

## Datasets Used

This project integrates three publicly available research datasets:

### MELD
- Conversational emotion dataset
- Used for text emotion classification

### IEMOCAP
- Multimodal dataset (text, audio, video)
- Used for training fusion head
- Standard benchmark in affective computing

### DAIC-WOZ
- Clinical interview dataset
- Used for speech-based depression risk estimation
- Output used as auxiliary non-diagnostic signal

Datasets must be downloaded separately and placed under `data/raw/`.

---

## Model Components

### Text Emotion Model
- Architecture: DistilBERT (fine-tuned)
- Output: Emotion probability distribution
- Loss: Cross-entropy
- Used in unimodal and multimodal modes

### Visual Encoder
- Architecture: Pretrained ResNet
- Frozen backbone
- Used as feature extractor
- No full video training to avoid overfitting

### Audio Branch
- Emotion encoder (optional)
- Separate depression-risk CNN-BiLSTM model trained on DAIC-WOZ
- Risk output modulates response tone (non-diagnostic)

### Fusion Model
- Late fusion via embedding concatenation
- Lightweight MLP
- Only fusion head trained
- Handles missing modalities gracefully

---

## Installation

Clone the repository:

```bash
git clone https://github.com/adhishnanda/emotion-adaptive-multimodal-cbt-assistant.git

cd emotion-adaptive-multimodal-cbt-assistant

```
## Training

### Train Text Model
```bash
python scripts/train_text_meld.py
```

### Train Fusion Head (IEMOCAP)
```bash
python scripts/train_fusion_iemocap.py --epochs 5 --batch-size 4
```

---

## Evaluation

### Evaluation includes:
- Accuracy
- Macro F1-score
- Confusion matrices
- Unimodal vs multimodal comparison
- Qualitative CBT response analysis

---

## Research Paper

The full research paper describing:
- Methodology
- Architectural rationale
- Dataset justification
- Experimental design
- Results and discussion
- Ethical considerations

is available in the ```docs/``` directory.

---

## Ethical Considerations

This system is:
- Not a medical diagnostic tool
- Not a replacement for professional therapy
- Designed for research and educational purposes
- Explicitly framed as supportive, non-clinical assistance
- Depression-risk estimation is used only as a contextual signal, not as a diagnosis.

---

## Technical Stack
- Python 3.x
- PyTorch
- HuggingFace Transformers
- OpenCV
- NumPy / SciPy
- Scikit-learn

---

## Skills Demonstrated
- Multimodal machine learning
- NLP with transformers
- Transfer learning
- Late fusion architectures
- Speech-based mental health modeling
- Applied deep learning research
- Reproducible ML system design
-  Ethical AI implementation

---

## Limitations
- Emotion datasets contain acted expressions
- Visual processing uses static frames
- Depression dataset limited in size
- Not clinically validated

---

## Future Work
- Temporal video modeling
- Attention-based fusion experiments
- Real-time deployment pipeline
- Expanded clinical evaluation
- More robust emotion calibration










