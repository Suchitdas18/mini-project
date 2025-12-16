# Continual Learning System for Hate-Speech Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.30+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📋 Overview

A production-ready **continual learning system** for hate-speech detection that adapts to evolving linguistic patterns while preventing catastrophic forgetting. The system combines memory-based rehearsal, knowledge distillation, and Elastic Weight Consolidation (EWC) to enable incremental model updates.

### 🎯 Key Features

- **Adaptive Learning**: Automatically adapts to new hate-speech patterns and slang
- **Catastrophic Forgetting Prevention**: Maintains performance on historical patterns using EWC + rehearsal
- **Privacy-Preserving**: Supports embedding-only storage to protect sensitive content
- **Explainable Predictions**: Attention-based explanations for model decisions
- **Production-Ready API**: FastAPI endpoints for real-time detection
- **Comprehensive Metrics**: BWT, FWT, forgetting metrics, fairness evaluation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Hate-Speech Detector                   │
│                   (RoBERTa + Adapters)                   │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │   EWC    │    │Knowledge│    │ Rehearsal │
  │  Regular-│    │Distilla-│    │  Memory   │
  │  ization │    │  tion   │    │  Buffer   │
  └──────────┘    └──────────┘    └──────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Suchitdas18/mini-project.git
cd mini-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for data augmentation)
python -c "import nltk; nltk.download('wordnet')"
```

## 🚀 Quick Start

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates a sample dataset in `data/sample_data.csv` with 5000 labeled examples.

### 2. Train Initial Model

```bash
python train.py --data data/sample_data.csv
```

Training progress will be displayed with metrics including:
- Loss
- F1 Score
- Precision/Recall
- Per-class performance

### 3. Run Continual Learning Update

```python
from src.model import create_detector
from src.continual_learning import ContinualLearningTrainer, RehearsalBuffer
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize model
model = create_detector(config["model"])
model.load_model("models/best_model")

# Initialize rehearsal buffer
buffer = RehearsalBuffer(
    capacity=config["continual_learning"]["rehearsal_buffer_size"],
    privacy_mode=config["rehearsal"]["privacy_mode"],
)

# Initialize trainer
trainer = ContinualLearningTrainer(
    model=model,
    rehearsal_buffer=buffer,
    lambda_distill=config["continual_learning"]["lambda_distill"],
    lambda_ewc=config["continual_learning"]["lambda_ewc"],
)

# Perform continual learning update
new_data = {
    "texts": ["new example 1", "new example 2"],
    "labels": ["hate_speech", "neutral"],
}

metrics = trainer.train_step(new_data)
print(metrics)
```

## 📊 Project Structure

```
mini-project/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── detector.py          # Core hate-speech detection model
│   ├── continual_learning/
│   │   ├── __init__.py
│   │   ├── rehearsal_memory.py  # Rehearsal buffer & exemplar selection
│   │   └── trainer.py           # Continual learning trainer
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Data loading & augmentation
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py           # Evaluation metrics
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── train.py                     # Training script
├── generate_sample_data.py      # Sample data generator
├── problem_explanation.md       # Problem statement
└── technical_specification.md   # Technical documentation
```

## ⚙️ Configuration

Key configuration parameters in `config.yaml`:

```yaml
model:
  base_model: "roberta-base"
  num_labels: 3
  use_adapters: true

continual_learning:
  drift_threshold: 0.25
  lambda_distill: 0.5      # Knowledge distillation weight
  lambda_ewc: 0.3          # EWC regularization weight
  rehearsal_buffer_size: 10000

training:
  num_epochs: 3
  batch_size: 32
  learning_rate: 2.0e-5
```

## 📈 Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall classification accuracy
- **Macro F1**: F1 score averaged across classes
- **Per-Class Metrics**: Precision, recall, F1 for each class

### Continual Learning Metrics
- **Backward Transfer (BWT)**: Measures forgetting of previous tasks
- **Forward Transfer (FWT)**: Measures knowledge transfer to new tasks
- **Average Forgetting**: Average performance degradation on past tasks

### Fairness Metrics
- **False Positive Disparity**: FPR difference across demographic groups
- **Demographic Parity**: Equal positive rate across groups

## 🔬 Key Components

### 1. HateSpeechDetector

Transformer-based classifier with adapter layers for efficient continual learning.

```python
from src.model import HateSpeechDetector

model = HateSpeechDetector(
    model_name="roberta-base",
    num_labels=3,
    use_adapters=True,
)

# Make predictions
results = model.predict(
    texts=["example text"],
    return_probabilities=True,
    return_embeddings=True,
)
```

### 2. RehearsalBuffer

Privacy-preserving memory buffer for representative samples.

```python
from src.continual_learning import RehearsalBuffer

buffer = RehearsalBuffer(
    capacity=10000,
    privacy_mode="embedding_only",  # 'raw_text', 'embedding_only', 'synthetic'
)

# Add samples
buffer.add_batch(texts=texts, labels=labels, embeddings=embeddings)

# Sample for training
rehearsal_data = buffer.sample(size=1000, strategy="balanced")
```

### 3. ContinualLearningTrainer

Combines EWC, distillation, and rehearsal for continual learning.

```python
from src.continual_learning import ContinualLearningTrainer

trainer = ContinualLearningTrainer(
    model=model,
    rehearsal_buffer=buffer,
    lambda_distill=0.5,
    lambda_ewc=0.3,
)

metrics = trainer.train_step(new_data)
```

## 🌟 Advanced Features

### Drift Detection

Automatically detects distribution shifts to trigger model updates:

```python
from src.continual_learning import DriftDetector

detector = DriftDetector(
    baseline_model=baseline_model,
    drift_threshold=0.25,
)

drift_score = detector.compute_drift(current_model, recent_texts)
if detector.should_update(drift_score):
    # Trigger continual learning update
    trainer.train_step(new_data)
```

### Explainability

Generate attention-based explanations for predictions:

```python
attention_data = model.get_attention_weights("This is hate speech")
print(f"Tokens: {attention_data['tokens']}")
print(f"Weights: {attention_data['attention_weights']}")
```

### Data Augmentation

Built-in augmentation techniques:

```python
from src.data import DataAugmentation

augmented = DataAugmentation.synonym_replacement(texts, n=3)
augmented = DataAugmentation.random_deletion(texts, p=0.1)
```

## 📚 Documentation

- **Problem Explanation**: See [`problem_explanation.md`](problem_explanation.md) for detailed problem statement
- **Technical Specification**: See [`technical_specification.md`](technical_specification.md) for complete system architecture

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/
```

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@software{hate_speech_continual_learning,
  title={Continual Learning System for Hate-Speech Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/Suchitdas18/mini-project}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue or contact [dassuchit18@gmail.com](mailto:dassuchit18@gmail.com).

---

<p align="center">
  Made with ❤️ for adaptive hate-speech detection
</p>
