# Source Code Documentation

This directory contains the core implementation of the continual learning hate-speech detection system.

## 📁 Structure

```
src/
├── __init__.py
├── model/                    # Core detection model
│   ├── __init__.py
│   └── detector.py          # HateSpeechDetector class
├── continual_learning/      # Continual learning components
│   ├── __init__.py
│   ├── rehearsal_memory.py  # RehearsalBuffer & ExemplarSelector
│   └── trainer.py           # ContinualLearningTrainer & DriftDetector
├── data/                    # Data pipeline
│   ├── __init__.py
│   └── dataset.py           # Dataset & augmentation utilities
└── evaluation/              # Metrics and evaluation
    ├── __init__.py
    └── metrics.py           # ContinualLearningMetrics & FairnessMetrics
```

## 🔧 Module Overview

### 1. `model/detector.py`

**Main Class**: `HateSpeechDetector`

Core transformer-based classifier with:
- RoBERTa base model with optional adapter layers
- Forward pass with attention extraction
- Prediction API with probabilities and embeddings
- Attention-based explainability
- Fisher Information computation for EWC

**Key Methods**:
```python
predict(texts, return_probabilities, return_embeddings)
get_attention_weights(text)
encode(texts)
get_parameter_importance(dataloader)
save_model(path) / load_model(path)
```

### 2. `continual_learning/rehearsal_memory.py`

**Classes**: `RehearsalBuffer`, `ExemplarSelector`

Implements privacy-preserving memory for continual learning:
- Reservoir sampling with class balancing
- Three privacy modes: raw_text, embedding_only, synthetic
- Diverse sampling strategies (random, balanced, recent)

**Key Methods**:
```python
# RehearsalBuffer
add(text, label, embedding, metadata, score)
add_batch(texts, labels, embeddings, metadata, scores)
sample(size, strategy, label)
get_statistics()
save(path) / load(path)

# ExemplarSelector
select_boundary_cases(predictions, labels, top_k)
select_rare_classes(labels, class_counts, top_k)
select_diverse_samples(embeddings, top_k, method)
select_combined(texts, labels, predictions, embeddings, class_counts, top_k)
```

### 3. `continual_learning/trainer.py`

**Classes**: `ContinualLearningTrainer`, `DriftDetector`

Orchestrates continual learning with EWC + distillation + rehearsal:
- Combined loss function with three components
- Fisher Information computation
- Automated rehearsal buffer updates

**Key Methods**:
```python
# ContinualLearningTrainer
train_step(new_data, rehearsal_ratio)
update_rehearsal_buffer(texts, labels, selection_strategy)

# DriftDetector
compute_drift(current_model, texts)
should_update(drift_score)
```

### 4. `data/dataset.py`

**Classes**: `HateSpeechDataset`, `DataAugmentation`

Data loading and preprocessing:
- PyTorch Dataset implementation
- Train/val/test splitting with stratification
- Data augmentation techniques

**Key Functions**:
```python
load_hate_speech_data(data_path, text_column, label_column)
create_dataloader(texts, labels, tokenizer, batch_size, max_length, shuffle)
split_data(texts, labels, train_ratio, val_ratio, test_ratio, random_seed)
```

### 5. `evaluation/metrics.py`

**Classes**: `ContinualLearningMetrics`, `FairnessMetrics`

Comprehensive evaluation framework:
- Standard classification metrics (accuracy, F1, precision, recall)
- Continual learning metrics (BWT, FWT, forgetting)
- Fairness metrics (FPR disparity, demographic parity)
- Visualization utilities

**Key Methods**:
```python
# ContinualLearningMetrics
compute_accuracy(predictions, labels)
compute_f1_scores(predictions, labels, average)
compute_per_class_metrics(predictions, labels, class_names)
compute_backward_transfer(task_performances)
compute_forward_transfer(task_performances, random_baseline)
compute_average_forgetting(task_performances)
plot_confusion_matrix(predictions, labels, class_names, save_path)
plot_task_performance_matrix(task_performances, task_names, save_path)

# FairnessMetrics
compute_false_positive_disparity(predictions, labels, groups)
compute_demographic_parity(predictions, groups)
```

## 🚀 Usage Examples

### Basic Model Usage

```python
from src.model import create_detector

# Create model
model = create_detector({
    "base_model": "roberta-base",
    "num_labels": 3,
    "use_adapters": True,
})

# Make predictions
results = model.predict(
    texts=["example text"],
    return_probabilities=True,
    return_embeddings=True,
)

print(results["labels"])      # Predicted labels
print(results["probabilities"])  # Class probabilities
print(results["embeddings"])     # Text embeddings
```

### Continual Learning Pipeline

```python
from src.model import create_detector
from src.continual_learning import ContinualLearningTrainer, RehearsalBuffer

# Initialize components
model = create_detector(config)
buffer = RehearsalBuffer(capacity=10000)
trainer = ContinualLearningTrainer(model, buffer)

# Perform update
new_data = {"texts": [...], "labels": [...]}
metrics = trainer.train_step(new_data)

# Update buffer
trainer.update_rehearsal_buffer(
    texts=new_data["texts"],
    labels=new_data["labels"],
)
```

### Evaluation

```python
from src.evaluation import ContinualLearningMetrics

# Compute metrics
f1_scores = ContinualLearningMetrics.compute_f1_scores(
    predictions=preds,
    labels=true_labels,
)

# Continual learning specific
bwt = ContinualLearningMetrics.compute_backward_transfer(
    task_performances=[[0.85], [0.90, 0.83], [0.91, 0.84, 0.88]]
)

# Visualization
ContinualLearningMetrics.plot_confusion_matrix(
    predictions=preds,
    labels=true_labels,
    save_path="results/confusion_matrix.png",
)
```

## 🔍 Design Principles

1. **Modularity**: Each component is independent and can be used separately
2. **Extensibility**: Easy to add new continual learning strategies or models
3. **Type Hints**: All functions have type annotations for clarity
4. **Documentation**: Comprehensive docstrings for all classes and methods
5. **Error Handling**: Robust error handling with informative messages

## 📚 Dependencies

Core dependencies (from requirements.txt):
- `torch >= 2.0.0` - Deep learning framework
- `transformers >= 4.30.0` - Hugging Face transformers
- `scikit-learn >= 1.3.0` - ML utilities and metrics
- `numpy >= 1.24.0` - Numerical computing

## 🛠️ Extending the Code

### Adding a New Continual Learning Strategy

```python
# In src/continual_learning/trainer.py

class NewCLStrategy:
    def __init__(self, model, ...):
        self.model = model
        # Initialize strategy-specific components
    
    def train_step(self, new_data):
        # Implement your strategy
        pass
```

### Adding a New Selection Strategy

```python
# In src/continual_learning/rehearsal_memory.py

class ExemplarSelector:
    @staticmethod
    def select_your_strategy(
        texts, labels, embeddings, top_k
    ):
        # Implement selection logic
        return selected_indices
```

### Adding a New Model

```python
# In src/model/detector.py

def create_custom_detector(config):
    # Use a different base model
    return HateSpeechDetector(
        model_name="bert-base-uncased",  # or any other model
        num_labels=config["num_labels"],
        use_adapters=config["use_adapters"],
    )
```

## 🧪 Testing

All modules should be tested with:
```bash
python test_setup.py
```

For individual module testing:
```python
# Test model
from src.model import create_detector
model = create_detector(config)
results = model.predict(["test"])
assert "labels" in results

# Test rehearsal
from src.continual_learning import RehearsalBuffer
buffer = RehearsalBuffer(capacity=100)
buffer.add("text", "label", None)
assert buffer.get_statistics()["current_size"] == 1
```

## 📖 Further Reading

- See `../technical_specification.md` for detailed algorithm descriptions
- See `../GETTING_STARTED.md` for usage tutorials
- See individual docstrings for method-level documentation
