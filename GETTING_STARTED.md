# Getting Started Guide

## 🎯 Quick Start (5 Minutes)

Follow these steps to get the continual learning hate-speech detection system up and running:

### Step 1: Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Note**: The first installation will download the RoBERTa model (~500MB), which may take a few minutes depending on your internet connection.

### Step 2: Verify Installation (30 seconds)

```bash
python test_setup.py
```

This will verify that:
- ✅ Model can be created
- ✅ Tokenization works
- ✅ Inference pipeline is functional
- ✅ Rehearsal buffer is operational

Expected output:
```
✅ ALL TESTS PASSED!
```

### Step 3: Generate Sample Data (10 seconds)

```bash
python generate_sample_data.py
```

This creates a sample dataset with 5,000 examples in `data/sample_data.csv`.

### Step 4: Run Quick Demo (1 minute)

```bash
python demo.py
```

This demonstrates:
- Model initialization
- Continual learning update
- Drift detection
- Explainability features
- Backward transfer evaluation

### Step 5 (Optional): Train on Real Data (10-30 minutes)

```bash
python train.py --data data/sample_data.csv
```

Training time depends on:
- GPU availability (10 min with GPU, 30+ min without)
- Dataset size
- Number of epochs

---

## 📚 Understanding the Components

### 1. Model Architecture

The system uses **RoBERTa** (Robustly Optimized BERT Pretraining Approach) with **adapter layers**:

```python
from src.model import create_detector

model = create_detector({
    "base_model": "roberta-base",
    "num_labels": 3,
    "use_adapters": True,
})
```

**Why adapters?**
- 🎯 Efficient: Only train ~1% of parameters
- 🧠 Prevents catastrophic forgetting
- ⚡ Faster updates for new patterns

### 2. Continual Learning Pipeline

The system uses **three complementary techniques**:

#### a) Rehearsal Memory
Stores representative examples to prevent forgetting:

```python
from src.continual_learning import RehearsalBuffer

buffer = RehearsalBuffer(
    capacity=10000,
    privacy_mode="embedding_only",  # Stores embeddings, not raw text
)
```

#### b) Knowledge Distillation
Prevents the model from deviating too much from previous knowledge:

```python
# Previous model's predictions are used as "soft targets"
distillation_loss = KL_divergence(
    new_model_output,
    previous_model_output
)
```

#### c) Elastic Weight Consolidation (EWC)
Protects important parameters from large changes:

```python
# Important parameters (high Fisher information) are regularized
ewc_loss = Σ (fisher_info * (new_param - old_param)²)
```

### 3. Privacy-Preserving Features

The system supports three privacy modes:

| Mode | Storage | Privacy | Use Case |
|------|---------|---------|----------|
| `raw_text` | Full text | ⚠️ Low | Internal systems |
| `embedding_only` | Vector embeddings | ✅ Medium | Most applications |
| `synthetic` | Generated prototypes | ✅✅ High | Sensitive domains |

---

## 🔧 Configuration Guide

Edit `config.yaml` to customize the system:

### Key Parameters

```yaml
continual_learning:
  drift_threshold: 0.25        # Lower = more frequent updates
  lambda_distill: 0.5          # Higher = more conservative (less forgetting)
  lambda_ewc: 0.3              # Higher = slower adaptation
  rehearsal_buffer_size: 10000 # More samples = better retention

training:
  num_epochs: 3                # More epochs = better accuracy, slower training
  batch_size: 32               # Higher = faster but needs more GPU memory
  learning_rate: 2.0e-5        # Lower = more stable, slower convergence
```

### Tuning Tips

**If the model forgets too much (low backward transfer):**
- ↑ Increase `lambda_distill` (0.5 → 0.7)
- ↑ Increase `lambda_ewc` (0.3 → 0.5)
- ↑ Increase `rehearsal_buffer_size`
- ↑ Increase rehearsal ratio in training

**If the model doesn't adapt well to new patterns:**
- ↓ Decrease `lambda_distill` (0.5 → 0.3)
- ↓ Decrease `lambda_ewc` (0.3 → 0.1)
- ↑ Increase `num_epochs`
- ↑ Increase `learning_rate` slightly

---

## 📊 Evaluation Metrics Explained

### Standard Metrics

- **Accuracy**: % of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Of predicted hate-speech, how many are actually hate-speech?
- **Recall**: Of actual hate-speech, how many did we detect?

### Continual Learning Metrics

#### Backward Transfer (BWT)
**Measures**: How much did we forget previous tasks?

```
BWT = Average(performance_after - performance_during)
```

- **BWT > 0**: Model improved on old tasks! (rare, positive transfer)
- **BWT ≈ 0**: No forgetting ✅
- **BWT < 0**: Some forgetting ⚠️
- **BWT < -0.1**: Significant forgetting ❌

**Target**: BWT > -0.05 (less than 5% degradation)

#### Forward Transfer (FWT)
**Measures**: Does old knowledge help learn new tasks faster?

```
FWT = Average(initial_performance_on_new_task - random_baseline)
```

- **FWT > 0**: Positive transfer (old knowledge helps) ✅
- **FWT = 0**: No transfer
- **FWT < 0**: Negative transfer (old knowledge hurts)

**Target**: FWT > 0.10 (10% boost from prior knowledge)

#### Average Forgetting
**Measures**: Average performance drop on past tasks

```
Forgetting = Average(max_performance - final_performance)
```

**Target**: < 0.03 (3% forgetting)

---

## 🎓 Common Use Cases

### Use Case 1: Weekly Model Updates

```python
# Automated weekly update workflow

# 1. Collect new labeled data from the past week
new_texts, new_labels = collect_weekly_data()

# 2. Check for drift
drift_score = drift_detector.compute_drift(model, new_texts)

# 3. Update if needed
if drift_score > 0.25:
    metrics = trainer.train_step({
        "texts": new_texts,
        "labels": new_labels,
    })
    
    # 4. Update rehearsal buffer
    trainer.update_rehearsal_buffer(new_texts, new_labels)
    
    # 5. Validate on historical benchmark
    validate_on_all_past_tasks()
    
    # 6. Deploy if performance is good
    if metrics["backward_transfer"] > -0.05:
        deploy_model(model)
```

### Use Case 2: Real-Time Detection API

```python
from fastapi import FastAPI
from src.model import HateSpeechDetector

app = FastAPI()
model = HateSpeechDetector()
model.load_model("models/best_model")

@app.post("/detect")
def detect_hate_speech(text: str):
    results = model.predict(
        [text],
        return_probabilities=True,
    )
    
    return {
        "label": results["labels"][0],
        "confidence": float(results["confidence"][0]),
        "probabilities": {
            "neutral": float(results["probabilities"][0][0]),
            "offensive": float(results["probabilities"][0][1]),
            "hate_speech": float(results["probabilities"][0][2]),
        }
    }
```

### Use Case 3: Explainable Moderation

```python
# Get prediction with explanation
text = "User comment here"

# 1. Get prediction
results = model.predict([text], return_probabilities=True)

# 2. Get attention weights for explanation
attention = model.get_attention_weights(text)

# 3. Build explanation
explanation = {
    "prediction": results["labels"][0],
    "confidence": results["confidence"][0],
    "key_phrases": [
        (token, weight) 
        for token, weight in zip(attention["tokens"], attention["attention_weights"])
        if weight > 0.1  # Significant attention
    ]
}

# 4. Present to moderator
print(f"Flagged as: {explanation['prediction']}")
print(f"Key indicators: {explanation['key_phrases']}")
```

---

## 🐛 Troubleshooting

### Issue: "Out of memory" error

**Solutions:**
1. Reduce `batch_size` in config (32 → 16 → 8)
2. Use gradient accumulation
3. Reduce `max_length` (512 → 256)
4. Use CPU instead of GPU (slower but works)

### Issue: Model not adapting to new patterns

**Solutions:**
1. Increase `num_epochs`
2. Decrease `lambda_ewc` and `lambda_distill`
3. Increase learning rate slightly
4. Ensure new data has sufficient examples per class

### Issue: Model forgetting old patterns

**Solutions:**
1. Increase `rehearsal_buffer_size`
2. Increase `lambda_ewc` and `lambda_distill`
3. Use higher rehearsal ratio (0.5 → 0.7)
4. Ensure rehearsal buffer has balanced class distribution

### Issue: Slow training

**Solutions:**
1. Use GPU (CUDA)
2. Increase `batch_size` if memory allows
3. Reduce `num_epochs`
4. Use adapters (already default)
5. Consider mixed precision training

---

## 📖 Next Steps

1. **Explore the code**: Check `src/` directory for implementation details
2. **Read the specs**: See `technical_specification.md` for complete system design
3. **Customize**: Modify `config.yaml` for your use case
4. **Deploy**: Set up FastAPI server for production use
5. **Monitor**: Use Weights & Biases for experiment tracking

## 🤝 Need Help?

- 📧 Email: dassuchit18@gmail.com
- 🐛 Issues: Open an issue on GitHub
- 📚 Docs: See `technical_specification.md` and `problem_explanation.md`

---

**Happy detecting! 🎯**
