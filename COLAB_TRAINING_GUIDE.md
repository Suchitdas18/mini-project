# 🚀 Google Colab GPU Training Guide

## Why Use Colab?

**Current Situation:**
- ⏰ CPU Training: 70+ hours (37 hours done, 30-36 hours remaining)
- 🐌 Progress: Only 81% of Epoch 1 complete

**With Google Colab:**
- ⚡ GPU Training: 2-3 hours total
- 🚀 15-30x faster
- 🆓 Free GPU access

---

## Step-by-Step Instructions

### Step 1: Prepare Your Script

The training script is ready: `train_colab.py`

### Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File → New Notebook**

### Step 3: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** (or any available GPU)
3. Click **Save**

### Step 4: Upload Training Script

**Option A: Direct Upload**
```python
# In first cell of Colab notebook
from google.colab import files
uploaded = files.upload()
# Then click "Choose Files" and select train_colab.py
```

**Option B: From GitHub (Recommended)**
```python
# In first cell of Colab notebook
!git clone https://github.com/Suchitdas18/mini-project.git
%cd mini-project
```

### Step 5: Run Training

```python
# In a new cell
!python train_colab.py
```

### Step 6: Monitor Progress

You'll see:
- ✅ GPU detection
- 📥 Dataset loading
- 🏋️ Training progress with progress bars
- 📊 Metrics after each epoch

### Step 7: Download Trained Model

After training completes (2-3 hours):

```python
# Zip the model folder
!zip -r hate_speech_model.zip hate_speech_model/

# Download it
from google.colab import files
files.download('hate_speech_model.zip')
```

### Step 8: Use Model Locally

1. Extract `hate_speech_model.zip` in your project folder
2. Update your code to load from this directory:

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained('./hate_speech_model')
tokenizer = RobertaTokenizer.from_pretrained('./hate_speech_model')
```

---

## Complete Colab Notebook Example

Here's a complete notebook you can copy-paste:

```python
# Cell 1: Setup
print("🚀 Setting up GPU training for Hate-Speech Detection")

# Check GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 2: Clone repository
!git clone https://github.com/Suchitdas18/mini-project.git
%cd mini-project

# Cell 3: Run training
!python train_colab.py

# Cell 4: Download model (run after training completes)
!zip -r hate_speech_model.zip hate_speech_model/
from google.colab import files
files.download('hate_speech_model.zip')
```

---

## Alternative: Upload Dataset Directly

If you want to use your specific dataset:

```python
# Upload dataset
from google.colab import files
uploaded = files.upload()  # Upload your CSV/JSON file

# Modify train_colab.py to load from uploaded file
import pandas as pd
df = pd.read_csv('your_dataset.csv')
train_texts = df['text'].tolist()
train_labels = df['label'].tolist()
```

---

## Expected Timeline

| Stage | Time |
|-------|------|
| Setup & Dependencies | 2-3 min |
| Dataset Download | 1-2 min |
| Epoch 1 | 45-60 min |
| Epoch 2 | 45-60 min |
| Saving Model | 1-2 min |
| **Total** | **~2-3 hours** |

Compare to: **70+ hours on CPU** ❌

---

## Troubleshooting

### "No GPU available"
- Go to Runtime → Change runtime type
- Select T4 GPU or any GPU
- If no GPU option, try again later (free tier has limits)

### "Dataset not found"
- Upload dataset manually
- Or modify script to use different dataset source

### "Out of memory"
- Reduce batch_size from 16 to 8 in CONFIG
- Reduce max_length from 128 to 64

### Session Timeout
- Colab free tier disconnects after 12 hours
- Click in the notebook every few hours to prevent disconnect
- Or upgrade to Colab Pro ($10/month)

---

## What to Do With Current Training?

You have 3 options:

**Option 1: Stop Current Training** (Recommended)
```bash
# Press Ctrl+C in the training terminal
# Then use Colab for much faster training
```

**Option 2: Keep Both Running**
- Let CPU training continue as backup
- Start Colab training in parallel
- Use whichever completes first

**Option 3: Continue CPU Training**
- Wait 30-36 more hours
- No additional setup needed
- Will eventually complete

---

## Next Steps After Training

1. ✅ Download trained model from Colab
2. ✅ Extract to your project folder
3. ✅ Update `app.py` to use new model
4. ✅ Test with `python demo.py`
5. ✅ Deploy web interface with `python app.py`
6. ✅ Push to GitHub

---

## Questions?

- Model saved in: `hate_speech_model/`
- Metrics saved in: `hate_speech_model/metrics.json`
- Expected accuracy: **85-90%**
- Expected F1: **0.82-0.85**

**Ready to train 30x faster!** 🚀
