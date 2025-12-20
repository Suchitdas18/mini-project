# 🤗 Deploy to Hugging Face Spaces - Complete Guide

Welcome! This guide will help you deploy your Hate-Speech Detection model to Hugging Face Spaces in **under 15 minutes**.

---

## 🎯 Why Hugging Face Spaces?

✅ **FREE Forever** - No credit card needed  
✅ **No Cold Starts** - Always online (unlike Render)  
✅ **ML Community** - Get discovered by thousands of ML practitioners  
✅ **Portfolio Ready** - Perfect for job applications & showcasing  
✅ **Easy Updates** - Auto-deploys from Git  
✅ **GPU Options** - Can upgrade to GPU if needed (paid)

---

## 📋 What You'll Deploy

✅ **Gradio Web Interface** (`app_gradio.py`)  
✅ **Trained Model** (from `models/best_model/`)  
✅ **Interactive UI** with real-time predictions  
✅ **Example texts** for quick testing

---

## 🚀 Quick Deployment (3 Methods)

### Method 1: Direct Upload (EASIEST - 10 minutes) ⭐

This is the fastest way to get started!

#### Step 1: Create Hugging Face Account

1. Go to **[huggingface.co](https://huggingface.co)**
2. Click **"Sign Up"**
3. Use GitHub login (recommended) or email
4. Verify your email

#### Step 2: Create a New Space

1. Click your profile icon → **"New Space"**
2. Fill in the details:
   - **Space Name**: `hate-speech-detector` (or your choice)
   - **License**: MIT
   - **SDK**: Select **Gradio**
   - **Gradio Version**: 4.11.0
   - **Hardware**: **CPU basic** (free)
   - **Visibility**: **Public** (recommended for portfolio)

3. Click **"Create Space"**

#### Step 3: Upload Your Files

You'll see a file upload interface. Upload these files:

**Required Files:**
1. **`app_gradio.py`** - Main application (rename to `app.py` during upload!)
2. **`config.yaml`** - Configuration
3. **`requirements_gradio.txt`** - Dependencies (rename to `requirements.txt` during upload!)
4. **`README_HF.md`** - Documentation (rename to `README.md` during upload!)
5. **`models/`** folder - Your trained model directory

**How to Upload:**

```
📁 Files to Upload:
├── app.py (renamed from app_gradio.py)
├── requirements.txt (renamed from requirements_gradio.txt)
├── README.md (renamed from README_HF.md)
├── config.yaml
└── models/
    └── best_model/
        ├── config.json
        ├── pytorch_model.bin
        └── ... (all model files)
```

**Upload Steps:**
1. Click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Drag and drop OR click to browse
4. **IMPORTANT**: Rename files as shown above during upload
5. Click **"Commit changes to main"**

#### Step 4: Wait for Build

1. Go to **"App"** tab
2. You'll see build logs in real-time
3. Wait 5-10 minutes for first build
4. When you see "Running on local URL", it's ready! 🎉

#### Step 5: Test Your App

1. The Gradio interface will appear
2. Try the example texts
3. Test with your own input
4. Share your Space URL!

Your Space URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
```

---

### Method 2: Git Push (For Developers) 🔧

If you prefer using Git:

#### Step 1: Create Space (same as Method 1, Steps 1-2)

#### Step 2: Clone Your Space

```bash
# Install Git LFS (for large files)
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
cd hate-speech-detector
```

#### Step 3: Copy Files

```bash
# Copy from your project
cp path/to/app_gradio.py app.py
cp path/to/requirements_gradio.txt requirements.txt
cp path/to/README_HF.md README.md
cp path/to/config.yaml .
cp -r path/to/models .
```

#### Step 4: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment"

# Push to Hugging Face
git push
```

#### Step 5: Monitor Build

Go to your Space URL and watch the **"App"** tab.

---

### Method 3: GitHub Sync (BEST for Updates) 🔄

Keep your Space auto-synced with GitHub!

#### Prerequisites:
- Your code already on GitHub: `https://github.com/Suchitdas18/mini-project`

#### Step 1: Prepare GitHub Repo

First, add HF-specific files to your repo:

```bash
# In your project directory
cd c:\project\miniproject(anisha)

# Create symbolic links or copies for HF
copy app_gradio.py app.py
copy requirements_gradio.txt requirements.txt  

# Commit to GitHub
git add app.py requirements.txt README_HF.md
git commit -m "Add Hugging Face Spaces support"
git push origin main
```

#### Step 2: Create Space with GitHub Sync

1. Create new Space (same as Method 1)
2. In creation screen, find **"Link to GitHub"**
3. Connect your GitHub account
4. Select: `Suchitdas18/mini-project`
5. Choose branch: `main`
6. Set path mapping (optional)
7. Click **"Create Space"**

#### Step 3: Auto-Deploy

Now, every time you push to GitHub, your Space auto-updates! 🎉

---

## 📝 File Checklist

Before deploying, make sure you have:

### ✅ Required Files

- [ ] `app_gradio.py` - Gradio interface  
- [ ] `requirements_gradio.txt` - Minimal dependencies  
- [ ] `README_HF.md` - Space documentation  
- [ ] `config.yaml` - Configuration  
- [ ] `models/best_model/` - Trained model files

### ✅ Files to Rename on Upload

| Your File | Upload As |
|-----------|-----------|
| `app_gradio.py` | `app.py` |
| `requirements_gradio.txt` | `requirements.txt` |
| `README_HF.md` | `README.md` |

---

## 🎨 Customize Your Space

### Update Space Metadata

Edit the README.md front matter:

```yaml
---
title: Your Custom Title
emoji: 🛡️  # Choose any emoji
colorFrom: red  # Start color for gradient
colorTo: orange  # End color for gradient
sdk: gradio
sdk_version: 4.11.0
app_file: app.py
pinned: false
---
```

### Add Custom CSS

In `app_gradio.py`, you can add custom CSS:

```python
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background: #f0f0f0}") as demo:
    # ... your code
```

### Change Examples

Edit the `examples` list in `app_gradio.py`:

```python
examples = [
    ["Your custom example 1"],
    ["Your custom example 2"],
    # Add more...
]
```

---

## 🔥 Test Locally First

Before deploying, test the Gradio app locally:

```bash
# Install Gradio dependencies
pip install -r requirements_gradio.txt

# Run the app
python app_gradio.py
```

Open in browser: http://localhost:7860

If it works locally, it will work on HF Spaces! ✅

---

## 🐛 Troubleshooting

### Issue: "Application startup failed"

**Cause**: Missing dependencies or incorrect file paths

**Fix**:
1. Check build logs in **"Logs"** tab
2. Verify all files uploaded correctly
3. Check `requirements.txt` has all needed packages
4. Ensure model path in `config.yaml` matches uploaded structure

### Issue: "Model not found"

**Cause**: Model files not uploaded or wrong path

**Fix**:
```bash
# Verify your model directory structure:
models/
└── best_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.json
    └── merges.txt
```

Make sure this EXACT structure is uploaded.

### Issue: "Out of memory"

**Cause**: Free CPU tier has limited RAM (16GB)

**Fix**:
1. Use floating point 16 (fp16) for model
2. Reduce batch size
3. Or upgrade to paid GPU tier ($0.60/hour, pause when not using)

### Issue: "Build takes forever"

**Cause**: Installing PyTorch from scratch

**Fix**: Be patient! First build takes 5-10 minutes. Subsequent builds are faster (cached).

### Issue: "App shows error on load"

**Cause**: Model loading failed

**Fix**: Check if trained model exists. If not, the app will show a warning but still work with base model.

---

## 🎯 Best Practices

### 1. Model Size Optimization

If your model is very large (>1GB):

**Option A: Use Git LFS**
```bash
# Track large files
git lfs track "*.bin"
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

**Option B: Host Model Separately**
Upload model to Hugging Face Model Hub, then load in app:
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "YOUR_USERNAME/hate-speech-model"
)
```

### 2. Add Usage Analytics

Track how many people use your Space:

```python
import gradio as gr

def predict_with_analytics(text):
    # Your prediction code
    result = predict(text)
    
    # Log usage (optional)
    print(f"Prediction made at {datetime.now()}")
    
    return result
```

### 3. Enable Sharing

Make your Space easy to share:
- Use clear, descriptive title
- Add good examples
- Write detailed README
- Add screenshots
- Include use cases

### 4. Community Features

Enable discussions and allow others to duplicate:
- ✅ Enable "Community" tab
- ✅ Allow "Duplicate this Space"
- ✅ Respond to issues/comments

---

## 📊 After Deployment

### Share Your Work!

Once deployed, share on:

1. **LinkedIn**:
   ```
   🎉 Just deployed my AI hate speech detection model!
   
   Try it here: https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
   
   Built with #MachineLearning #NLP #AI #Gradio #Transformers
   ```

2. **Twitter**:
   ```
   Check out my hate speech detector 🛡️
   
   Live demo: https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
   
   #MachineLearning #NLP
   ```

3. **GitHub README**:
   Add this badge to your project README:
   ```markdown
   [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector)
   ```

4. **Portfolio**:
   Add the live demo link to your portfolio website

### Monitor Usage

1. Go to your Space
2. Click **"Settings"** → **"Analytics"**
3. See views, runs, and user engagement

### Update Your Model

When you improve your model:

**Method 1 (Direct Upload):**
1. Go to **"Files"** tab
2. Upload new model files
3. Commit changes

**Method 2 (Git):**
```bash
# Update model locally
# Then push
git add models/
git commit -m "Update model with better accuracy"
git push
```

---

## 🚀 Advanced: Add GPU Support

If you need faster inference:

1. Go to **"Settings"** → **"Hardware"**
2. Select GPU tier:
   - **T4 Small**: $0.60/hour
   - **A10G Small**: $1.50/hour
3. Enable **"Pause after idle"** to save costs
4. Click **"Save"**

Update your code to use GPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

---

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Gradio Examples](https://github.com/gradio-app/gradio/tree/main/demo)
- [Your Project Docs](https://github.com/Suchitdas18/mini-project)

---

## ✅ Quick Start Checklist

### Pre-Deployment:
- [ ] Test `app_gradio.py` locally
- [ ] Verify model files exist in `models/best_model/`
- [ ] Check `config.yaml` paths are correct
- [ ] Review `requirements_gradio.txt`

### Deployment:
- [ ] Create Hugging Face account
- [ ] Create new Space
- [ ] Upload/push all files
- [ ] Rename files correctly (app.py, requirements.txt, README.md)
- [ ] Wait for build to complete

### Post-Deployment:
- [ ] Test the live app
- [ ] Try all examples
- [ ] Check prediction quality
- [ ] Share on social media
- [ ] Add to portfolio
- [ ] Monitor analytics

---

## 🎉 Ready to Deploy?

You have everything you need! Choose your method:

1. **Quick Start** → Method 1 (Direct Upload)
2. **Developer** → Method 2 (Git Push)
3. **Auto-Sync** → Method 3 (GitHub Integration)

Your Space will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
```

**Good luck! 🚀** If you run into issues, check the troubleshooting section or comment on your Space for community help.

---

## 📞 Need Help?

- **HF Community**: [Hugging Face Discord](https://discord.gg/hugging-face)
- **Gradio Help**: [Gradio Discord](https://discord.gg/feTf9x3ZSB)
- **GitHub Issues**: [Your Repo Issues](https://github.com/Suchitdas18/mini-project/issues)

---

Built with ❤️ for the ML community
