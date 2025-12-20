# 🚀 Quick Reference: Hugging Face Spaces Deployment

## 📦 Files You Need

```bash
# These 3 files are ready in your project:
✅ app_gradio.py       # Gradio interface
✅ requirements_gradio.txt  # Dependencies
✅ README_HF.md        # Space documentation

# You'll also need:
✅ config.yaml         # Already in your project
✅ models/best_model/  # Your trained model
```

---

## 🎯 Quick Deploy (Method 1: Direct Upload)

### 1️⃣ Create Account
→ [huggingface.co](https://huggingface.co) → Sign Up

### 2️⃣ Create Space
→ Profile → "New Space" → Name: `hate-speech-detector` → SDK: Gradio → Create

### 3️⃣ Upload Files
Upload these files with THESE EXACT NAMES:

| Take This File | Upload As |
|----------------|-----------|
| `app_gradio.py` | **`app.py`** ⚠️ |
| `requirements_gradio.txt` | **`requirements.txt`** ⚠️ |
| `README_HF.md` | **`README.md`** ⚠️ |
| `config.yaml` | `config.yaml` ✅ |
| `models/` folder | `models/` ✅ |

### 4️⃣ Wait & Test
→ "App" tab → Wait 5-10 min → Test! 🎉

---

## 💻 Quick Deploy (Method 2: Git)

```bash
# 1. Clone your space
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
cd hate-speech-detector

# 2. Copy files (from your project directory)
cp ../app_gradio.py app.py
cp ../requirements_gradio.txt requirements.txt
cp ../README_HF.md README.md
cp ../config.yaml .
cp -r ../models .

# 3. Push
git add .
git commit -m "Initial deployment"
git push
```

---

## 🔍 Test Locally First

```bash
# Install dependencies
pip install -r requirements_gradio.txt

# Run Gradio app
python app_gradio.py

# Open browser
http://localhost:7860
```

---

## 📋 File Mapping

```
Your Project                  Hugging Face Space
────────────────────         ─────────────────────
app_gradio.py         →      app.py
requirements_gradio.txt →    requirements.txt
README_HF.md           →      README.md
config.yaml            →      config.yaml
models/                →      models/
```

---

## ⚡ Your Space URL

After deployment, your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector
```

Replace `YOUR_USERNAME` with your Hugging Face username.

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Check `requirements.txt` uploaded correctly |
| "Model not found" | Verify `models/best_model/` uploaded |
| "Build failed" | Check logs in "Logs" tab |
| "Out of memory" | Free tier should work; check model size |

---

## 📊 Share Your Work

After deployment, share on:

- LinkedIn: Add to portfolio/projects
- Twitter: Tweet the demo link
- GitHub: Add badge to README:
  ```markdown
  [![HF Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector)
  ```

---

## 📚 Full Guide

For detailed instructions, see: `HUGGINGFACE_DEPLOYMENT.md`

---

## ✅ Ready?

1. Go to [huggingface.co](https://huggingface.co)
2. Create new Space
3. Upload files
4. Share! 🎉
