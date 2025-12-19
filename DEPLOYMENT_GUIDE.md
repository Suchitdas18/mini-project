# 🚀 Deploy Your Hate-Speech Detection App Online

## Quick Deploy to Render (FREE) - Recommended ⭐

Render is the easiest way to host your Flask app for free with automatic deployments from GitHub.

---

## Prerequisites

✅ Your code is already on GitHub: `https://github.com/Suchitdas18/mini-project.git`
✅ You have a trained model in `models/best_model/`
✅ Requirements and deployment files are ready (Procfile, render.yaml)

---

## Step-by-Step Deployment

### Step 1: Push Latest Changes to GitHub

```bash
# Make sure all files are committed
git add .
git commit -m "Add deployment configuration for Render"
git push origin main
```

### Step 2: Sign Up for Render

1. Go to **[render.com](https://render.com)**
2. Click **"Get Started"**
3. Sign up with **GitHub** (easiest method)
4. Authorize Render to access your repositories

### Step 3: Create New Web Service

1. On Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your repository:
   - Search for: `mini-project`
   - Click **"Connect"**

### Step 4: Configure Your Service

Fill in these settings:

**Basic Settings:**
- **Name**: `hate-speech-detector` (or any name you prefer)
- **Region**: Choose closest to you (e.g., Oregon, Frankfurt, Singapore)
- **Branch**: `main`
- **Root Directory**: Leave blank
- **Runtime**: `Python 3`

**Build & Deploy Settings:**
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
  ```

**Instance Type:**
- Select **"Free"** (0.1 CPU, 512 MB RAM)

**Advanced Settings (Optional):**
- **Environment Variables**: None needed for now
- **Auto-Deploy**: ✅ Enabled (automatically deploys when you push to GitHub)

### Step 5: Deploy!

1. Click **"Create Web Service"**
2. Wait for deployment (5-10 minutes first time)
3. You'll see deployment logs in real-time

### Step 6: Get Your Live URL

Once deployed, you'll get a URL like:
```
https://hate-speech-detector.onrender.com
```

🎉 **Your app is now live!** Share this URL with anyone!

---

## What Gets Deployed

✅ Your Flask web application
✅ The trained model (models/best_model/)
✅ All web interface files (HTML, CSS, JS)
✅ API endpoints (/api/detect, /api/status)

---

## Important Notes

### ⚠️ Free Tier Limitations

- **Spin down after inactivity**: App sleeps after 15 minutes of no requests
- **First load slow**: Takes 30-60 seconds to wake up
- **RAM limit**: 512 MB (sufficient for your model)
- **No persistent storage**: Model loads from GitHub each time

### 💡 Tips for Better Performance

1. **Keep app awake**: Use a service like [UptimeRobot](https://uptimerobot.com/) to ping your app every 5 minutes

2. **Reduce dependencies**: Already done! requirements.txt only includes essentials

3. **Optimize model loading**: The app caches the model once loaded

---

## Alternative: Quick Public Access with ngrok (Instant)

If you want **immediate** public access without deployment:

### Install ngrok:
```bash
# Download from https://ngrok.com/download
# Or use chocolatey on Windows:
choco install ngrok
```

### Run ngrok:
```bash
# Your Flask app is running on port 5000
ngrok http 5000
```

### Get Public URL:
You'll see output like:
```
Forwarding: https://abc123.ngrok.io -> http://localhost:5000
```

✅ Share the ngrok URL: `https://abc123.ngrok.io`
⚠️ **Warning**: URL changes every time you restart ngrok (free tier)

---

## Alternative: Railway (Another Free Option)

### Deploy to Railway:

1. Go to **[railway.app](https://railway.app)**
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose `Suchitdas18/mini-project`
5. Railway auto-detects Python and deploys!

**Advantages:**
- Even easier than Render
- Generous free tier
- Faster deploys

**Disadvantages:**
- Free tier has monthly hour limits

---

## Monitoring Your Deployment

### Check if it's working:

1. **Visit your URL**: `https://your-app.onrender.com`
2. **Check status**: `https://your-app.onrender.com/api/status`
3. **View logs**: On Render dashboard → Logs tab

### Test the API:

```bash
# Test detection endpoint
curl -X POST https://your-app.onrender.com/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "This is stupid"}'
```

---

## Troubleshooting

### Issue: "Application failed to respond"
**Solution**: Check Render logs for errors. Usually means:
- Model files missing (ensure they're pushed to GitHub)
- Dependencies failed to install
- Out of memory (free tier = 512MB)

### Issue: "Build failed"
**Solution**: 
- Check requirements.txt is valid
- Ensure Python version is 3.9-3.11
- Remove heavy dependencies not needed for production

### Issue: Model not loading
**Solution**:
- Verify `models/best_model/` exists in GitHub repo
- Check file size (free tier has limits)
- May need to use model from Hugging Face Hub instead

---

## Next Steps After Deployment

### 1. Update GitHub README with your live URL

```markdown
## 🌐 Live Demo
Try it here: [https://your-app.onrender.com](https://your-app.onrender.com)
```

### 2. Share your link!

- Portfolio
- LinkedIn
- Twitter
- Course submission
- Job applications

### 3. Monitor usage

- Check Render dashboard for metrics
- See how many people use it
- Track errors and performance

---

## Cost Optimization

### Free Forever (Recommended)

**Current Setup:**
- Render Free tier: ✅ No credit card needed
- GitHub: ✅ Free public repo
- **Total cost: $0/month**

**Limitations:**
- Sleeps after 15 min inactivity
- 512 MB RAM
- 0.1 CPU

**Good for:**
- Portfolio projects
- Demos
- Course submissions
- Learning

### Upgrade Options (If Needed)

**Render Starter ($7/month):**
- No sleep
- 512 MB RAM
- 0.5 CPU
- Custom domain

**Render Standard ($25/month):**
- 2 GB RAM
- 1 CPU
- Better for production

---

## Files Created for Deployment

✅ **Procfile** - Tells Render how to run your app
✅ **render.yaml** - Render configuration
✅ **requirements.txt** - Updated with Flask and gunicorn

---

## Quick Commands Reference

```bash
# Push to GitHub
git add .
git commit -m "Update deployment files"
git push origin main

# Run locally (for testing before deploy)
python app.py

# Test with gunicorn (same as production)
gunicorn app:app --bind 0.0.0.0:5000 --timeout 120

# Check model exists
ls models/best_model/
```

---

## Success Checklist

Before deploying, verify:

- ✅ Code pushed to GitHub
- ✅ `models/best_model/` exists with trained model
- ✅ `Procfile` exists
- ✅ `render.yaml` exists
- ✅ `requirements.txt` has Flask, gunicorn
- ✅ `app.py` runs locally without errors
- ✅ Web interface loads at http://localhost:5000

---

## Ready to Deploy?

Choose your method:

1. **Render** (Recommended) - Permanent hosting, free forever
2. **ngrok** - Instant public URL, temporary
3. **Railway** - Alternative to Render, also free

Let's deploy! 🚀
