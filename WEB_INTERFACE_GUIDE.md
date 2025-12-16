# 🌐 Web Interface Guide

## ✨ Beautiful Web Interface Created!

You now have a **modern, premium web interface** to interact with your hate-speech detection model!

---

## 🚀 How to Access the Interface

### Step 1: Start the Server

```bash
python app.py
```

### Step 2: Open Your Browser

Navigate to: **http://localhost:5000**

That's it! 🎉

---

## 🎨 Interface Features

### Main Features:

✅ **Text Input Area** - Enter or paste text to analyze  
✅ **Real-Time Analysis** - Click "Analyze" to get instant results  
✅ **Visual Results** - Beautiful display with:
   - Prediction label with color coding
   - Confidence percentage
   - Probability bars for all classes
   
✅ **Quick Examples** - Click pre-loaded examples to test  
✅ **Status Indicator** - Shows if model is trained/untrained  
✅ **Responsive Design** - Works on desktop and mobile  

### Design Highlights:

- 🎨 **Dark theme** with glassmorphism effects
- ✨ **Smooth animations** and transitions
- 🌈 **Color-coded predictions**:
  - 😊 **Green** = Neutral
  - ⚠️ **Orange** = Offensive
  - 🚫 **Red** = Hate Speech
- 📊 **Interactive probability bars**
- 🎯 **Premium, modern aesthetic**

---

## 📱 Using the Interface

### 1. Enter Text

Type or paste text in the input area. Or click one of the example texts to try it quickly.

### 2. Analyze

Click the **"Analyze Text"** button. The system will:
- Send your text to the model
- Process it in real-time
- Display results with probabilities

### 3. View Results

You'll see:
- **Main prediction** (Neutral/Offensive/Hate Speech)
- **Confidence level** (0-100%)
- **Probability bars** showing likelihood for each class

### 4. Try More Examples

Click **"Clear"** to reset and try another text!

---

## ⚠️ Important Notes

### If Model is Untrained

If you see "Model Ready (Untrained)" in the status:
- Predictions will be **random**
- You need to train the model first

**To train the model:**

```bash
# 1. Generate training data
python generate_sample_data.py

# 2. Train the model (25-30 min)
python train.py --data data/sample_data.csv

# 3. Restart the web server
python app.py
```

### Server Running

When you run `python app.py`, you'll see:

```
🚀 HATE-SPEECH DETECTION WEB INTERFACE
==================================================
🖥️  Device: cpu
🤖 Model loaded: True
✅ Using trained model from: models/best_model
   (or)
⚠️  Using untrained model (random predictions)

🌐 Starting server at: http://localhost:5000
   Open this URL in your browser to use the interface
```

---

## 🎯 Interface Structure

```
┌─────────────────────────────────────────────┐
│  Header                                     │
│  - Title & Status Badge                     │
├─────────────────────────────────────────────┤
│  Input Section    │  Results Section        │
│  - Text area      │  - Prediction           │
│  - Analyze button │  - Confidence          │
│  - Examples       │  - Probability bars     │
├─────────────────────────────────────────────┤
│  Info Cards                                 │
│  - Features & Benefits                      │
├─────────────────────────────────────────────┤
│  Footer                                     │
└─────────────────────────────────────────────┘
```

---

## 📊 API Endpoints

The Flask server provides these endpoints:

### GET `/`
Main web interface (HTML page)

### POST `/api/detect`
Analyze text for hate-speech

**Request:**
```json
{
  "text": "example text to analyze"
}
```

**Response:**
```json
{
  "text": "example text",
  "prediction": "hate_speech",
  "confidence": 0.95,
  "probabilities": {
    "neutral": 0.02,
    "offensive": 0.03,
    "hate_speech": 0.95
  },
  "model_status": "trained",
  "status": "success"
}
```

### GET `/api/status`
Get system status

**Response:**
```json
{
  "model_loaded": true,
  "model_trained": true,
  "device": "cpu",
  "cuda_available": false,
  "status": "online"
}
```

### GET `/api/examples`
Get example texts for testing

---

## 🛠️ Customization

### Change Port

Edit `app.py`, line at the bottom:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to your port
```

### Modify Design

Edit files in `web/`:
- `templates/index.html` - HTML structure
- `static/style.css` - Styling and colors
- `static/script.js` - JavaScript functionality

### Add Features

The Flask app (`app.py`) is well-documented. Add new API endpoints or modify existing ones easily.

---

## 🐛 Troubleshooting

### Server Won't Start

**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install flask flask-cors
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Change the port in `app.py` or kill the process using port 5000

### Can't Connect in Browser

**Check**:
1. Server is running (`python app.py`)
2. Using correct URL: `http://localhost:5000`
3. No firewall blocking

### Model Not Loading

**Check**:
1. Dependencies installed: `pip install torch transformers`
2. Model directory exists (or train a model first)
3. Check console output when starting server

---

## 🌟 What's Next?

### After Training Your Model:

1. **Better Predictions** - Trained model gives accurate results
2. **Share It** - Send the URL to others on your network
3. **Deploy It** - Host on cloud (Heroku, AWS, Google Cloud)
4. **Integrate It** - Use the API in your applications

### Deployment Options:

- **Heroku**: Easy deployment with free tier
- **AWS EC2**: Full control, scalable
- **Google Cloud Run**: Serverless, auto-scaling
- **Docker**: Containerize for any platform

---

## 📸 Screenshot Description

The interface features:
- **Dark, premium theme** with purple/blue gradient accents
- **Large, clear text input area**
- **Color-coded results** with smooth animations
- **Probability visualization** with animated bars
- **Info cards** explaining system features
- **Professional footer** with attribution

---

## ✅ Summary

**You now have:**

✨ A beautiful, modern web interface  
✨ Real-time hate-speech detection  
✨ Visual probability displays  
✨ Complete REST API  
✨ Responsive design  
✨ Professional aesthetics  

**Access it at: http://localhost:5000** (after running `python app.py`)

---

**Built with ❤️ using Flask + Modern Web Design**

🎯 **Ready to detect hate-speech with style!**
