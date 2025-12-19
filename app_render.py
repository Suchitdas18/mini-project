"""
Optimized Flask API for Render deployment (memory-efficient)
Loads model from Hugging Face Hub instead of local files
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import yaml
from pathlib import Path
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__, 
            static_folder='web/static',
            template_folder='web/templates')
CORS(app)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model (lazy loading)
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Lazy load model only when needed"""
    global model, tokenizer
    
    if model is not None:
        return model, tokenizer
    
    print("Loading model (first time)...")
    
    # Use lightweight model for deployment
    # You can replace this with your uploaded model on Hugging Face
    model_name = "distilroberta-base"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try to load trained model if available locally
        model_path = Path(config["paths"]["model_store"]) / "best_model"
        
        if model_path.exists() and not os.environ.get('RENDER'):
            # Load local model (for local development)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            print(f"✅ Loaded local trained model from {model_path}")
            model_trained = True
        else:
            # Use base model for deployment (to save memory)
            print("⚠️ Using base model (untrained) for deployment")
            print("   For better predictions, upload your trained model to Hugging Face Hub")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                problem_type="single_label_classification"
            )
            model_trained = False
        
        model = model.to(device)
        model.eval()
        print(f"✅ Model loaded on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    return model, tokenizer

# Label mapping
label_map = {0: "neutral", 1: "offensive", 2: "hate_speech"}

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect hate-speech in text"""
    try:
        # Lazy load model
        model, tokenizer = load_model()
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
        
        if model is None or tokenizer is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'status': 'error'
            }), 500
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Reduced for memory efficiency
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        prediction = label_map[pred_class]
        probabilities = {
            'neutral': float(probs[0, 0]),
            'offensive': float(probs[0, 1]),
            'hate_speech': float(probs[0, 2]),
        }
        
        # Check if trained model
        model_path = Path(config["paths"]["model_store"]) / "best_model"
        model_status = "trained" if model_path.exists() and not os.environ.get('RENDER') else "base"
        
        return jsonify({
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_status': model_status,
            'status': 'success'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    model_path = Path(config["paths"]["model_store"]) / "best_model"
    
    return jsonify({
        'model_loaded': model is not None,
        'model_trained': model_path.exists() and not os.environ.get('RENDER'),
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'environment': 'render' if os.environ.get('RENDER') else 'local',
        'status': 'online'
    })

@app.route('/api/examples', methods=['GET'])
def examples():
    """Get example texts for testing"""
    return jsonify({
        'examples': [
            {
                'category': 'Hate Speech',
                'texts': [
                    "You're worthless trash",
                    "Get out of here, nobody wants you",
                    "You make me sick",
                ]
            },
            {
                'category': 'Offensive',
                'texts': [
                    "This is stupid",
                    "What a waste of time",
                    "That's dumb",
                ]
            },
            {
                'category': 'Neutral',
                'texts': [
                    "Great work on the project!",
                    "Thanks for your help",
                    "Have a good day",
                ]
            }
        ]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 HATE-SPEECH DETECTION WEB INTERFACE (RENDER OPTIMIZED)")
    print("="*60)
    print(f"\n🖥️  Device: {device}")
    print(f"🌐 Environment: {'Render' if os.environ.get('RENDER') else 'Local'}")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server will run on port: {port}\n")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)
