"""
🤗 Hugging Face Spaces - Gradio Interface
Hate-Speech Detection with Continual Learning
"""

import gradio as gr
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load the hate-speech detection model"""
    global model, tokenizer, model_loaded
    
    try:
        print("🔄 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Try to load trained model first
        model_path = Path(config["paths"]["model_store"]) / "best_model"
        if model_path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            print(f"✅ Loaded trained model from {model_path}")
            model_loaded = True
        else:
            # Use pre-trained base model
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=3,
                problem_type="single_label_classification"
            )
            print("⚠️ Using base model (not fine-tuned)")
            model_loaded = False
        
        model = model.to(device)
        model.eval()
        print(f"✅ Model ready on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
load_model()

# Label mapping
label_map = {0: "Neutral", 1: "Offensive", 2: "Hate Speech"}
label_colors = {
    "Neutral": "🟢",
    "Offensive": "🟡", 
    "Hate Speech": "🔴"
}

def predict_hate_speech(text):
    """
    Predict if text contains hate speech
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Prediction results with confidence scores
    """
    if not text or len(text.strip()) == 0:
        return {
            "Error": "Please enter some text to analyze"
        }, "⚠️ No input provided", ""
    
    if model is None or tokenizer is None:
        return {
            "Error": "Model not loaded. Please refresh the page."
        }, "❌ Model Error", ""
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["model"]["max_length"]
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        # Prepare results
        prediction = label_map[pred_class]
        emoji = label_colors[prediction]
        
        # Create probability dictionary for Gradio Label component
        probabilities = {
            f"{label_colors['Neutral']} Neutral": float(probs[0, 0]),
            f"{label_colors['Offensive']} Offensive": float(probs[0, 1]),
            f"{label_colors['Hate Speech']} Hate Speech": float(probs[0, 2]),
        }
        
        # Create result message
        result_msg = f"{emoji} **{prediction}** (Confidence: {confidence:.1%})"
        
        # Create detailed explanation
        if prediction == "Hate Speech":
            explanation = "⚠️ **High Risk**: This text contains language that may be hateful or discriminatory."
        elif prediction == "Offensive":
            explanation = "⚠️ **Medium Risk**: This text contains potentially offensive language."
        else:
            explanation = "✅ **Low Risk**: This text appears to be neutral and respectful."
        
        return probabilities, result_msg, explanation
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "Error": f"Prediction failed: {str(e)}"
        }, "❌ Error", ""

# Example texts for quick testing
examples = [
    ["You're worthless trash"],
    ["This is stupid"],
    ["Great work on the project!"],
    ["Get out of here, nobody wants you"],
    ["Thanks for your help"],
    ["What a waste of time"],
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Hate-Speech Detection") as demo:
    
    gr.Markdown("""
    # 🛡️ Hate-Speech Detection System
    
    AI-powered hate speech detection with continual learning capabilities.
    This system can identify **hate speech**, **offensive language**, and **neutral** content.
    
    ### How it works:
    1. Enter text in the box below
    2. Click "Analyze" to get predictions
    3. View confidence scores for each category
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type or paste text here...",
                lines=5,
                max_lines=10
            )
            
            with gr.Row():
                clear_btn = gr.ClearButton()
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            result_label = gr.Label(
                label="Prediction Probabilities",
                num_top_classes=3
            )
            
            result_text = gr.Markdown(
                label="Result",
                value="*Results will appear here*"
            )
            
            explanation = gr.Markdown(
                label="Explanation",
                value=""
            )
    
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Click to test"
    )
    
    gr.Markdown("""
    ---
    
    ### ℹ️ About this Model
    
    - **Base Model**: RoBERTa (Robustly Optimized BERT)
    - **Categories**: 3 classes (Neutral, Offensive, Hate Speech)
    - **Features**: Continual learning with rehearsal memory
    - **Use Cases**: Content moderation, social media monitoring, community safety
    
    ### ⚠️ Disclaimer
    
    This is an AI model and may not be 100% accurate. Always use human judgment for final moderation decisions.
    
    ### 🔗 Links
    
    - [GitHub Repository](https://github.com/Suchitdas18/mini-project)
    - [Technical Documentation](https://github.com/Suchitdas18/mini-project/blob/main/technical_specification.md)
    - [Project Summary](https://github.com/Suchitdas18/mini-project/blob/main/PROJECT_SUMMARY.md)
    
    ---
    
    Built with ❤️ using Gradio and Transformers
    """)
    
    # Connect button to prediction function
    analyze_btn.click(
        fn=predict_hate_speech,
        inputs=text_input,
        outputs=[result_label, result_text, explanation]
    )
    
    # Also allow Enter key to trigger analysis
    text_input.submit(
        fn=predict_hate_speech,
        inputs=text_input,
        outputs=[result_label, result_text, explanation]
    )
    
    # Clear button functionality
    clear_btn.add(text_input)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 HATE-SPEECH DETECTION - GRADIO INTERFACE")
    print("="*60)
    print(f"\n🖥️  Device: {device}")
    print(f"🤖 Model loaded: {model is not None}")
    print(f"✅ Model trained: {model_loaded}")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
