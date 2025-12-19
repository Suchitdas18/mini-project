"""
Test the trained model with sample predictions
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def test_model():
    """Test the trained hate-speech detection model"""
    
    print("=" * 80)
    print("TESTING TRAINED MODEL")
    print("=" * 80)
    
    # Load model and tokenizer
    model_path = "models/best_model"
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("   Please train the model first using: python train_quick.py --data data/davidson_hate_speech.csv")
        return
    
    print(f"\n📥 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Label mapping
    label_map = {0: 'Neutral', 1: 'Offensive', 2: 'Hate Speech'}
    
    # Test samples
    test_texts = [
        "You're an idiot",
        "Get lost loser",
        "What a stupid moron",
        "Thanks for your help",
        "Have a great day",
        "Nice to meet you",
        "You're trash 🤡",
        "Absolute waste of oxygen",
        "KYS loser",
        "This project looks great",
        "Good point",
        "I appreciate your effort",
        "I hate you so much",
        "Go kill yourself",
        "You should die",
    ]
    
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    
    # Make predictions
    for text in test_texts:
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_label].item()
        
        label_name = label_map[pred_label]
        
        # Color code based on prediction
        if label_name == "Hate Speech":
            icon = "🔴"
        elif label_name == "Offensive":
            icon = "🟡"
        else:
            icon = "🟢"
        
        print(f"\n{icon} Text: '{text}'")
        print(f"   Prediction: {label_name} (confidence: {confidence:.2%})")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)
    print("\n📊 Legend:")
    print("   🟢 Neutral - Safe content")
    print("   🟡 Offensive - Potentially problematic")
    print("   🔴 Hate Speech - Harmful content")
    
    print("\n🚀 Next steps:")
    print("   1. Start web interface: python app.py")
    print("   2. Visit: http://localhost:5000")
    print("   3. Try your own text!")

if __name__ == "__main__":
    test_model()
