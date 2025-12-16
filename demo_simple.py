"""
Simplified demonstration without adapters to avoid dependency issues
"""

import torch
import yaml
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 80)
print("CONTINUAL LEARNING HATE-SPEECH DETECTION - SIMPLIFIED DEMO")
print("=" * 80)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Using device: {device}")

# Step 1: Initialize simplified components
print("\n" + "=" * 80)
print("STEP 1: Initialize Components")
print("=" * 80)

print("\n✅ Initializing tokenizer and model...")
print("   Note: Using simplified version without adapters for demo")
print("   This will download RoBERTa model (~500MB) on first run...")

try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # Create a simple 3-class classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=3,
        problem_type="single_label_classification"
    )
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model initialized: roberta-base")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 This is likely the first run and the model needs to be downloaded.")
    print("   The download is ~500MB and may take a few minutes.")
    exit(1)

# Step 2: Demonstrate predictions
print("\n" + "=" * 80)
print("STEP 2: Test Predictions (Random Initialization)")
print("=" * 80)

test_texts = [
    "You're an idiot",
    "Get lost loser",
    "Thanks for your help",
    "Have a great day",
]

print("\n🔍 Making predictions on sample texts:")
print("   (Note: Model is randomly initialized, so predictions are random)")

label_map = {0: "neutral", 1: "offensive", 2: "hate_speech"}

for text in test_texts:
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    print(f"\n   Text: '{text}'")
    print(f"   → Prediction: {label_map[pred_class]} (conf: {confidence:.3f})")
    print(f"   → Probabilities: neutral={probs[0,0]:.3f}, offensive={probs[0,1]:.3f}, hate={probs[0,2]:.3f}")

# Step 3: Demonstrate continual learning concepts
print("\n" + "=" * 80)
print("STEP 3: Continual Learning Concepts")
print("=" * 80)

print("\n📚 What this demo demonstrates:")
print("   ✓ Model can be initialized for 3-class hate-speech detection")
print("   ✓ Tokenization and inference pipeline works")
print("   ✓ Ready to integrate continual learning components:")
print("      • Rehearsal Memory Buffer")
print("      • EWC Regularization")
print("      • Knowledge Distillation")
print("      • Drift Detection")

# Step 4: Show architecture
print("\n" + "=" * 80)
print("STEP 4: System Architecture")
print("=" * 80)

print("\n🏗️  Full System Components:")
print("""
┌─────────────────────────────────────────────────────────┐
│              Hate-Speech Detector (RoBERTa)             │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │   EWC    │    │Knowledge│    │ Rehearsal │
  │  Regular-│    │Distilla-│    │  Memory   │
  │  ization │    │  tion   │    │  Buffer   │
  └──────────┘    └──────────┘    └──────────┘
""")

print("\n💡 How Continual Learning Works:")
print("   1. New data arrives → detect distribution drift")
print("   2. If drift detected → trigger continual learning update")
print("   3. Combine new data + rehearsal samples from buffer")
print("   4. Train with combined loss:")
print("      Loss = TaskLoss + λ₁·DistillationLoss + λ₂·EWC_Loss")
print("   5. Update rehearsal buffer with exemplars")
print("   6. Validate on historical benchmarks")
print("   7. Deploy if BWT > -0.05 (minimal forgetting)")

# Step 5: Configuration
print("\n" + "=" * 80)
print("STEP 5: Configuration")
print("=" * 80)

print("\n⚙️  Key Hyperparameters (from config.yaml):")
print(f"   • Drift Threshold: {config['continual_learning']['drift_threshold']}")
print(f"   • λ_distill: {config['continual_learning']['lambda_distill']} (prevents forgetting)")
print(f"   • λ_ewc: {config['continual_learning']['lambda_ewc']} (protects important params)")
print(f"   • Rehearsal Buffer: {config['continual_learning']['rehearsal_buffer_size']:,} samples")
print(f"   • Learning Rate: {config['training']['learning_rate']}")
print(f"   • Batch Size: {config['training']['batch_size']}")

print("\n" + "=" * 80)
print("✅ DEMONSTRATION COMPLETE!")
print("=" * 80)

print("\n📝 Summary:")
print("   ✓ RoBERTa model successfully initialized")
print("   ✓ Tokenization and inference pipeline working")
print("   ✓ Ready for continual learning training")
print(f"   ✓ Running on: {device.upper()}")

print("\n🚀 Next Steps:")
print("   1. Generate training data:")
print("      python generate_sample_data.py")
print()
print("   2. Train the model:")
print("      python train.py --data data/sample_data.csv")
print()
print("   3. This will train on ~5000 examples for 3 epochs")
print(f"      Estimated time on CPU: ~25-30 minutes")
print(f"      Estimated time on GPU: ~5-10 minutes")
print()
print("   4. The trained model will learn to:")
print("      • Detect hate-speech vs offensive vs neutral content")
print("      • Adapt to new patterns without forgetting old ones")
print("      • Provide explainable predictions")

print("\n💡 Full System Features:")
print("   • Continual Learning with EWC + Knowledge Distillation + Rehearsal")
print("   • Privacy-preserving rehearsal memory")
print("   • Drift detection for automated updates")
print("   • Comprehensive metrics (BWT, FWT, Forgetting)")
print("   • Attention-based explainability")
print(f"   • Fairness evaluation tools")

print("\n" + "=" * 80)
