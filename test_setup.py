"""
Quick test script to verify the model implementation works
"""

import torch
from src.model import create_detector
from src.continual_learning import RehearsalBuffer
import yaml


def quick_test():
    """Run a quick test to verify everything works"""
    
    print("=" * 60)
    print("QUICK TEST - Model Implementation Verification")
    print("=" * 60)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    
    # Test 1: Model Creation
    print("\n[1/5] Testing model creation...")
    try:
        model = create_detector(config["model"])
        print("   ✓ Model created successfully")
        print(f"   ✓ Model type: {config['model']['base_model']}")
        print(f"   ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return
    
    # Test 2: Tokenization
    print("\n[2/5] Testing tokenization...")
    try:
        test_texts = [
            "This is a test",
            "Hello world",
        ]
        
        encoded = model.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        print(f"   ✓ Tokenized {len(test_texts)} texts")
        print(f"   ✓ Input shape: {encoded['input_ids'].shape}")
    except Exception as e:
        print(f"   ✗ Error in tokenization: {e}")
        return
    
    # Test 3: Model Inference
    print("\n[3/5] Testing model inference...")
    try:
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)
        
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {logits.shape}")
        print(f"   ✓ Predictions: {predictions}")
    except Exception as e:
        print(f"   ✗ Error in model inference: {e}")
        return
    
    # Test 4: Prediction API
    print("\n[4/5] Testing prediction API...")
    try:
        results = model.predict(
            texts=test_texts,
            return_probabilities=True,
            return_embeddings=True,
        )
        
        print(f"   ✓ Predictions: {results['labels']}")
        print(f"   ✓ Confidence: {results['confidence']}")
        print(f"   ✓ Embeddings shape: {results['embeddings'].shape}")
        
        # Show detailed results
        for i, text in enumerate(test_texts):
            print(f"\n   Text: '{text}'")
            print(f"   → Label: {results['labels'][i]}")
            print(f"   → Confidence: {results['confidence'][i]:.4f}")
            print(f"   → Probabilities: {results['probabilities'][i]}")
    except Exception as e:
        print(f"   ✗ Error in prediction API: {e}")
        return
    
    # Test 5: Rehearsal Buffer
    print("\n[5/5] Testing rehearsal buffer...")
    try:
        buffer = RehearsalBuffer(
            capacity=100,
            privacy_mode="embedding_only",
        )
        
        # Add samples
        buffer.add_batch(
            labels=["neutral", "hate_speech", "offensive"],
            embeddings=results["embeddings"][:3] if len(results["embeddings"]) >= 3 else results["embeddings"],
        )
        
        stats = buffer.get_statistics()
        print(f"   ✓ Buffer created successfully")
        print(f"   ✓ Current size: {stats['current_size']}")
        print(f"   ✓ Class distribution: {stats['class_distribution']}")
        
        # Sample from buffer
        sampled = buffer.sample(size=2, strategy="random")
        print(f"   ✓ Sampled {len(sampled['labels'])} examples")
    except Exception as e:
        print(f"   ✗ Error in rehearsal buffer: {e}")
        return
    
    # Success
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
    print("\n📝 Summary:")
    print("   ✓ Model creation and initialization")
    print("   ✓ Tokenization pipeline")
    print("   ✓ Forward pass and inference")
    print("   ✓ Prediction API with probabilities and embeddings")
    print("   ✓ Rehearsal buffer functionality")
    
    print("\n🚀 Ready to proceed with:")
    print("   1. Generate sample data: python generate_sample_data.py")
    print("   2. Train model: python train.py --data data/sample_data.csv")
    print("   3. Run demo: python demo.py")
    
    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        if not success:
            print("\n⚠️  Some tests failed. Please check the error messages above.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
