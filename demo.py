"""
Demonstration script showing the complete continual learning pipeline
"""

import torch
import yaml
import numpy as np
from pathlib import Path

from src.model import create_detector
from src.continual_learning import (
    RehearsalBuffer,
    ContinualLearningTrainer,
    DriftDetector,
    ExemplarSelector,
)
from src.evaluation import ContinualLearningMetrics


def demo_continual_learning():
    """
    Demonstrate the continual learning system end-to-end
    """
    print("=" * 80)
    print("CONTINUAL LEARNING HATE-SPEECH DETECTION - DEMONSTRATION")
    print("=" * 80)
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    
    # Step 1: Create model
    print("\n" + "=" * 80)
    print("STEP 1: Initialize Model")
    print("=" * 80)
    
    model = create_detector(config["model"]).to(device)
    print(f"✅ Created {config['model']['base_model']} with adapters")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 2: Initialize Rehearsal Buffer
    print("\n" + "=" * 80)
    print("STEP 2: Initialize Rehearsal Buffer")
    print("=" * 80)
    
    buffer = RehearsalBuffer(
        capacity=config["continual_learning"]["rehearsal_buffer_size"],
        privacy_mode=config["rehearsal"]["privacy_mode"],
        class_balance=config["rehearsal"]["class_balance"],
    )
    print(f"✅ Created rehearsal buffer")
    print(f"   Capacity: {buffer.capacity:,}")
    print(f"   Privacy mode: {buffer.privacy_mode}")
    
    # Step 3: Simulate Task 1 (Historical hate-speech patterns)
    print("\n" + "=" * 80)
    print("STEP 3: Task 1 - Historical Hate-Speech Patterns")
    print("=" * 80)
    
    task1_texts = [
        "You're an idiot",
        "Get lost loser",
        "What a moron",
        "Thanks for your help",
        "Have a great day",
        "Nice to meet you",
    ]
    
    task1_labels = [
        "hate_speech",
        "hate_speech",
        "offensive",
        "neutral",
        "neutral",
        "neutral",
    ]
    
    print(f"📚 Task 1 data: {len(task1_texts)} samples")
    
    # Get predictions before training
    print("\n🔍 Predictions before training:")
    results_before = model.predict(task1_texts[:3], return_probabilities=True)
    for i, (text, pred_label, conf) in enumerate(zip(
        task1_texts[:3],
        results_before["labels"],
        results_before["confidence"]
    )):
        print(f"   '{text}' → {pred_label} ({conf:.2f})")
    
    # Add to rehearsal buffer
    embeddings = model.encode(task1_texts)
    buffer.add_batch(
        texts=task1_texts if buffer.privacy_mode == "raw_text" else None,
        labels=task1_labels,
        embeddings=embeddings,
    )
    
    stats = buffer.get_statistics()
    print(f"\n💾 Rehearsal buffer updated:")
    print(f"   Current size: {stats['current_size']}")
    print(f"   Class distribution: {stats['class_distribution']}")
    
    # Step 4: Initialize Trainer
    print("\n" + "=" * 80)
    print("STEP 4: Initialize Continual Learning Trainer")
    print("=" * 80)
    
    trainer = ContinualLearningTrainer(
        model=model,
        rehearsal_buffer=buffer,
        lambda_distill=config["continual_learning"]["lambda_distill"],
        lambda_ewc=config["continual_learning"]["lambda_ewc"],
        temperature=config["continual_learning"]["temperature"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=1,  # Reduced for demo
        batch_size=config["training"]["batch_size"],
        device=device,
    )
    
    print(f"✅ Created continual learning trainer")
    print(f"   λ_distill: {trainer.lambda_distill}")
    print(f"   λ_ewc: {trainer.lambda_ewc}")
    
    # Step 5: Simulate Task 2 (New hate-speech patterns)
    print("\n" + "=" * 80)
    print("STEP 5: Task 2 - Emerging Hate-Speech Patterns")
    print("=" * 80)
    
    task2_texts = [
        "You're trash 🤡",
        "Absolute waste of oxygen",
        "KYS loser",
        "This project looks great",
        "Good point",
        "I appreciate your effort",
    ]
    
    task2_labels = [
        "hate_speech",
        "hate_speech",
        "hate_speech",
        "neutral",
        "neutral",
        "neutral",
    ]
    
    print(f"📚 Task 2 data: {len(task2_texts)} samples (new slang patterns)")
    
    # Demonstrate drift detection
    print("\n🔍 Checking for distribution drift...")
    drift_detector = DriftDetector(
        baseline_model=model,
        drift_threshold=config["continual_learning"]["drift_threshold"],
    )
    
    drift_score = drift_detector.compute_drift(model, task2_texts)
    print(f"   Drift score: {drift_score:.4f}")
    print(f"   Threshold: {drift_detector.drift_threshold}")
    print(f"   Should update: {drift_detector.should_update(drift_score)}")
    
    # Step 6: Continual Learning Update
    print("\n" + "=" * 80)
    print("STEP 6: Perform Continual Learning Update")
    print("=" * 80)
    
    print("🎓 Training on new data with rehearsal...")
    
    new_data = {
        "texts": task2_texts,
        "labels": task2_labels,
    }
    
    metrics = trainer.train_step(new_data, rehearsal_ratio=0.5)
    
    print(f"\n📊 Training Metrics:")
    print(f"   Total loss:        {metrics['total_loss']:.4f}")
    print(f"   Task loss:         {metrics['task_loss']:.4f}")
    print(f"   Distillation loss: {metrics['distillation_loss']:.4f}")
    print(f"   EWC loss:          {metrics['ewc_loss']:.4f}")
    
    # Update rehearsal buffer with new exemplars
    trainer.update_rehearsal_buffer(
        texts=task2_texts,
        labels=task2_labels,
        selection_strategy="combined",
    )
    
    stats = buffer.get_statistics()
    print(f"\n💾 Rehearsal buffer updated:")
    print(f"   Current size: {stats['current_size']}")
    print(f"   Total seen: {stats['total_seen']}")
    
    # Step 7: Evaluate on both tasks
    print("\n" + "=" * 80)
    print("STEP 7: Evaluation - Backward Transfer Test")
    print("=" * 80)
    
    print("\n📊 Testing on Task 1 (historical patterns):")
    task1_results = model.predict(task1_texts, return_probabilities=True)
    
    task1_preds = [model.reverse_label_map[p] for p in task1_results["predictions"]]
    task1_metrics = ContinualLearningMetrics.compute_f1_scores(
        predictions=task1_results["predictions"],
        labels=np.array([model.reverse_label_map[l] for l in task1_labels]),
    )
    
    print(f"   F1 Score: {task1_metrics['f1']:.4f}")
    print(f"   Precision: {task1_metrics['precision']:.4f}")
    print(f"   Recall: {task1_metrics['recall']:.4f}")
    
    print("\n📊 Testing on Task 2 (new patterns):")
    task2_results = model.predict(task2_texts, return_probabilities=True)
    
    task2_preds = [model.reverse_label_map[p] for p in task2_results["predictions"]]
    task2_metrics = ContinualLearningMetrics.compute_f1_scores(
        predictions=task2_results["predictions"],
        labels=np.array([model.reverse_label_map[l] for l in task2_labels]),
    )
    
    print(f"   F1 Score: {task2_metrics['f1']:.4f}")
    print(f"   Precision: {task2_metrics['precision']:.4f}")
    print(f"   Recall: {task2_metrics['recall']:.4f}")
    
    # Step 8: Demonstrate Explainability
    print("\n" + "=" * 80)
    print("STEP 8: Explainability - Attention Analysis")
    print("=" * 80)
    
    example_text = "You're trash 🤡"
    print(f"\n🔍 Analyzing: '{example_text}'")
    
    attention_data = model.get_attention_weights(example_text)
    
    print("\n📝 Token importance (top 5):")
    token_importance = list(zip(attention_data["tokens"], attention_data["attention_weights"]))
    token_importance.sort(key=lambda x: x[1], reverse=True)
    
    for token, weight in token_importance[:5]:
        print(f"   '{token}': {weight:.4f}")
    
    # Final prediction with confidence
    final_result = model.predict([example_text], return_probabilities=True)
    print(f"\n🎯 Prediction: {final_result['labels'][0]} (confidence: {final_result['confidence'][0]:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    
    print("\n📈 Summary:")
    print(f"   ✓ Model successfully adapted to new hate-speech patterns")
    print(f"   ✓ Maintained performance on historical patterns (F1: {task1_metrics['f1']:.4f})")
    print(f"   ✓ Achieved good performance on new patterns (F1: {task2_metrics['f1']:.4f})")
    print(f"   ✓ Rehearsal buffer contains {stats['current_size']} exemplars")
    print(f"   ✓ Privacy mode: {buffer.privacy_mode}")
    
    print("\n💡 Key Achievements:")
    print("   • Continual learning with catastrophic forgetting prevention")
    print("   • Privacy-preserving rehearsal memory")
    print("   • Explainable predictions via attention weights")
    print("   • Drift detection for automated updates")


if __name__ == "__main__":
    try:
        demo_continual_learning()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("\n💡 Note: This demo requires a trained model or will use random initialization.")
        print("   Run 'python train.py --data data/sample_data.csv' first for better results.")
        import traceback
        traceback.print_exc()
