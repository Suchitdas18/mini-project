"""
Training script for initial model training
"""

import torch
import yaml
import argparse
from pathlib import Path
import wandb
from datetime import datetime

from src.model import create_detector
from src.data import load_hate_speech_data, split_data, create_dataloader
from src.evaluation import ContinualLearningMetrics
from tqdm import tqdm


def train_initial_model(config_path: str, data_path: str):
    """
    Train initial hate-speech detection model
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("CONTINUAL LEARNING HATE-SPEECH DETECTION - INITIAL TRAINING")
    print("=" * 60)
    
    # Initialize wandb
    if config["monitoring"]["use_wandb"]:
        wandb.init(
            project="hate-speech-continual-learning",
            config=config,
            name=f"initial_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    
    # Load data
    print(f"\n📚 Loading data from {data_path}...")
    texts, labels = load_hate_speech_data(data_path)
    print(f"   Total samples: {len(texts)}")
    
    # Split data
    print("\n✂️  Splitting data...")
    splits = split_data(texts, labels)
    train_texts, train_labels = splits["train"]
    val_texts, val_labels = splits["val"]
    test_texts, test_labels = splits["test"]
    
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Val:   {len(val_texts)} samples")
    print(f"   Test:  {len(test_texts)} samples")
    
    # Create model
    print("\n🤖 Initializing model...")
    model = create_detector(config["model"]).to(device)
    print(f"   Model: {config['model']['base_model']}")
    print(f"   Adapters: {config['model']['use_adapters']}")
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    train_loader = create_dataloader(
        train_texts,
        train_labels,
        model.tokenizer,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        val_texts,
        val_labels,
        model.tokenizer,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    num_epochs = config["training"]["num_epochs"]
    
    print(f"\n🎓 Training for {num_epochs} epochs...")
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config["training"]["max_grad_norm"],
            )
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track predictions
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute training metrics
        train_metrics = ContinualLearningMetrics.compute_f1_scores(
            predictions=train_preds,
            labels=train_true,
        )
        
        print(f"\n📊 Training Metrics:")
        print(f"   Loss:      {train_loss / len(train_loader):.4f}")
        print(f"   F1 Score:  {train_metrics['f1']:.4f}")
        print(f"   Precision: {train_metrics['precision']:.4f}")
        print(f"   Recall:    {train_metrics['recall']:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        # Compute validation metrics
        val_metrics = ContinualLearningMetrics.compute_f1_scores(
            predictions=val_preds,
            labels=val_true,
        )
        
        print(f"\n📊 Validation Metrics:")
        print(f"   F1 Score:  {val_metrics['f1']:.4f}")
        print(f"   Precision: {val_metrics['precision']:.4f}")
        print(f"   Recall:    {val_metrics['recall']:.4f}")
        
        # Log to wandb
        if config["monitoring"]["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
            })
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_path = Path(config["paths"]["model_store"]) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_path))
            print(f"\n💾 Saved best model (F1: {best_val_f1:.4f})")
    
    print(f"\n{'='*60}")
    print("✅ Training completed!")
    print(f"{'='*60}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    
    # Final evaluation on test set
    print("\n🧪 Final evaluation on test set...")
    test_loader = create_dataloader(
        test_texts,
        test_labels,
        model.tokenizer,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    # Test metrics
    test_metrics = ContinualLearningMetrics.compute_f1_scores(
        predictions=test_preds,
        labels=test_true,
    )
    
    print(f"\n📊 Test Metrics:")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    
    # Per-class metrics
    per_class = ContinualLearningMetrics.compute_per_class_metrics(
        predictions=test_preds,
        labels=test_true,
    )
    
    print(f"\n📊 Per-Class Metrics:")
    for class_name, metrics in per_class.items():
        print(f"   {class_name}:")
        print(f"      F1:        {metrics['f1']:.4f}")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
    
    # Save classification report
    report = ContinualLearningMetrics.generate_classification_report(
        predictions=test_preds,
        labels=test_true,
    )
    
    results_dir = Path(config["paths"]["validation_results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "initial_training_report.txt", "w") as f:
        f.write(report)
    
    print(f"\n📄 Classification report saved to {results_dir / 'initial_training_report.txt'}")
    
    if config["monitoring"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train initial hate-speech detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data CSV file",
    )
    
    args = parser.parse_args()
    
    train_initial_model(args.config, args.data)
