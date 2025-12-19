"""
Quick training script - uses subset of data for fast testing
Perfect for CPU training and quick demonstrations
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import argparse

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(data_path, sample_size=2000, epochs=2, batch_size=16):
    """
    Quick training on subset of data
    
    Args:
        data_path: Path to full dataset
        sample_size: Number of samples to use (default: 2000 for speed)
        epochs: Number of training epochs (default: 2)
        batch_size: Batch size (default: 16)
    """
    
    print("=" * 80)
    print("QUICK TRAINING - HATE-SPEECH DETECTION")
    print("=" * 80)
    print(f"\n⚡ Fast training mode for CPU")
    print(f"   Using {sample_size:,} samples (subset of full dataset)")
    print(f"   This will complete in ~15-30 minutes on CPU")
    
    # Load data
    print(f"\n📥 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Full dataset: {len(df):,} samples")
    
    # Sample data for quick training
    print(f"\n🎲 Sampling {sample_size:,} samples for quick training...")
    
    # Stratified sampling to maintain class balance
    label_map = {'neutral': 0, 'offensive': 1, 'hate_speech': 2}
    df['label_id'] = df['label'].map(label_map)
    
    if len(df) > sample_size:
        # Simple random sampling from entire dataset
        df_sampled = df.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df
    
    print(f"✅ Using {len(df_sampled):,} samples")
    print(f"\n📊 Label distribution:")
    print(df_sampled['label'].value_counts())
    
    # Split data
    print(f"\n🔀 Splitting data (70% train, 15% val, 15% test)...")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df_sampled['text'].values,
        df_sampled['label_id'].values,
        test_size=0.3,
        stratify=df_sampled['label_id'].values,
        random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    print(f"   Train: {len(train_texts):,} samples")
    print(f"   Val:   {len(val_texts):,} samples")
    print(f"   Test:  {len(test_texts):,} samples")
    
    # Initialize model and tokenizer
    print(f"\n🤖 Initializing DistilRoBERTa model (lighter/faster)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Use DistilRoBERTa for faster training
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets with shorter max_length for speed
    print(f"\n📦 Creating dataloaders...")
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # Training loop
    print(f"\n🎓 Training for {epochs} epochs...")
    print("=" * 80)
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n>>> EPOCH {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Training metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Validation metrics
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train -> Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_dir = Path("models/best_model")
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  >>> Saved new best model! (F1: {val_f1:.4f})")
    
    # Test evaluation
    print(f"\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    # Test metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print(f"\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    label_names = ['Neutral', 'Offensive', 'Hate Speech']
    print(classification_report(test_targets, test_preds, target_names=label_names))
    
    print(f"\n💾 Model saved to: models/best_model/")
    print(f"\n📝 Note: This is a quick training on {sample_size:,} samples")
    print(f"   For production, train on full dataset ({len(df):,} samples)")
    print(f"   Use Google Colab with GPU for full training (see COLAB_TRAINING_GUIDE.md)")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Test the model: python demo.py")
    print(f"  2. Start web server: python app.py")
    print(f"  3. Visit: http://localhost:5000")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to use')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    train_model(args.data, args.samples, args.epochs, args.batch_size)
