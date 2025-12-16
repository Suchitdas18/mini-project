"""
Simplified training script for real hate-speech data
Works without wandb dependency
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
import yaml

# Simple Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

def train_model(data_path, epochs=3, batch_size=16, learning_rate=2e-5):
    """Train the hate-speech detection model"""
    
    print("=" * 80)
    print("TRAINING HATE-SPEECH DETECTION MODEL")
    print("=" * 80)
    
    # Load data
    print(f"\n📥 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} samples")
    
    # Label mapping
    label_map = {'neutral': 0, 'offensive': 1, 'hate_speech': 2}
    df['label_id'] = df['label'].map(label_map)
    
    print(f"\n📊 Label distribution:")
    print(df['label'].value_counts())
    
    # Split data
    print(f"\n🔀 Splitting data (80% train, 10% val, 10% test)...")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].values,
        df['label_id'].values,
        test_size=0.2,
        stratify=df['label_id'].values,
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
    print(f"\n🤖 Initializing RoBERTa model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=3,
        problem_type="single_label_classification"
    )
    model = model.to(device)
    
    # Create datasets
    print(f"\n📦 Creating dataloaders...")
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer)
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer)
    
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n🎓 Training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
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
        
        # Calculate training metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_dir = Path("models/best_model")
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"   ✅ Saved best model (F1: {val_f1:.4f})")
    
    # Final evaluation on test set
    print(f"\n🧪 Final evaluation on test set...")
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
    
    # Calculate test metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print(f"\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE!")
    print(f"=" * 80)
    print(f"\n📈 Final Test Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Macro F1: {test_f1:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    label_names = ['Neutral', 'Offensive', 'Hate Speech']
    print(classification_report(test_targets, test_preds, target_names=label_names))
    
    print(f"\n💾 Model saved to: models/best_model/")
    print(f"\n🚀 Your model is ready to use!")
    print(f"   Restart the web server: python app.py")
    print(f"   Then test at: http://localhost:5000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.batch_size, args.lr)
