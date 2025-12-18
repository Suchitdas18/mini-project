"""
Google Colab Training Script for Hate-Speech Detection
This script is optimized for GPU training on Google Colab.
Expected training time: 2-3 hours (vs 70+ hours on CPU)
"""

# Installation and Setup
print("📦 Installing dependencies...")
import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "transformers", "torch", "datasets", "scikit-learn", 
                      "tqdm", "pandas", "numpy"])

# Imports
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm
import json
import os

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Configuration
CONFIG = {
    "model_name": "roberta-base",
    "num_labels": 3,
    "max_length": 128,
    "batch_size": 16,  # Increased for GPU
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "warmup_steps": 500,
    "save_dir": "hate_speech_model",
}

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
            add_special_tokens=True,
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

def load_davidson_dataset():
    """Load the Davidson Hate-Speech dataset"""
    print("📥 Loading Davidson Hate-Speech dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", "binary")
    
    # Alternatively, if that doesn't work, use a CSV approach
    # You can upload the dataset to Colab or download from a URL
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Extract texts and labels
    # Adjust field names based on the actual dataset structure
    train_texts = train_data['text']
    train_labels = train_data['label']
    
    test_texts = test_data['text']
    test_labels = test_data['label']
    
    print(f"✅ Loaded {len(train_texts)} training samples")
    print(f"✅ Loaded {len(test_texts)} test samples")
    
    return train_texts, train_labels, test_texts, test_labels

def train_model(model, train_loader, optimizer, device, epoch):
    """Training loop"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, accuracy, f1

def evaluate_model(model, test_loader, device):
    """Evaluation loop"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    
    return avg_loss, accuracy, f1

def main():
    print("\n" + "="*50)
    print("🚀 HATE-SPEECH DETECTION TRAINING")
    print("="*50 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}\n")
    
    # Load tokenizer and model
    print("📦 Loading RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG['model_name'])
    model = RobertaForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(device)
    print("✅ Model loaded\n")
    
    # Load dataset
    try:
        train_texts, train_labels, test_texts, test_labels = load_davidson_dataset()
    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}")
        print("Using sample data for demonstration...")
        # Fallback to sample data
        train_texts = ["This is hate speech"] * 1000
        train_labels = [0] * 1000
        test_texts = ["This is normal text"] * 200
        test_labels = [1] * 200
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Test batches: {len(test_loader)}\n")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print("🏋️ Starting training...\n")
    best_f1 = 0
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*50}\n")
        
        # Train
        train_loss, train_acc, train_f1 = train_model(model, train_loader, optimizer, device, epoch)
        
        print(f"\n📈 Training Results:")
        print(f"   Loss: {train_loss:.4f}")
        print(f"   Accuracy: {train_acc:.4f}")
        print(f"   Macro F1: {train_f1:.4f}")
        
        # Evaluate
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device)
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            print(f"\n💾 New best model! F1: {test_f1:.4f}")
            
            # Save model
            os.makedirs(CONFIG['save_dir'], exist_ok=True)
            model.save_pretrained(CONFIG['save_dir'])
            tokenizer.save_pretrained(CONFIG['save_dir'])
            
            # Save metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
            }
            
            with open(os.path.join(CONFIG['save_dir'], 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print(f"🎯 Best F1 Score: {best_f1:.4f}")
    print(f"📁 Model saved to: {CONFIG['save_dir']}")
    print("\n💡 Download the model folder to use in your local project!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
