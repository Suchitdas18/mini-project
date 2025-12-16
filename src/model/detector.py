"""
Core hate-speech detection model with continual learning capabilities
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdapterConfig,
)
from typing import Dict, List, Tuple, Optional
import numpy as np


class HateSpeechDetector(nn.Module):
    """
    Transformer-based hate-speech classifier with adapter layers
    for continual learning plasticity.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 3,
        use_adapters: bool = True,
        adapter_reduction_factor: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_adapters = use_adapters
        
        # Load pretrained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        
        # Add adapter layers if specified
        if use_adapters:
            adapter_config = AdapterConfig.load(
                "pfeiffer",
                reduction_factor=adapter_reduction_factor,
            )
            self.model.add_adapter("hate_speech", config=adapter_config)
            self.model.train_adapter("hate_speech")
            self.model.set_active_adapters("hate_speech")
        
        self.dropout = nn.Dropout(dropout)
        
        # Label mapping
        self.label_map = {
            0: "neutral",
            1: "offensive",
            2: "hate_speech"
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Dictionary containing loss, logits, and hidden states
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = True,
        return_embeddings: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on a list of texts
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
            return_embeddings: Whether to return embeddings
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predictions, probabilities, and optionally embeddings
        """
        self.eval()
        device = next(self.parameters()).device
        
        all_predictions = []
        all_probabilities = []
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                
                # Forward pass
                outputs = self.forward(input_ids, attention_mask)
                logits = outputs["logits"]
                
                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                
                if return_probabilities:
                    all_probabilities.extend(probs.cpu().numpy())
                
                if return_embeddings:
                    # Use [CLS] token embedding
                    embeddings = outputs["hidden_states"][-1][:, 0, :]
                    all_embeddings.extend(embeddings.cpu().numpy())
        
        results = {
            "predictions": np.array(all_predictions),
            "labels": [self.label_map[p] for p in all_predictions],
        }
        
        if return_probabilities:
            results["probabilities"] = np.array(all_probabilities)
            results["confidence"] = np.max(all_probabilities, axis=1)
        
        if return_embeddings:
            results["embeddings"] = np.array(all_embeddings)
        
        return results
    
    def get_attention_weights(
        self,
        text: str,
    ) -> Dict[str, any]:
        """
        Get attention weights for explainability
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokens and attention weights
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Get last layer attention
            attentions = outputs["attentions"][-1]
            
            # Average across heads and batch
            avg_attention = attentions.mean(dim=1).squeeze(0)
            
            # Get attention for CLS token
            cls_attention = avg_attention[0, 1:]  # Skip CLS itself
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
        return {
            "tokens": tokens[1:],  # Skip CLS
            "attention_weights": cls_attention.cpu().numpy(),
            "text": text,
        }
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Get text embeddings for drift detection and diversity sampling
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings [num_texts, hidden_dim]
        """
        predictions = self.predict(
            texts,
            return_probabilities=False,
            return_embeddings=True,
            batch_size=batch_size,
        )
        
        return predictions["embeddings"]
    
    def freeze_base_model(self):
        """Freeze base transformer, keep adapters trainable"""
        for name, param in self.model.named_parameters():
            if "adapter" not in name.lower():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_parameter_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for EWC
        
        Args:
            dataloader: DataLoader with samples for computing importance
            
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        self.eval()
        device = next(self.parameters()).device
        
        importance = {}
        for name, param in self.named_parameters():
            importance[name] = torch.zeros_like(param)
        
        num_samples = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = self.forward(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            
            # Backward pass
            self.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.data ** 2
            
            num_samples += len(batch)
        
        # Normalize by number of samples
        for name in importance:
            importance[name] /= num_samples
        
        return importance
    
    def save_model(self, save_path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path: str):
        """Load model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)


def create_detector(config: Dict) -> HateSpeechDetector:
    """
    Factory function to create detector from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized HateSpeechDetector
    """
    return HateSpeechDetector(
        model_name=config.get("base_model", "roberta-base"),
        num_labels=config.get("num_labels", 3),
        use_adapters=config.get("use_adapters", True),
        adapter_reduction_factor=config.get("adapter_reduction_factor", 16),
    )
