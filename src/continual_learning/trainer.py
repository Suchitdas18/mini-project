"""
Continual learning trainer with EWC, knowledge distillation, and rehearsal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import copy

from .rehearsal_memory import RehearsalBuffer


class ContinualLearningTrainer:
    """
    Trainer for continual learning with combined:
    - Memory-based rehearsal
    - Knowledge distillation
    - Elastic Weight Consolidation (EWC)
    """

    def __init__(
        self,
        model: nn.Module,
        rehearsal_buffer: RehearsalBuffer,
        lambda_distill: float = 0.5,
        lambda_ewc: float = 0.3,
        temperature: float = 2.0,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize continual learning trainer
        
        Args:
            model: HateSpeechDetector model
            rehearsal_buffer: Rehearsal memory buffer
            lambda_distill: Weight for distillation loss
            lambda_ewc: Weight for EWC regularization
            temperature: Temperature for knowledge distillation
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            device: Training device
        """
        self.model = model.to(device)
        self.rehearsal_buffer = rehearsal_buffer
        self.lambda_distill = lambda_distill
        self.lambda_ewc = lambda_ewc
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Previous model for distillation
        self.previous_model = None
        
        # Fisher information for EWC
        self.fisher_information = None
        self.previous_parameters = None
    
    def train_step(
        self,
        new_data: Dict[str, any],
        rehearsal_ratio: float = 0.5,
    ) -> Dict[str, float]:
        """
        Perform one continual learning update step
        
        Args:
            new_data: Dictionary with 'texts' and 'labels'
            rehearsal_ratio: Proportion of rehearsal samples in each batch
            
        Returns:
            Dictionary of training metrics
        """
        # Store previous model for distillation
        self.previous_model = copy.deepcopy(self.model)
        self.previous_model.eval()
        
        # Compute Fisher information if we have previous parameters
        if self.rehearsal_buffer.get_statistics()["current_size"] > 0:
            self._compute_fisher_information()
        
        # Prepare combined dataset (new + rehearsal)
        train_loader = self._prepare_dataloader(new_data, rehearsal_ratio)
        
        # Training loop
        metrics = self._training_loop(train_loader)
        
        return metrics
    
    def _prepare_dataloader(
        self,
        new_data: Dict[str, any],
        rehearsal_ratio: float,
    ) -> DataLoader:
        """
        Combine new data with rehearsal samples
        
        Args:
            new_data: New labeled data
            rehearsal_ratio: Proportion of rehearsal samples
            
        Returns:
            Combined DataLoader
        """
        new_texts = new_data["texts"]
        new_labels = new_data["labels"]
        
        # Sample from rehearsal buffer
        rehearsal_size = int(len(new_texts) * rehearsal_ratio / (1 - rehearsal_ratio))
        rehearsal_data = self.rehearsal_buffer.sample(
            size=rehearsal_size,
            strategy="balanced",
        )
        
        # Combine datasets
        all_texts = new_texts + rehearsal_data["texts"]
        all_labels = new_labels + rehearsal_data["labels"]
        
        # Tokenize
        tokenizer = self.model.tokenizer
        encoded = tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Convert labels to integers
        label_map = self.model.reverse_label_map
        label_ids = torch.tensor([label_map[l] for l in all_labels])
        
        # Create dataset
        dataset = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            label_ids,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        return dataloader
    
    def _training_loop(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Execute training loop with combined loss
        
        Args:
            train_loader: Training DataLoader
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_task_loss = 0
        total_distill_loss = 0
        total_ewc_loss = 0
        num_batches = 0
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in pbar:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                # Task loss (cross-entropy)
                task_loss = outputs["loss"]
                
                # Knowledge distillation loss
                distill_loss = torch.tensor(0.0).to(self.device)
                if self.previous_model is not None:
                    with torch.no_grad():
                        prev_outputs = self.previous_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        prev_logits = prev_outputs["logits"]
                    
                    # Soft targets from previous model
                    soft_targets = F.softmax(prev_logits / self.temperature, dim=-1)
                    soft_predictions = F.log_softmax(
                        outputs["logits"] / self.temperature,
                        dim=-1,
                    )
                    
                    distill_loss = F.kl_div(
                        soft_predictions,
                        soft_targets,
                        reduction="batchmean",
                    ) * (self.temperature ** 2)
                
                # EWC regularization loss
                ewc_loss = torch.tensor(0.0).to(self.device)
                if self.fisher_information is not None:
                    for name, param in self.model.named_parameters():
                        if name in self.fisher_information:
                            fisher = self.fisher_information[name]
                            prev_param = self.previous_parameters[name]
                            ewc_loss += (fisher * (param - prev_param) ** 2).sum()
                
                # Combined loss
                loss = (
                    task_loss
                    + self.lambda_distill * distill_loss
                    + self.lambda_ewc * ewc_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_task_loss += task_loss.item()
                total_distill_loss += distill_loss.item()
                total_ewc_loss += ewc_loss.item()
                num_batches += 1
                epoch_loss += loss.item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "task": f"{task_loss.item():.4f}",
                    "distill": f"{distill_loss.item():.4f}",
                    "ewc": f"{ewc_loss.item():.4f}",
                })
        
        metrics = {
            "total_loss": total_loss / num_batches,
            "task_loss": total_task_loss / num_batches,
            "distillation_loss": total_distill_loss / num_batches,
            "ewc_loss": total_ewc_loss / num_batches,
        }
        
        return metrics
    
    def _compute_fisher_information(self):
        """
        Compute Fisher Information Matrix for EWC
        Uses rehearsal buffer samples
        """
        # Sample from rehearsal buffer
        rehearsal_data = self.rehearsal_buffer.sample(
            size=min(1000, self.rehearsal_buffer.get_statistics()["current_size"]),
            strategy="balanced",
        )
        
        if len(rehearsal_data["texts"]) == 0:
            return
        
        # Tokenize
        tokenizer = self.model.tokenizer
        encoded = tokenizer(
            rehearsal_data["texts"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        label_map = self.model.reverse_label_map
        label_ids = torch.tensor([label_map[l] for l in rehearsal_data["labels"]])
        
        # Create dataset
        dataset = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            label_ids,
        )
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Initialize Fisher information
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
        
        # Store current parameters
        self.previous_parameters = {}
        for name, param in self.model.named_parameters():
            self.previous_parameters[name] = param.clone().detach()
        
        # Compute Fisher
        self.model.eval()
        num_samples = 0
        
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
            
            num_samples += len(batch)
        
        # Normalize by number of samples
        for name in self.fisher_information:
            self.fisher_information[name] /= num_samples
        
        self.model.train()
    
    def update_rehearsal_buffer(
        self,
        texts: List[str],
        labels: List[str],
        selection_strategy: str = "combined",
    ):
        """
        Add new exemplars to rehearsal buffer
        
        Args:
            texts: New texts
            labels: Corresponding labels
            selection_strategy: How to select exemplars
        """
        # Get embeddings and predictions
        results = self.model.predict(
            texts,
            return_probabilities=True,
            return_embeddings=True,
        )
        
        embeddings = results["embeddings"]
        probabilities = results["probabilities"]
        
        # Select exemplars based on strategy
        from .rehearsal_memory import ExemplarSelector
        
        if selection_strategy == "boundary":
            indices = ExemplarSelector.select_boundary_cases(
                probabilities,
                labels,
                top_k=min(1000, len(texts)),
            )
        
        elif selection_strategy == "diverse":
            indices = ExemplarSelector.select_diverse_samples(
                embeddings,
                top_k=min(1000, len(texts)),
            )
        
        elif selection_strategy == "combined":
            class_counts = self.rehearsal_buffer.class_counts
            indices = ExemplarSelector.select_combined(
                texts,
                labels,
                probabilities,
                embeddings,
                class_counts,
                top_k=min(1000, len(texts)),
            )
        
        else:
            # Random selection
            indices = np.random.choice(len(texts), min(1000, len(texts)), replace=False)
        
        # Add selected samples to buffer
        selected_texts = [texts[i] for i in indices]
        selected_labels = [labels[i] for i in indices]
        selected_embeddings = embeddings[indices]
        
        self.rehearsal_buffer.add_batch(
            texts=selected_texts if self.rehearsal_buffer.privacy_mode == "raw_text" else None,
            labels=selected_labels,
            embeddings=selected_embeddings,
        )


class DriftDetector:
    """
    Detect distribution drift to trigger model updates
    """

    def __init__(
        self,
        baseline_model: nn.Module,
        drift_threshold: float = 0.25,
    ):
        """
        Initialize drift detector
        
        Args:
            baseline_model: Reference model for comparison
            drift_threshold: Threshold for triggering update
        """
        self.baseline_model = baseline_model
        self.baseline_model.eval()
        self.drift_threshold = drift_threshold
    
    def compute_drift(
        self,
        current_model: nn.Module,
        texts: List[str],
    ) -> float:
        """
        Compute drift score based on prediction disagreement
        
        Args:
            current_model: Current model
            texts: Sample of recent texts
            
        Returns:
            Drift score (0-1)
        """
        current_model.eval()
        
        # Get predictions from both models
        with torch.no_grad():
            current_results = current_model.predict(texts, return_embeddings=True)
            baseline_results = self.baseline_model.predict(texts, return_embeddings=True)
        
        # Prediction disagreement
        pred_disagreement = (
            current_results["predictions"] != baseline_results["predictions"]
        ).mean()
        
        # Embedding shift (cosine distance)
        current_emb_mean = current_results["embeddings"].mean(axis=0)
        baseline_emb_mean = baseline_results["embeddings"].mean(axis=0)
        
        cosine_sim = np.dot(current_emb_mean, baseline_emb_mean) / (
            np.linalg.norm(current_emb_mean) * np.linalg.norm(baseline_emb_mean)
        )
        embedding_drift = 1 - cosine_sim
        
        # Combined drift score
        drift_score = 0.5 * pred_disagreement + 0.5 * embedding_drift
        
        return drift_score
    
    def should_update(self, drift_score: float) -> bool:
        """Check if drift exceeds threshold"""
        return drift_score > self.drift_threshold
