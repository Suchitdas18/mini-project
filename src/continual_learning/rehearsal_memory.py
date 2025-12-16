"""
Rehearsal memory for continual learning with privacy-preserving storage
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle
import json
from pathlib import Path
import hashlib


class RehearsalBuffer:
    """
    Memory buffer for storing representative samples to prevent catastrophic forgetting
    Supports multiple storage strategies: raw text, embeddings only, or synthetic prototypes
    """

    def __init__(
        self,
        capacity: int = 10000,
        privacy_mode: str = "embedding_only",
        class_balance: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize rehearsal buffer
        
        Args:
            capacity: Maximum number of samples to store
            privacy_mode: Storage mode - 'raw_text', 'embedding_only', or 'synthetic'
            class_balance: Target class distribution {label: proportion}
        """
        self.capacity = capacity
        self.privacy_mode = privacy_mode
        self.class_balance = class_balance or {
            "hate_speech": 0.30,
            "offensive": 0.20,
            "neutral": 0.50,
        }
        
        # Storage
        self.samples = []
        self.labels = []
        self.embeddings = []
        self.metadata = []
        self.insertion_order = []
        
        # Statistics
        self.total_seen = 0
        self.class_counts = defaultdict(int)
    
    def add(
        self,
        text: Optional[str] = None,
        label: str = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        score: float = 1.0,
    ):
        """
        Add a single sample to the buffer using reservoir sampling
        
        Args:
            text: Raw text (only stored if privacy_mode != 'embedding_only')
            label: Class label
            embedding: Text embedding vector
            metadata: Additional metadata (timestamp, source, etc.)
            score: Importance score for weighted reservoir sampling
        """
        self.total_seen += 1
        
        # Determine class-specific capacity
        class_capacity = int(self.capacity * self.class_balance[label])
        class_current = self.class_counts[label]
        
        # Reservoir sampling with replacement probability
        if class_current < class_capacity:
            # Always add if below capacity
            accept = True
        else:
            # Probabilistic replacement
            accept_prob = class_capacity / self.total_seen
            accept = np.random.random() < accept_prob * score
        
        if accept:
            sample = {
                "text": text if self.privacy_mode == "raw_text" else self._hash_text(text),
                "label": label,
                "embedding": embedding,
                "metadata": metadata or {},
                "score": score,
                "timestamp": metadata.get("timestamp") if metadata else None,
            }
            
            if len(self.samples) < self.capacity:
                # Buffer not full, append
                self.samples.append(sample)
                self.class_counts[label] += 1
            else:
                # Buffer full, replace random sample of same class
                same_class_indices = [
                    i for i, s in enumerate(self.samples) if s["label"] == label
                ]
                
                if same_class_indices:
                    replace_idx = np.random.choice(same_class_indices)
                    self.samples[replace_idx] = sample
    
    def add_batch(
        self,
        texts: Optional[List[str]] = None,
        labels: List[str] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
        scores: Optional[List[float]] = None,
    ):
        """
        Add multiple samples to the buffer
        
        Args:
            texts: List of raw texts
            labels: List of labels
            embeddings: Array of embeddings [batch_size, embedding_dim]
            metadata: List of metadata dicts
            scores: List of importance scores
        """
        batch_size = len(labels)
        
        if texts is None:
            texts = [None] * batch_size
        if embeddings is None:
            embeddings = [None] * batch_size
        if metadata is None:
            metadata = [{}] * batch_size
        if scores is None:
            scores = [1.0] * batch_size
        
        for i in range(batch_size):
            self.add(
                text=texts[i] if texts[i] is not None else None,
                label=labels[i],
                embedding=embeddings[i] if embeddings is not None else None,
                metadata=metadata[i],
                score=scores[i],
            )
    
    def sample(
        self,
        size: int,
        strategy: str = "random",
        label: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Sample from the buffer
        
        Args:
            size: Number of samples to draw
            strategy: Sampling strategy - 'random', 'balanced', or 'recent'
            label: If specified, sample only from this class
            
        Returns:
            Dictionary with sampled data
        """
        if label is not None:
            # Filter by label
            eligible_indices = [
                i for i, s in enumerate(self.samples) if s["label"] == label
            ]
        else:
            eligible_indices = list(range(len(self.samples)))
        
        if len(eligible_indices) == 0:
            return self._empty_batch()
        
        # Clamp size
        size = min(size, len(eligible_indices))
        
        if strategy == "random":
            sampled_indices = np.random.choice(eligible_indices, size, replace=False)
        
        elif strategy == "balanced":
            # Sample proportionally from each class
            sampled_indices = []
            for class_label, proportion in self.class_balance.items():
                class_indices = [
                    i for i in eligible_indices if self.samples[i]["label"] == class_label
                ]
                if class_indices:
                    class_size = int(size * proportion)
                    class_size = min(class_size, len(class_indices))
                    sampled = np.random.choice(class_indices, class_size, replace=False)
                    sampled_indices.extend(sampled)
        
        elif strategy == "recent":
            # Sample recent additions
            sorted_indices = sorted(
                eligible_indices,
                key=lambda i: self.samples[i]["metadata"].get("timestamp", 0),
                reverse=True,
            )
            sampled_indices = sorted_indices[:size]
        
        # Extract samples
        sampled_data = [self.samples[i] for i in sampled_indices]
        
        return {
            "texts": [s["text"] for s in sampled_data],
            "labels": [s["label"] for s in sampled_data],
            "embeddings": np.array([s["embedding"] for s in sampled_data if s["embedding"] is not None]),
            "metadata": [s["metadata"] for s in sampled_data],
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get buffer statistics"""
        return {
            "current_size": len(self.samples),
            "capacity": self.capacity,
            "total_seen": self.total_seen,
            "class_distribution": dict(self.class_counts),
            "privacy_mode": self.privacy_mode,
        }
    
    def save(self, save_path: str):
        """Save buffer to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "samples": self.samples,
                    "capacity": self.capacity,
                    "privacy_mode": self.privacy_mode,
                    "class_balance": self.class_balance,
                    "total_seen": self.total_seen,
                    "class_counts": dict(self.class_counts),
                },
                f,
            )
    
    def load(self, load_path: str):
        """Load buffer from disk"""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self.samples = data["samples"]
        self.capacity = data["capacity"]
        self.privacy_mode = data["privacy_mode"]
        self.class_balance = data["class_balance"]
        self.total_seen = data["total_seen"]
        self.class_counts = defaultdict(int, data["class_counts"])
    
    def clear(self):
        """Clear all samples"""
        self.samples = []
        self.total_seen = 0
        self.class_counts = defaultdict(int)
    
    def _hash_text(self, text: str) -> str:
        """Create privacy-preserving hash of text"""
        if text is None:
            return None
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _empty_batch(self) -> Dict[str, any]:
        """Return empty batch structure"""
        return {
            "texts": [],
            "labels": [],
            "embeddings": np.array([]),
            "metadata": [],
        }


class ExemplarSelector:
    """
    Select high-value examples for rehearsal based on multiple criteria
    """

    @staticmethod
    def select_boundary_cases(
        predictions: np.ndarray,
        labels: List[str],
        top_k: int = 100,
    ) -> List[int]:
        """
        Select samples near decision boundary (high uncertainty)
        
        Args:
            predictions: Probability distributions [num_samples, num_classes]
            labels: True labels
            top_k: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        # Compute entropy
        epsilon = 1e-10
        entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
        
        # Select top-k highest entropy
        top_indices = np.argsort(entropy)[-top_k:]
        
        return top_indices.tolist()
    
    @staticmethod
    def select_rare_classes(
        labels: List[str],
        class_counts: Dict[str, int],
        top_k: int = 100,
    ) -> List[int]:
        """
        Prioritize samples from underrepresented classes
        
        Args:
            labels: List of labels
            class_counts: Current class distribution
            top_k: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        # Compute inverse frequency scores
        scores = []
        for label in labels:
            score = 1.0 / (class_counts.get(label, 1) + 1)
            scores.append(score)
        
        scores = np.array(scores)
        top_indices = np.argsort(scores)[-top_k:]
        
        return top_indices.tolist()
    
    @staticmethod
    def select_diverse_samples(
        embeddings: np.ndarray,
        top_k: int = 100,
        method: str = "maximin",
    ) -> List[int]:
        """
        Select diverse samples to maximize coverage
        
        Args:
            embeddings: Sample embeddings [num_samples, embedding_dim]
            top_k: Number of samples to select
            method: Selection method - 'maximin' or 'kmeans'
            
        Returns:
            Indices of selected samples
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.cluster import KMeans
        
        if method == "maximin":
            # Greedy maximin selection
            selected = []
            remaining = list(range(len(embeddings)))
            
            # Start with random sample
            first_idx = np.random.choice(remaining)
            selected.append(first_idx)
            remaining.remove(first_idx)
            
            for _ in range(top_k - 1):
                if not remaining:
                    break
                
                # Compute min distance to selected set
                selected_embs = embeddings[selected]
                remaining_embs = embeddings[remaining]
                
                distances = euclidean_distances(remaining_embs, selected_embs)
                min_distances = distances.min(axis=1)
                
                # Select sample with maximum min-distance
                best_idx = remaining[np.argmax(min_distances)]
                selected.append(best_idx)
                remaining.remove(best_idx)
            
            return selected
        
        elif method == "kmeans":
            # K-means clustering
            kmeans = KMeans(n_clusters=top_k, random_state=42)
            kmeans.fit(embeddings)
            
            # Select nearest sample to each centroid
            selected = []
            for centroid in kmeans.cluster_centers_:
                distances = euclidean_distances([centroid], embeddings)[0]
                nearest_idx = np.argmin(distances)
                selected.append(nearest_idx)
            
            return selected
    
    @staticmethod
    def select_combined(
        texts: List[str],
        labels: List[str],
        predictions: np.ndarray,
        embeddings: np.ndarray,
        class_counts: Dict[str, int],
        top_k: int = 100,
        weights: Dict[str, float] = None,
    ) -> List[int]:
        """
        Combine multiple selection criteria
        
        Args:
            texts: Input texts
            labels: True labels
            predictions: Model predictions  
            embeddings: Text embeddings
            class_counts: Class distribution
            top_k: Number to select
            weights: Weights for each criterion
            
        Returns:
            Indices of selected samples
        """
        weights = weights or {
            "boundary": 0.3,
            "rare_class": 0.3,
            "diversity": 0.4,
        }
        
        num_samples = len(labels)
        scores = np.zeros(num_samples)
        
        # Boundary score (entropy)
        if weights.get("boundary", 0) > 0:
            epsilon = 1e-10
            entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
            entropy_normalized = entropy / entropy.max()
            scores += weights["boundary"] * entropy_normalized
        
        # Rare class score
        if weights.get("rare_class", 0) > 0:
            rare_scores = np.array([1.0 / (class_counts.get(l, 1) + 1) for l in labels])
            rare_scores_normalized = rare_scores / rare_scores.max()
            scores += weights["rare_class"] * rare_scores_normalized
        
        # Diversity score (distance to mean embedding)
        if weights.get("diversity", 0) > 0:
            mean_embedding = embeddings.mean(axis=0)
            distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
            distances_normalized = distances / distances.max()
            scores += weights["diversity"] * distances_normalized
        
        # Select top-k
        top_indices = np.argsort(scores)[-top_k:]
        
        return top_indices.tolist()
