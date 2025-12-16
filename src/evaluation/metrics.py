"""
Evaluation metrics for continual learning
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ContinualLearningMetrics:
    """
    Metrics for evaluating continual learning performance
    """

    @staticmethod
    def compute_accuracy(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute accuracy"""
        return accuracy_score(labels, predictions)
    
    @staticmethod
    def compute_f1_scores(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "macro",
    ) -> Dict[str, float]:
        """
        Compute F1 scores
        
        Args:
            predictions: Predicted labels
            labels: True labels
            average: Averaging strategy
            
        Returns:
            Dictionary with precision, recall, f1
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average=average,
            zero_division=0,
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    @staticmethod
    def compute_per_class_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Names of classes
            
        Returns:
            Dictionary with per-class metrics
        """
        if class_names is None:
            class_names = ["neutral", "offensive", "hate_speech"]
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0,
        )
        
        metrics = {}
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            }
        
        return metrics
    
    @staticmethod
    def compute_backward_transfer(
        task_performances: List[List[float]],
    ) -> float:
        """
        Compute Backward Transfer (BWT)
        Measures forgetting of previous tasks
        
        Args:
            task_performances: Matrix where task_performances[i][j] is
                              performance on task j after training on task i
        
        Returns:
            Backward transfer score (negative = forgetting)
        """
        n_tasks = len(task_performances)
        
        if n_tasks < 2:
            return 0.0
        
        bwt_sum = 0.0
        count = 0
        
        for i in range(1, n_tasks):
            for j in range(i):
                # Performance on task j after training on task i
                # vs performance on task j immediately after training on task j
                bwt_sum += task_performances[i][j] - task_performances[j][j]
                count += 1
        
        return bwt_sum / count if count > 0 else 0.0
    
    @staticmethod
    def compute_forward_transfer(
        task_performances: List[List[float]],
        random_baseline: List[float],
    ) -> float:
        """
        Compute Forward Transfer (FWT)
        Measures ability to leverage past knowledge for new tasks
        
        Args:
            task_performances: Performance matrix
            random_baseline: Random initialization performance on each task
            
        Returns:
            Forward transfer score (positive = positive transfer)
        """
        n_tasks = len(task_performances)
        
        if n_tasks < 2:
            return 0.0
        
        fwt_sum = 0.0
        count = 0
        
        for i in range(1, n_tasks):
            # Performance on task i before training on it (using model from i-1)
            # vs random baseline
            fwt_sum += task_performances[i-1][i] - random_baseline[i]
            count += 1
        
        return fwt_sum / count if count > 0 else 0.0
    
    @staticmethod
    def compute_average_forgetting(
        task_performances: List[List[float]],
    ) -> float:
        """
        Compute average forgetting across all tasks
        
        Args:
            task_performances: Performance matrix
            
        Returns:
            Average forgetting (lower is better)
        """
        n_tasks = len(task_performances)
        
        if n_tasks < 2:
            return 0.0
        
        forgetting_sum = 0.0
        count = 0
        
        for j in range(n_tasks - 1):
            # Max performance achieved on task j
            max_perf = max([task_performances[i][j] for i in range(j, n_tasks)])
            
            # Final performance on task j
            final_perf = task_performances[-1][j]
            
            forgetting_sum += max_perf - final_perf
            count += 1
        
        return forgetting_sum / count if count > 0 else 0.0
    
    @staticmethod
    def plot_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None,
    ):
        """
        Plot confusion matrix
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Names of classes
            save_path: Path to save plot
        """
        if class_names is None:
            class_names = ["neutral", "offensive", "hate_speech"]
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_task_performance_matrix(
        task_performances: List[List[float]],
        task_names: List[str] = None,
        save_path: str = None,
    ):
        """
        Visualize task performance matrix
        
        Args:
            task_performances: Performance matrix
            task_names: Names of tasks
            save_path: Path to save plot
        """
        n_tasks = len(task_performances)
        
        if task_names is None:
            task_names = [f"Task {i+1}" for i in range(n_tasks)]
        
        # Convert to numpy array
        perf_matrix = np.array(task_performances)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            perf_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=task_names,
            yticklabels=[f"After {t}" for t in task_names],
            vmin=0,
            vmax=1,
        )
        plt.title("Task Performance Matrix")
        plt.ylabel("Training Stage")
        plt.xlabel("Evaluated Task")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_classification_report(
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None,
    ) -> str:
        """
        Generate detailed classification report
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Names of classes
            
        Returns:
            Classification report string
        """
        if class_names is None:
            class_names = ["neutral", "offensive", "hate_speech"]
        
        return classification_report(
            labels,
            predictions,
            target_names=class_names,
            digits=4,
        )


class FairnessMetrics:
    """
    Metrics for evaluating fairness across demographic groups
    """

    @staticmethod
    def compute_false_positive_disparity(
        predictions: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute false positive rate disparity across groups
        
        Args:
            predictions: Predicted labels
            labels: True labels
            groups: Group memberships
            
        Returns:
            Dictionary with FPR per group and max disparity
        """
        unique_groups = np.unique(groups)
        fprs = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            # True negatives and false positives
            tn = np.sum((group_preds == 0) & (group_labels == 0))
            fp = np.sum((group_preds > 0) & (group_labels == 0))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fprs[f"group_{group}"] = fpr
        
        # Max disparity
        fpr_values = list(fprs.values())
        max_disparity = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        fprs["max_disparity"] = max_disparity
        
        return fprs
    
    @staticmethod
    def compute_demographic_parity(
        predictions: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute demographic parity (equal positive rate across groups)
        
        Args:
            predictions: Predicted labels
            groups: Group memberships
            
        Returns:
            Dictionary with positive rates and disparity
        """
        unique_groups = np.unique(groups)
        positive_rates = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_preds = predictions[group_mask]
            
            positive_rate = (group_preds > 0).mean()
            positive_rates[f"group_{group}"] = positive_rate
        
        # Max disparity
        pr_values = list(positive_rates.values())
        max_disparity = max(pr_values) - min(pr_values) if pr_values else 0
        
        positive_rates["max_disparity"] = max_disparity
        
        return positive_rates
