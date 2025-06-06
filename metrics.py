# pylint: disable=no-member
import numpy as np
import torch
from torchmetrics import Metric
from typing import Container

from go_wrapper import GOGraph


def compute_f_betas(precision: torch.Tensor, recall: torch.Tensor, device: torch.device, beta: float = 1.0, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Compute the F-beta score for a matched pair of tensors of precision and recall.

    Args:
        precision: The precision tensor.
        recall: The recall tensor.
        device: The device to compute the F-beta score on.
        beta: The beta parameter for the F-beta score.
        epsilon: The epsilon parameter for the F-beta score (to avoid division by zero).
    """
    beta_squared = torch.tensor(beta**2).to(device)
    epsilon = torch.tensor(epsilon).to(device)
    return (torch.tensor(1).to(device) + beta_squared) * (precision * recall) / (beta_squared * precision + recall + epsilon)

def apply_prob_threshold(all_probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a probability threshold to a matrix of probabilities.

    Args:
        all_probs: The matrix of probabilities.
        threshold: The threshold to apply.

    Returns:
        A boolean matrix indicating whether a given term index (column) is predicted to apply to each sample (row).
    """
    return all_probs >= threshold


def get_predicted_go_terms(prediction_matrix: np.ndarray, label_indices_to_terms: dict[int, str]) -> list[list[str]]:
    """
    Get the predicted GO terms for a given prediction matrix.

    Args:
        prediction_matrix: A matrix of shape (num_samples, num_terms) where each entry is a boolean indicating whether the term is predicted to apply to the sample.
        label_indices_to_terms: A dictionary mapping term indices to GO terms.

    Returns:
        A list of lists of predicted GO terms, one list per sample.
    """
    # Convert dictionary to array for faster lookup
    terms_array = np.array([label_indices_to_terms[i] for i in range(len(label_indices_to_terms))])
    
    # Get indices where predictions are True
    predicted_indices = np.where(prediction_matrix)[1]
    
    # Group indices by sample
    sample_sizes = np.sum(prediction_matrix, axis=1)
    split_indices = np.cumsum(sample_sizes[:-1])
    grouped_indices = np.split(predicted_indices, split_indices)
    
    # Convert indices to terms and sort
    return [sorted(terms_array[indices]) for indices in grouped_indices]


class SimplePrecision(Metric):
    """
    Simple precision metric that computes precision over all predictions at once.
    This is more efficient than computing per-label precision and averaging.
    """
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        Initialize the metric.
        
        Args:
            threshold: The threshold to use for converting probabilities to binary predictions.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds: Predicted probabilities of shape [N, C] where N is the batch size and C is the number of labels.
            target: Ground truth labels of shape [N, C] where N is the batch size and C is the number of labels.
        """
        # Convert probabilities to binary predictions using threshold
        preds = (preds > self.threshold).float()
        
        # Flatten predictions and targets to treat all predictions as one set
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        
        # Compute true positives and false positives
        true_positives = (preds_flat * target_flat).sum()
        false_positives = (preds_flat * (1 - target_flat)).sum()
        
        # Update state variables
        self.true_positives += true_positives
        self.false_positives += false_positives
    
    def compute(self) -> torch.Tensor:
        """
        Compute the precision from the accumulated statistics.
        
        Returns:
            The computed precision.
        """
        return self.true_positives / (self.true_positives + self.false_positives + 1e-10)


class SimpleRecall(Metric):
    """
    Simple recall metric that computes recall over all predictions at once.
    This is more efficient than computing per-label recall and averaging.
    """
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        """
        Initialize the metric.
        
        Args:
            threshold: The threshold to use for converting probabilities to binary predictions.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds: Predicted probabilities of shape [N, C] where N is the batch size and C is the number of labels.
            target: Ground truth labels of shape [N, C] where N is the batch size and C is the number of labels.
        """
        # Convert probabilities to binary predictions using threshold
        preds = (preds > self.threshold).float()
        
        # Flatten predictions and targets to treat all predictions as one set
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        
        # Compute true positives and false negatives
        true_positives = (preds_flat * target_flat).sum()
        false_negatives = ((1 - preds_flat) * target_flat).sum()
        
        # Update state variables
        self.true_positives += true_positives
        self.false_negatives += false_negatives
    
    def compute(self) -> torch.Tensor:
        """
        Compute the recall from the accumulated statistics.
        
        Returns:
            The computed recall.
        """
        return self.true_positives / (self.true_positives + self.false_negatives + 1e-10)


class PrecisionRecallCurve(Metric):
    """
    Computes precision and recall across multiple thresholds.
    This allows for plotting a precision-recall curve and finding the optimal threshold.
    """
    
    def __init__(self, num_thresholds: int = 100, epsilon: float = 1e-10, **kwargs):
        """
        Initialize the metric.
        
        Args:
            num_thresholds: Number of thresholds to evaluate between 0 and 1.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        
        # Create evenly spaced thresholds between 0 and 1
        self.thresholds = torch.linspace(0, 1, num_thresholds)
        self.epsilon = epsilon
        
        # Add state variables for each threshold
        self.add_state('true_positives', default=torch.zeros(num_thresholds), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_thresholds), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_thresholds), dist_reduce_fx='sum')
    
    @property
    def epsilon_tensor(self) -> torch.Tensor:
        """Get epsilon as a tensor on the current device"""
        return torch.tensor(self.epsilon, device=self.device)
 
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions and targets.
        
        Args:
            preds: Predicted probabilities of shape [N, C] where N is the batch size and C is the number of labels.
            target: Ground truth labels of shape [N, C] where N is the batch size and C is the number of labels.
        """
        # Move thresholds to the same device as predictions
        thresholds = self.thresholds.to(preds.device)
        
        # Reshape predictions and targets for broadcasting
        preds = preds.unsqueeze(1)  # [N, 1, C]
        target = target.unsqueeze(1)  # [N, 1, C]
        thresholds = thresholds.view(1, -1, 1)  # [1, T, 1]
        
        # Convert probabilities to binary predictions for each threshold
        preds = (preds > thresholds).float()  # [N, T, C]
        
        # Compute statistics for each threshold
        true_positives = (preds * target).sum(dim=(0, 2))  # [T]
        false_positives = (preds * (1 - target)).sum(dim=(0, 2))  # [T]
        false_negatives = ((1 - preds) * target).sum(dim=(0, 2))  # [T]

        # Update state variables
        self.true_positives.add_(true_positives)
        self.false_positives.add_(false_positives)
        self.false_negatives.add_(false_negatives)
    
    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
        """
        Compute precision and recall for each threshold.
        
        Returns:
            A tuple containing:
            - thresholds: The threshold values used
            - precision: Precision values for each threshold
            - recall: Recall values for each threshold
        """
        # Compute precision and recall for each threshold
        precision = self.true_positives / (self.true_positives + self.false_positives + self.epsilon_tensor)
        recall = self.true_positives / (self.true_positives + self.false_negatives + self.epsilon_tensor)
        
        return self.thresholds, precision, recall
    
    def get_optimal_threshold(self, beta: float = 1.0) -> tuple[float, float]:
        """
        Find the threshold that maximizes the F-beta score.
        
        Args:
            beta: The beta parameter for the F-beta score. beta=1 gives equal weight to precision and recall.
                 beta<1 emphasizes precision, beta>1 emphasizes recall.
        
        Returns:
            The optimal threshold value.
        """
        _, precision, recall = self.compute()
        
        # Compute F-beta score for each threshold
        f_betas = compute_f_betas(precision, recall, device=self.device, beta=beta, epsilon=self.epsilon)
        
        # Find threshold with maximum F-beta score
        optimal_idx = f_betas.argmax()
        f_max = f_betas[optimal_idx]
        return self.thresholds[optimal_idx].item(), f_max.item()
    

class ExcessGraphComponents(Metric):
    """
    Computes the excess number of disconnected graph components in the subgraph induced by the predicted GO terms,
    normalized by the number of predicted GO terms.
    A proper set of GO terms should have at most three disconnected components, corresponding to the three GO sub-ontologies.
    """
    def __init__(self, go_graph: GOGraph, **kwargs):
        """
        Initialize the metric.
        
        Args:
            num_thresholds: Number of thresholds to evaluate between 0 and 1.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        
        self.go_graph = go_graph
        
        # Add state variable
        self.add_state('excess_components_per_term', default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0, dtype=torch.long)) # Our own accumulator for number of observations/samples

    def update(self, pred_go_terms: Container[Container[str]]):
        """
        Update the metric state with new predicted GO terms.
        
        Args:
            pred_go_terms: A container of predicted GO terms.
        """
        self.excess_components_per_term.add_(torch.tensor([self.go_graph.compute_excess_components_per_term(this_terms) for this_terms in pred_go_terms]).sum())
        self.num_samples.add_(torch.tensor(len(pred_go_terms), dtype=torch.long, device=self.device))
    
    def compute(self) -> torch.Tensor:
        """
        Compute the mean excess components per term.
        
        Returns:
            The mean number of excess components per term.
        """
        return self.excess_components_per_term / self.num_samples