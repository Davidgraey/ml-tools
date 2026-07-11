"""
Helpers for training, like the sigmoid decay / accumulate curve
"""
import numpy as np
from itertools import product
from numpy.typing import NDArray
import torch
import torch.nn.functional as torch_F


class DecayHelper:
    def __init__(self, learning_rate: tuple[float, float], total_epochs: int, accumulation: tuple[float, float]):
        """
        Decay helper to handle a sigmoid decay, or a sigmoid accumulation.
        Parameters
        ----------
        learning_rate : (learning rate stop, learning rate stop)
        total_epochs : total training epochs between start and stop
        """
        self.current_epoch: int = 0
        self.total_epochs: int = total_epochs
        
        self.learning_rate: tuple[float, float] = learning_rate
        self.accumulation_rate: tuple[float, float] = accumulation
        
        self.current_accumulation_rate = accumulation[0]
        self.current_learning_rate = learning_rate[0]

        # From -6 to 6 to cover most of the sigmoid curve
        self._sigmoid_steps = self.sigmoid(np.linspace(start=-10, stop=10, num=total_epochs))

    def step(self, epoch: int) -> None:
        """
        step the decay / accumulator to the given epoch
        Parameters
        ----------
        epoch :  int - the current epoch
        """
        if epoch > self.current_epoch:
            self.current_epoch = epoch
        self.current_learning_rate = self.sigmoid_transition(*self.learning_rate, epoch)
        self.current_accumulation_rate = self.sigmoid_transition(*self.accumulation_rate, epoch)

    def get_curves(self) -> list:
        return [self.current_learning_rate]

    @staticmethod
    def sigmoid(x: NDArray):
        """Standard sigmoid function."""
        return 1 / (1 + np.exp(-x * 0.5))

    def sigmoid_transition(self, starting_value: float, ending_value: float, current_step: int):
        """
        Grow or decay a value y from start to end over N steps using a sigmoid curve.
        Parameters:
        start (float): The starting value.
        end (float): The final value.
        N (int): The number of steps for the transition.
        Returns:
        list: A list of values representing the transition over N steps.
        """
        y_values = starting_value + ((ending_value - starting_value) * self._sigmoid_steps)
        return y_values[current_step]


class GridSearch:
    """
    A container class to hold some running values and do grid search.
    """
    def __init__(self,
                 learning_rates: tuple,
                 batch_sizes: tuple,
                 weight_decay: tuple,
                 positive_sample_w: tuple):
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.weight_decay = weight_decay
        self.positive_sample_w = positive_sample_w

        self.validation_scores = {}
        self.high_score = 0

        self._product = product(learning_rates, batch_sizes, weight_decay, positive_sample_w)

        self.active_step = None

    def _step(self):
        return next(self._product)

    def __next__(self):
        self.active_step = next(self._product)
        return self.active_step

    def __iter__(self):
        return self

    def log_values(self, precision: float, recall: float, accuracy: float, f1: float):
        self.validation_scores[self.active_step] = (precision, recall, accuracy, float)

    def is_best(self, f1: float):
        if f1 > self.high_score:
            self.high_score = f1
            return True
        return False

def cross_entropy_one_hot(logits: torch.Tensor, target: torch.Tensor, weights) -> torch.Tensor:
    _, labels = target.max(dim=1)
    return torch_F.cross_entropy(logits, labels, weight=weights)


def bin_cross_entropy_one_hot(logits: torch.Tensor, target: torch.Tensor, weights) -> torch.Tensor:
    return torch_F.binary_cross_entropy_with_logits(logits, target, weight=weights)

def class_balance_loss(
    labels: torch.Tensor,
    logits: torch.Tensor,
    samples_per_class: list[int],
    beta: float = 0.85,
    multilabel: bool = False,
) -> torch.Tensor:
    """
    Compute the Class Balanced Loss between logits and the ground truth labels.
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

    CBL: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    Loss here is either Binary Cross Entropy or crossentropy;

    Args:


    Returns:
      cb_loss: A float tensor representing class balanced loss
    Parameters
    ----------
    labels: A int tensor of size [batch, num_classes]. in one-hot encoding
    logits: Logits - output of our network. Tensor of size [batch, num_classes].
    samples_per_cls: list of size [num_classes].
    beta: float. Hyperparameter for Class balanced loss - adjusts the strength of weighting (value of 0.8 is a good
        starting place) - beta = 0.0 is no reweighting, and beta = 1.0 is full weighting by inverse class freq
    multilabel: bool - if the prediction problem is multilabel

    Returns
    -------
    dict of loss, prediction
    """
    batch_size, num_classes = logits.shape
    samples_per_class = torch.Tensor(samples_per_class) / sum(samples_per_class)

    effective_num = 1.0 - torch.pow(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * (num_classes + (1 if num_classes == 1 else 0))

    # UPDATE NOVEMBER -- ok - this is where we only use the labels which are not missing; if we didn't do this we
    # would be overwhelming zeros or whatever the NaNs are imputed as. This way we can skip those invalids,
    # thanks to the way we set up the one-hot encoding.
    valid_idxs = torch.any(labels, axis=1)
    valid_logits = logits[valid_idxs]
    valid_labels = labels[valid_idxs]

    # If there are no valid labels, return zero loss
    if valid_labels.size(0) == 0:
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    if num_classes == 2:
        # weights for [1] for the positive class
        loss = bin_cross_entropy_one_hot(logits=valid_logits, target=valid_labels, weights=weights)
        loss = loss.mean()

    elif num_classes > 2 and multilabel is True:
        loss = torch_F.binary_cross_entropy_with_logits(
            input=valid_logits, target=valid_labels, reduction="none"
        )
        loss = loss * weights
        loss = loss.mean()

    elif num_classes > 2:
        loss = cross_entropy_one_hot(logits=valid_logits, target=valid_labels, weights=weights)

    return loss


def contrastive_loss(
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        margin: float = 0.2
) -> torch.Tensor:
    """
    Constrastive loss - we should sample the same number of positive and negative samples.
    Parameters
    ----------
    anchor_emb : the anchor sample (should be (n, xdim))
    positive_emb : the positive examples (num_positives, xdim)
    negative_emb : the negative examples (num_negatives, xdim)
    margin : a nonzero value - this is the "minimum distance" for positive -> negative distances. "Larger margins penalize cases where the negative examples are not distant enough from the anchors, relative to the positives."

    Returns
    -------
    Loss calculated object (criterion)
    """
    triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch_F.pairwise_distance, margin=margin, swap=True, reduction="mean"
    )

    return triplet_loss(anchor_emb, positive_emb, negative_emb)
