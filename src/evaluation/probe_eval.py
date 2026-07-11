"""Linear probe validation for trait directions.

Trains logistic regression classifiers on activations to verify that
extracted trait directions capture real signal.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train_and_evaluate_probe(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """Train a logistic regression probe and return cross-validated accuracy.

    Args:
        positive_activations: (n_pos, hidden_dim) tensor.
        negative_activations: (n_neg, hidden_dim) tensor.
        n_folds: Number of cross-validation folds.

    Returns:
        Mean cross-validated accuracy.
    """
    X_pos = positive_activations.float().numpy()
    X_neg = negative_activations.float().numpy()

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return scores.mean()


def train_probe(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
) -> LogisticRegression:
    """Train a logistic regression probe on all data and return the fitted model."""
    X_pos = positive_activations.float().numpy()
    X_neg = negative_activations.float().numpy()

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X, y)
    return clf


def evaluate_probe_per_layer(
    layer_activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_folds: int = 5,
) -> dict[int, float]:
    """Evaluate probes across multiple layers.

    Args:
        layer_activations: {layer_idx: (positive_acts, negative_acts)}.

    Returns:
        {layer_idx: accuracy}.
    """
    results = {}
    for layer_idx, (pos, neg) in layer_activations.items():
        results[layer_idx] = train_and_evaluate_probe(pos, neg, n_folds)
    return results
