from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


def to_onehot(class_array: NDArray, num_classes=None) -> NDArray:
    """Convert integer class array to one-hot"""
    class_array = np.asarray(class_array)
    if num_classes is None:
        num_classes = int(class_array.max() + 1)
    return np.eye(num_classes)[class_array]


def to_int_classes(onehot_array: NDArray, is_multilabel: bool = False) -> NDArray:
    """Convert one-hot style array into an integer rep of the class number apparently I cannot spell integer"""
    if is_multilabel:
        int_array = np.round(onehot_array, axis=-1).astype(int)
    else:
        int_array = np.argmax(onehot_array, axis=-1).astype(int)
    return int_array


def to_multilabel(row_indices: NDArray | list, num_classes: int) -> NDArray:
    """Convert list of sets to multilabel (multi-label one-hot) array"""
    result = np.zeros((len(row_indices), num_classes), dtype=int)
    for i, indices in enumerate(row_indices):
        result[i, indices] = 1
    return result


@dataclass
class GenConfig:
    num_samples: int = 1000
    num_features: int = 10
    noise_scale: float = 0.5
    num_classes: int = 3  # for multinomial / multilabel
    num_clusters: int = 3  # for clustering
    ensure_label: bool = True  # for multilabel: ensure at least one sample for a class
    onehot: bool = False  # for classification
    verbose: bool = True # for stupid extra prints everywhere


class RandomDatasetGenerator:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    # --------------- Core Shared Utilities ---------------

    def _features(
        self, n: int, d: int, mode: Literal["uniform", "normal"] = "normal"
    ) -> NDArray:
        if mode == "uniform":
            return self.rng.uniform(0.0, 1.0, size=(n, d))
        return self.rng.normal(0.0, 1.0, size=(n, d))

    def _shuffle(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        idx = self.rng.permutation(len(X))
        return X[idx], y[idx]

    def _linear_scores(
        self, X: NDArray, W: NDArray, b: Optional[NDArray] = None
    ) -> NDArray:
        S = X @ W
        if b is not None:
            S += b
        return S

    def _softmax(self, Z: NDArray) -> NDArray:
        Z = Z - Z.max(axis=1, keepdims=True)
        EZ = np.exp(Z)
        return EZ / EZ.sum(axis=1, keepdims=True)

    def _sigmoid(self, Z: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-Z))

    def _split_counts(self, total: int, groups: int) -> list[int]:
        base = total // groups
        counts = [base] * groups
        for i in range(total % groups):
            counts[i] += 1
        return counts

    # --------------- Public Dispatcher ---------------

    def generate(
        self,
        task: Literal["regression", "binary", "multiclass", "multilabel", "clustering"],
        **kwargs,
    ):
        config = GenConfig(**kwargs)
        task = task.lower()
        if task == "regression":
            return self._regression(config)
        if task == "binary":
            return self._binary(config)
        if task == "multiclass":
            return self._multiclass(config)
        if task == "multilabel":
            return self._multilabel(config)
        if task == "clustering":
            return self._clustering(config)
        raise ValueError(f"Unknown task: {task}")

    # --------------- Task Implementations ---------------

    def _regression(self, config: GenConfig):
        X = self._features(config.num_samples, config.num_features, mode="uniform")
        w = self.rng.normal(0, 1, size=config.num_features)
        b = self.rng.normal()
        noise = self.rng.normal(0, config.noise_scale, size=config.num_samples)
        y = X @ w + b + noise
        X, y = self._shuffle(X, y)
        meta = dict(weights=w, bias=b, y_min=y.min(), y_max=y.max(), y_mean=y.mean())
        if config.verbose:
            print(f"Regression: X{X.shape}, y range=({y.min():.2f},{y.max():.2f})")
        return X, y, meta

    def _binary(self, config: GenConfig):
        X = self._features(config.num_samples, config.num_features)
        w = self.rng.normal(0, 1, size=config.num_features)
        b = self.rng.normal()
        logits = self._linear_scores(X, w, b) + self.rng.normal(
            0, config.noise_scale, config.num_samples
        )
        probs = self._sigmoid(logits)
        y = (probs >= 0.5).astype(int)
        if config.onehot:
            y_out = to_onehot(y, 2)
        else:
            y_out = y
        X, y_out = self._shuffle(X, y_out)
        meta = dict(
            weights=w,
            bias=b,
            mean_prob=probs.mean(),
            class_counts=np.bincount(y, minlength=2),
        )
        if config.verbose:
            print(f"Binary: X{X.shape}, counts={meta['class_counts']}")
        return X, y_out, meta

    def _multiclass(self, config: GenConfig, enforce_balance: bool = True):
        k = config.num_classes
        n = config.num_samples
        counts = self._split_counts(n, k)
        Y = np.concatenate([np.full(c, cls, dtype=int) for cls, c in enumerate(counts)])
        self.rng.shuffle(Y)

        # 2
        X = self._features(n, config.num_features)
        W = self.rng.normal(0, 1, size=(config.num_features, k))
        b = self.rng.normal(0, 1, size=k)
        scores = self._linear_scores(X, W, b)

        # add margin between
        margin = 0.5
        scores[np.arange(n), Y] += margin

        probs = self._softmax(scores)
        Y = probs.argmax(-1)
        X, y_out = self._shuffle(X, Y)

        final_counts = np.bincount(Y, minlength=k)

        meta = dict(
            weights=W,
            bias=b,
            class_counts=final_counts,
            probs_mean=probs.mean(axis=0),
            balanced=enforce_balance,
            margin=margin,
        )
        if config.verbose:
            print(f"Multiclass: X{X.shape}, counts={final_counts}")
        return X, y_out, meta

    def _multilabel(self, config: GenConfig):
        k = config.num_classes
        X = self._features(config.num_samples, config.num_features)
        W = self.rng.normal(0, 1, size=(config.num_features, k))
        b = self.rng.normal(0, 1, size=k)
        logits = self._linear_scores(X, W, b) + self.rng.normal(
            0, config.noise_scale, size=(config.num_samples, k)
        )
        probs = self._sigmoid(logits)

        Y = (probs >= 0.5).astype(int)
        if config.ensure_label:
            empty = Y.sum(axis=1) == 0
            if empty.any():
                # force the argmax label if empty
                forced = np.argmax(probs[empty], axis=1)
                Y[empty, forced] = 1
        X, Y = self._shuffle(X, Y)

        meta = dict(weights=W, bias=b, label_frequencies=Y.sum(axis=0))
        if config.verbose:
            print(
                f"Multilabel: X{X.shape}, per-label counts={meta['label_frequencies']}"
            )
        return X, Y, meta

    def _clustering(self, config: GenConfig):
        k = config.num_clusters
        counts = self._split_counts(config.num_samples, k)
        centroids = self.rng.uniform(-3, 3, size=(k, config.num_features))
        clusters = []
        labels = []
        for i, c in enumerate(counts):
            block = centroids[i] + self.rng.normal(
                0, config.noise_scale, size=(c, config.num_features)
            )
            clusters.append(block)
            labels.append(np.full(c, i, dtype=int))
        X = np.vstack(clusters)
        y = np.concatenate(labels)
        X, y = self._shuffle(X, y)
        meta = dict(centroids=centroids, cluster_sizes=counts)
        if config.verbose:
            print(f"Clustering: X{X.shape}, sizes={counts}")
        return X, y, meta


# Example usage:
if __name__ == "__main__":
    gen = RandomDatasetGenerator(random_seed=123)
    x_class, y_class, meta_class = gen.generate(
        "classification", num_samples=200, num_features=5, num_classes=2, onehot=True
    )
    x_multi, y_multi, meta_multi = gen.generate(
        "multilabel", num_samples=100, num_features=4, num_classes=4
    )
    x_regression, y_regression, meta_regression = gen.generate(
        "regression", num_samples=100
    )
    x_clust, y_clust, meta_clust = gen.generate(
        "clustering", num_samples=150, num_features=2, num_clusters=4
    )
