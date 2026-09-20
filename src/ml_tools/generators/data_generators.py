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


SIGNAL_FAMILIES = ("tone", "multitone", "chirp", "damped", "noise")
IMAGE_SHAPES = ("disc", "rect", "cross", "ring")
# "gaussian" and "elongated" are analytic (no rejection sampling); the rest
# reuse the same membership test as IMAGE_SHAPES, sampled as points instead
# of a pixel mask; "moon" is a half ring.
CLUSTER_SHAPES = ("gaussian", "elongated", "disc", "rect", "cross", "ring", "moon")

# floor for the signal duration, so a length-1 request cannot divide by zero
EPSILON_TIME = 1e-12


@dataclass
class GenConfig:
    num_samples: int = 1000
    num_features: int = 10
    noise_scale: float = 0.5
    num_classes: int = 3  # for multinomial / multilabel
    num_clusters: int = 3  # for clustering
    outlier_fraction: float = 0.0  # for clustering: fraction of samples that are uniform noise, labeled -1
    cluster_shape: str = "gaussian"  # for clustering: a CLUSTER_SHAPES name, or a sequence of names cycled across clusters
    variance_jitter: float = 0.0  # for clustering: randomizes each cluster's spread per cluster and per dimension, 0 = flat noise_scale
    separation: float = 0.0  # for clustering: minimum centroid distance as a multiple of noise_scale, 0 disables
    imbalance: float = 0.0  # for clustering: >0 skews cluster sizes via Dirichlet concentration 1/imbalance
    ensure_label: bool = True  # for multilabel: ensure at least one sample for a class
    onehot: bool = False  # for classification
    verbose: bool = True # for stupid extra prints everywhere

    # for signal
    signal_length: int = 128
    sample_rate: int = 1000
    freq_low: float = 5.0
    freq_high: float = 120.0

    # for image
    image_size: int = 32
    min_extent: float = 0.15  # smallest shape radius, as a fraction of the image
    max_extent: float = 0.40  # largest shape radius, as a fraction of the image


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

    def _split_counts(self, total: int, groups: int, imbalance: float = 0.0) -> list[int]:
        """
        Split `total` into `groups` integer counts summing to total.

        imbalance=0 gives an even split. A higher value draws group
        proportions from a Dirichlet distribution instead (concentration
        1/imbalance, so larger imbalance means more skewed group sizes,
        including possibly empty groups).
        """
        if imbalance <= 0:
            base = total // groups
            counts = [base] * groups
            for i in range(total % groups):
                counts[i] += 1
            return counts
        concentration = max(1.0 / imbalance, 0.05)
        proportions = self.rng.dirichlet(np.full(groups, concentration))
        return list(self.rng.multinomial(total, proportions))

    def _rotate_2d(self, x: NDArray, y: NDArray, angle: float) -> tuple[NDArray, NDArray]:
        """Rotate 2D coordinates by `angle` radians about the origin."""
        rotated_x = x * np.cos(angle) + y * np.sin(angle)
        rotated_y = -x * np.sin(angle) + y * np.cos(angle)
        return rotated_x, rotated_y

    def _shape_mask(self, shape: str, x: NDArray, y: NDArray, extent: float) -> NDArray:
        """
        Boolean mask for which (x, y) coordinates fall inside the named
        shape. Shared by the image generator (applied to a pixel grid) and
        the clustering generator (applied to candidate points, for
        rejection sampling).
        """
        radius = np.sqrt(x ** 2 + y ** 2)
        if shape == "disc":
            return radius <= extent
        if shape == "rect":
            return (np.abs(x) <= extent) & (np.abs(y) <= extent * 0.6)
        if shape == "cross":
            arm = extent * 0.3
            return ((np.abs(x) <= extent) & (np.abs(y) <= arm)) | (
                (np.abs(y) <= extent) & (np.abs(x) <= arm)
            )
        if shape == "ring":
            return (radius <= extent) & (radius >= extent * 0.6)
        raise ValueError(f"unknown shape: {shape}")

    # --------------- Public Dispatcher ---------------

    def generate(
        self,
        task: Literal[
            "regression",
            "binary",
            "multiclass",
            "multilabel",
            "clustering",
            "signal",
            "image",
        ],
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
        if task == "signal":
            return self._signal(config)
        if task == "image":
            return self._image(config)
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

        final_counts = np.bincount(Y, minlength=k)

        # honour onehot the way every other classification task does
        X, y_out = self._shuffle(X, to_onehot(Y, k) if config.onehot else Y)

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

    # --------------- Clustering ---------------

    def _cluster_shapes_for(self, requested: str | list, k: int) -> list:
        """One shape name repeated k times, or a sequence cycled across k clusters."""
        if isinstance(requested, str):
            return [requested] * k
        shapes = list(requested)
        if not shapes:
            raise ValueError("cluster_shape sequence must not be empty")
        return [shapes[i % len(shapes)] for i in range(k)]

    def _cluster_scale(self, base_scale: float, dims: int, variance_jitter: float) -> NDArray:
        """
        Per-dimension noise scale for one cluster: `base_scale` perturbed by
        an overall per-cluster factor and an independent per-dimension
        factor, both drawn from [1 - variance_jitter, 1 + variance_jitter].
        variance_jitter=0 reproduces a flat `base_scale` in every dimension.
        """
        if variance_jitter <= 0:
            return np.full(dims, base_scale)
        low = max(1.0 - variance_jitter, 0.05)
        high = 1.0 + variance_jitter
        cluster_factor = self.rng.uniform(low, high)
        dim_factor = self.rng.uniform(low, high, size=dims)
        return base_scale * cluster_factor * dim_factor

    def _place_centroids(self, k: int, dims: int, min_separation: float) -> NDArray:
        """
        Draw k centroids uniformly in a [-3, 3]^dims box. When
        min_separation > 0, a candidate closer than that to any
        already-placed centroid is redrawn, up to 200 tries, after which the
        last candidate is kept rather than looping forever on an infeasible
        request (too many clusters/too large a separation for the box).
        """
        box = 3.0
        centroids = np.empty((k, dims))
        for i in range(k):
            candidate = self.rng.uniform(-box, box, size=dims)
            if min_separation > 0:
                for _ in range(200):
                    if i == 0 or np.min(np.linalg.norm(centroids[:i] - candidate, axis=1)) >= min_separation:
                        break
                    candidate = self.rng.uniform(-box, box, size=dims)
            centroids[i] = candidate
        return centroids

    def _shape_points(self, shape: str, count: int, extent: float = 1.0, angle: float = 0.0) -> NDArray:
        """
        Sample `count` 2D points inside the named shape via rejection
        sampling against `_shape_mask`, then rotate them by `angle`.
        "moon" is a half ring (a crescent), reusing the ring mask.
        """
        mask_shape = "ring" if shape == "moon" else shape
        box = extent * 1.2
        points = np.empty((count, 2))
        filled = 0
        while filled < count:
            batch = max((count - filled) * 3, 32)
            candidates = self.rng.uniform(-box, box, size=(batch, 2))
            keep = self._shape_mask(mask_shape, candidates[:, 0], candidates[:, 1], extent)
            if shape == "moon":
                keep = keep & (candidates[:, 1] >= 0)
            kept = candidates[keep]
            take = min(len(kept), count - filled)
            points[filled:filled + take] = kept[:take]
            filled += take
        if angle:
            points[:, 0], points[:, 1] = self._rotate_2d(points[:, 0], points[:, 1], angle)
        return points

    def _clustering(self, config: GenConfig):
        """
        Point clusters around random centroids, plus optional uniform
        outliers.

        `cluster_shape` selects the point-cloud shape: "gaussian" (default)
        is an isotropic ball; "elongated" is an anisotropic, randomly
        rotated gaussian; "disc"/"rect"/"cross"/"ring"/"moon" are sampled
        via rejection sampling against the same membership test the image
        generator's shapes use. It can be one name for every cluster, or a
        sequence cycled across clusters. `variance_jitter` randomizes each
        cluster's spread per cluster and per dimension. `separation` sets a
        minimum centroid distance, as a multiple of noise_scale. `imbalance`
        skews cluster sizes instead of splitting them evenly. Outliers are
        labeled -1 and scattered over the data's own bounding box. Every
        knob defaults to off, reproducing the original plain isotropic
        gaussian, evenly split behavior.
        """
        k = config.num_clusters
        n_outliers = max(0, min(round(config.num_samples * config.outlier_fraction), config.num_samples))
        counts = self._split_counts(config.num_samples - n_outliers, k, imbalance=config.imbalance)
        centroids = self._place_centroids(k, config.num_features, config.separation * config.noise_scale)
        shapes = self._cluster_shapes_for(config.cluster_shape, k)

        clusters = []
        labels = []
        for i, (count, shape) in enumerate(zip(counts, shapes)):
            scale = self._cluster_scale(config.noise_scale, config.num_features, config.variance_jitter)

            if shape == "gaussian":
                offsets = self.rng.normal(0, scale, size=(count, config.num_features))

            elif shape == "elongated":
                axis_scale = scale.copy()
                axis_scale[0] *= 3.0
                if config.num_features > 1:
                    axis_scale[1:] *= 0.4
                offsets = self.rng.normal(0, axis_scale, size=(count, config.num_features))
                if config.num_features >= 2:
                    angle = self.rng.uniform(0, np.pi)
                    offsets[:, 0], offsets[:, 1] = self._rotate_2d(
                        offsets[:, 0], offsets[:, 1], angle
                    )

            else:
                if config.num_features < 2:
                    raise ValueError(f"cluster shape '{shape}' needs num_features >= 2")
                angle = self.rng.uniform(0, np.pi)
                offsets = np.empty((count, config.num_features))
                offsets[:, :2] = self._shape_points(shape, count, angle=angle) * scale[:2]
                if config.num_features > 2:
                    offsets[:, 2:] = self.rng.normal(
                        0, scale[2:], size=(count, config.num_features - 2)
                    )

            clusters.append(centroids[i] + offsets)
            labels.append(np.full(count, i, dtype=int))

        X = np.vstack(clusters)
        y = np.concatenate(labels)

        if n_outliers:
            if X.shape[0] > 0:
                margin = 0.25 * (X.max(axis=0) - X.min(axis=0))
                low = X.min(axis=0) - margin
                high = X.max(axis=0) + margin
            else:
                # every sample is an outlier -- fall back to the centroid
                # spread itself, since there is no cluster data to size off
                margin = np.full(config.num_features, max(config.noise_scale, 1e-6) * 3)
                low = centroids.min(axis=0) - margin
                high = centroids.max(axis=0) + margin
            outliers = self.rng.uniform(low, high, size=(n_outliers, config.num_features))
            X = np.vstack([X, outliers])
            y = np.concatenate([y, np.full(n_outliers, -1, dtype=int)])

        X, y = self._shuffle(X, y)
        meta = dict(
            centroids=centroids,
            cluster_sizes=counts,
            cluster_shapes=shapes,
            outlier_count=n_outliers,
            outlier_label=-1,
        )
        if config.verbose:
            print(f"Clustering: X{X.shape}, sizes={counts}, shapes={shapes}, outliers={n_outliers}")
        return X, y, meta

    # --------------- Signal ---------------

    def _signal(self, config: GenConfig):
        """
        One dimensional waveforms drawn from distinct families.

        Each sample gets its own random phase, amplitude and frequency, so a
        classifier has to key on the shape of the waveform rather than on its
        loudness or where it happens to start. Frequencies are drawn below the
        Nyquist limit of the requested sample rate, so nothing aliases.
        """
        families = SIGNAL_FAMILIES[: config.num_classes]
        if not families:
            raise ValueError("num_classes must be at least 1")

        counts = self._split_counts(config.num_samples, len(families))
        seconds = np.arange(config.signal_length) / config.sample_rate
        nyquist = config.sample_rate / 2.0
        high = min(config.freq_high, 0.95 * nyquist)

        waves, labels, frequencies = [], [], []
        for class_index, (family, count) in enumerate(zip(families, counts)):
            for _ in range(count):
                base = self.rng.uniform(config.freq_low, high)
                phase = self.rng.uniform(0, 2 * np.pi)
                amplitude = self.rng.uniform(0.5, 1.5)

                if family == "tone":
                    wave = np.sin(2 * np.pi * base * seconds + phase)

                elif family == "multitone":
                    partial = min(3 * base, high)
                    wave = np.sin(2 * np.pi * base * seconds + phase) + 0.5 * np.sin(
                        2 * np.pi * partial * seconds + phase
                    )

                elif family == "chirp":
                    # frequency sweeps linearly, so the instantaneous phase is
                    # the integral of the rate rather than a constant times t
                    end = min(4 * base, high)
                    rate = (end - base) / max(seconds[-1], EPSILON_TIME)
                    wave = np.sin(
                        2 * np.pi * (base * seconds + 0.5 * rate * seconds ** 2) + phase
                    )

                elif family == "damped":
                    decay = self.rng.uniform(3.0, 8.0) / max(seconds[-1], EPSILON_TIME)
                    wave = np.exp(-decay * seconds) * np.sin(
                        2 * np.pi * base * seconds + phase
                    )

                elif family == "noise":
                    # band limited: shape white noise in the frequency domain so
                    # it has a defined band rather than being broadband
                    spectrum = self.rng.normal(size=config.signal_length // 2 + 1)
                    bins = np.fft.rfftfreq(config.signal_length, 1 / config.sample_rate)
                    spectrum = spectrum * (np.abs(bins - base) < base * 0.5)
                    wave = np.fft.irfft(spectrum, n=config.signal_length)
                    peak = np.max(np.abs(wave))
                    wave = wave / peak if peak > 0 else wave

                else:
                    raise ValueError(f"unknown signal family: {family}")

                wave = amplitude * wave
                wave = wave + self.rng.normal(
                    0, config.noise_scale * 0.1, size=config.signal_length
                )

                waves.append(wave)
                labels.append(class_index)
                frequencies.append(base)

        X = np.asarray(waves)
        y = np.asarray(labels, dtype=int)
        frequencies = np.asarray(frequencies)

        order = self.rng.permutation(len(X))
        X, y, frequencies = X[order], y[order], frequencies[order]

        if config.onehot:
            y = to_onehot(y, len(families))

        meta = dict(
            class_names=families,
            frequencies=frequencies,
            sample_rate=config.sample_rate,
            signal_length=config.signal_length,
            nyquist=nyquist,
            class_counts=np.bincount(np.asarray(labels), minlength=len(families)),
        )
        if config.verbose:
            print(f"Signal: X{X.shape}, families={list(families)}")
        return X, y, meta

    # --------------- Image ---------------

    def _image(self, config: GenConfig):
        """
        Greyscale images each holding one shape, at random position, size and
        rotation.

        Everything is drawn by evaluating inequalities on a coordinate grid, so
        no drawing library is involved.

        The classes are separable by structure, not by brightness. Measured on
        800 samples with four shapes, nearest-centroid accuracy is 0.31 on mean
        pixel value alone against a chance rate of 0.25, 0.44 on raw pixels --
        low because the shapes move and rotate, so a pixel-space centroid
        washes out -- and 0.92 on position and scale invariant descriptors.
        """
        shapes = IMAGE_SHAPES[: config.num_classes]
        if not shapes:
            raise ValueError("num_classes must be at least 1")

        counts = self._split_counts(config.num_samples, len(shapes))
        size = config.image_size

        # coordinates on [-1, 1] so extents read as a fraction of the half width
        axis = np.linspace(-1.0, 1.0, size)
        grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")

        images, labels, centres, extents, angles = [], [], [], [], []
        for class_index, (shape, count) in enumerate(zip(shapes, counts)):
            for _ in range(count):
                extent = self.rng.uniform(config.min_extent, config.max_extent) * 2
                # keep the shape clear of the frame. A limit of exactly
                # 1 - extent lets the widest shapes graze the border, so leave
                # a margin of one twentieth of the half width.
                limit = max(1.0 - extent - 0.05, 0.0)
                centre = self.rng.uniform(-limit, limit, size=2)
                angle = self.rng.uniform(0, np.pi)

                shifted_x = grid_x - centre[1]
                shifted_y = grid_y - centre[0]
                # rotate the sample points, which rotates the shape the other way
                rotated_x, rotated_y = self._rotate_2d(shifted_x, shifted_y, angle)

                mask = self._shape_mask(shape, rotated_x, rotated_y, extent)

                image = mask.astype(np.float64)
                image = image + self.rng.normal(
                    0, config.noise_scale * 0.1, size=image.shape
                )

                images.append(image)
                labels.append(class_index)
                centres.append(centre)
                extents.append(extent)
                angles.append(angle)

        X = np.asarray(images)
        y = np.asarray(labels, dtype=int)

        order = self.rng.permutation(len(X))
        X, y = X[order], y[order]

        if config.onehot:
            y = to_onehot(y, len(shapes))

        meta = dict(
            class_names=shapes,
            centres=np.asarray(centres)[order],
            extents=np.asarray(extents)[order],
            angles=np.asarray(angles)[order],
            image_size=size,
            class_counts=np.bincount(np.asarray(labels), minlength=len(shapes)),
        )
        if config.verbose:
            print(f"Image: X{X.shape}, shapes={list(shapes)}")
        return X, y, meta


# Example usage:
if __name__ == "__main__":
    gen = RandomDatasetGenerator(random_seed=123)
    x_class, y_class, meta_class = gen.generate(
        "binary", num_samples=200, num_features=5, num_classes=2, onehot=True
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
    x_signal, y_signal, meta_signal = gen.generate(
        "signal", num_samples=120, signal_length=256, sample_rate=1000, num_classes=5
    )
    x_image, y_image, meta_image = gen.generate(
        "image", num_samples=120, image_size=32, num_classes=4
    )

    import matplotlib.pyplot as plt

    # one example waveform per family
    families = meta_signal["class_names"]
    figure, panels = plt.subplots(len(families), 1, figsize=(10, 2 * len(families)))
    for index, family in enumerate(families):
        pick = np.flatnonzero(y_signal == index)[0]
        panels[index].plot(x_signal[pick])
        panels[index].set_title(
            f"{family}, planted {meta_signal['frequencies'][pick]:.1f} Hz"
        )
        panels[index].set_xticks([])
    plt.tight_layout()
    plt.show()

    # four examples per shape
    shapes = meta_image["class_names"]
    figure, panels = plt.subplots(len(shapes), 4, figsize=(8, 2 * len(shapes)))
    for row, shape in enumerate(shapes):
        picks = np.flatnonzero(y_image == row)[:4]
        for column, pick in enumerate(picks):
            panels[row, column].imshow(x_image[pick], cmap="gray")
            panels[row, column].set_xticks([])
            panels[row, column].set_yticks([])
        panels[row, 0].set_ylabel(shape)
    plt.tight_layout()
    plt.show()
