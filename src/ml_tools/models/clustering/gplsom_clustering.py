import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ml_tools.models.clustering.plsom_clustering import PLSOM
from ml_tools.visuals.cluster_visuals import plot_clusters


class GPLSOM(PLSOM):
    """
    Growing and Shrinking Parameterless SOM (GPLSOM)

    Extends the PLSOM with adaptive topology:
    - **Growth**: triggered when a neuron has high quantization error (mean distance to
      the samples it represents).  Boundary neurons spawn a new neighbor outward;
      interior neurons spawn a new node between themselves and their highest-error
      neighbor, with interpolated weights.
    - **Shrinking / Pruning**: triggered when a neuron is rarely selected as BMU *and*
      its weight vector is close to all of its grid neighbors (i.e. it is redundant).

    The grid is represented as a flexible adjacency graph so that topology changes
    do not require the map to stay rectangular.

    Core PLSOM equations (unchanged):
        ε(t)   = ||x(t) - w_c(t)||_2 / r(t)
        r(0)   = ||x(0) - w_c(0)||_2
        r(t)   = max( ||x(t)-w_c(t)||_2, r(t-1) )
        Θ(ε)   = neighborhood scale from ε
        h_ci   = exp( - d(i,c)^2 / Θ(ε)^2 )
        Δw_i   = ε · h_ci · (x - w_i)

    References
    ----------
    - PLSOM: https://arxiv.org/pdf/0705.0199
    - Growing SOM / Neural Gas: Fritzke (1995)
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        theta_min=None,
        theta_max=None,
        lock_seed: int = 42,
        distance: str = "euclidean",
        verbose: bool = False,
        # ---- growth / shrink knobs ----
        growth_threshold: float = 0.85,
        shrink_threshold: float = 0.02,
        grow_every: int = 1,
        min_neurons: int = 4,
        max_neurons: int = 400,
    ):
        """
        Parameters
        ----------
        width, height : initial rectangular grid size
        input_dim : feature dimensionality
        growth_threshold : quantile of per-neuron distortion above which growth is
            triggered (0-1, higher = more conservative).
        shrink_threshold : fraction of mean hits below which a neuron is a candidate
            for pruning (0-1, lower = more aggressive).
        grow_every : apply grow/shrink logic every N epochs.
        min_neurons : never shrink below this count.
        max_neurons : never grow beyond this count.
        """
        super().__init__(
            width=width,
            height=height,
            input_dim=input_dim,
            theta_min=theta_min,
            theta_max=theta_max,
            lock_seed=lock_seed,
            distance=distance,
            verbose=verbose,
        )

        # Growth / shrink configuration
        self.growth_threshold = growth_threshold
        self.shrink_threshold = shrink_threshold
        self.grow_every = grow_every
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        # Per-neuron accumulated distortion (quantization error)
        self.distortion_map = np.zeros(self.n_neurons)

        # Adjacency graph: node_id -> set of neighbor node_ids
        self.adjacency: dict[int, set[int]] = self._build_adjacency_from_grid()

        # Track topology change history for diagnostics
        self.topology_history: list[dict] = []

    # ------------------------------------------------------------------
    #  PLSOM hook overrides (called from super().fit)
    # ------------------------------------------------------------------
    def _on_sample_update(self, bmu_i: int | NDArray, sample_distances: NDArray) -> None:
        """Accumulate per-neuron distortion for the BMU."""
        self.distortion_map[bmu_i] += sample_distances[bmu_i]

    def _on_epoch_end(self, step: int, num_iterations: int) -> None:
        """Apply topology adaptation (grow/shrink) at configured intervals."""
        if (step + 1) % self.grow_every == 0 and step < num_iterations - 1:
            self.grow()
            self.shrink()

    # ------------------------------------------------------------------
    #  Adjacency helpers
    # ------------------------------------------------------------------
    def _build_adjacency_from_grid(self) -> dict[int, set[int]]:
        """Build an adjacency graph from the current rectangular grid."""
        rows, cols = self.network_shape
        adj: dict[int, set[int]] = {i: set() for i in range(self.n_neurons)}
        for idx in range(self.n_neurons):
            r, c = self._idx_to_grid(idx)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_idx = self._grid_to_idx((nr, nc))
                    adj[idx].add(neighbor_idx)
        return adj

    def _rebuild_graph_distances(self) -> NDArray:
        """
        Compute shortest-path (BFS) distances between all neurons using the
        adjacency graph.  This replaces the rectangular manhattan-distance grid
        and works for arbitrary topologies.
        """
        n = self.n_neurons
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)

        for source in range(n):
            visited = {source}
            frontier = [source]
            d = 0
            while frontier:
                d += 1
                next_frontier = []
                for node in frontier:
                    for neighbor in self.adjacency.get(node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[source, neighbor] = d
                            next_frontier.append(neighbor)
                frontier = next_frontier
        return dist

    def _is_boundary_node(self, idx: int) -> bool:
        """A node is on the boundary if it has fewer than 4 neighbors."""
        return len(self.adjacency.get(idx, set())) < 4

    def _get_neighbors(self, idx: int) -> list[int]:
        """Return sorted list of neighbor indices for *idx*."""
        return sorted(self.adjacency.get(idx, set()))

    # ------------------------------------------------------------------
    #  Topology mutations
    # ------------------------------------------------------------------
    def _add_neuron(self, new_weights: NDArray, connect_to: list[int]) -> int:
        """
        Insert a single neuron into the map.

        Parameters
        ----------
        new_weights : weight vector for the new neuron (shape: input_dim,)
        connect_to  : list of existing neuron indices to connect to

        Returns
        -------
        The index of the newly created neuron.
        """
        new_idx = self.n_neurons

        # Expand weight matrix
        self.weights = np.vstack([self.weights, new_weights.reshape(1, -1)])

        # Expand tracking arrays
        self.hit_map = np.append(self.hit_map, 0.0)
        self.distortion_map = np.append(self.distortion_map, 0.0)

        # Update counts
        self.n_neurons += 1

        # Update adjacency
        self.adjacency[new_idx] = set(connect_to)
        for neighbor in connect_to:
            self.adjacency[neighbor].add(new_idx)

        return new_idx

    def _remove_neuron(self, idx: int) -> None:
        """
        Remove a single neuron from the map.  Neighbors of the removed node
        are *not* reconnected to each other (the gap is simply closed).
        """
        if self.n_neurons <= self.min_neurons:
            return

        # Disconnect from neighbors
        for neighbor in list(self.adjacency.get(idx, set())):
            self.adjacency[neighbor].discard(idx)
        del self.adjacency[idx]

        # Delete from weight matrix & tracking arrays
        self.weights = np.delete(self.weights, idx, axis=0)
        self.hit_map = np.delete(self.hit_map, idx)
        self.distortion_map = np.delete(self.distortion_map, idx)
        self.n_neurons -= 1

        # Re-index everything above `idx` (shift down by 1)
        new_adj: dict[int, set[int]] = {}
        for old_key, old_neighbors in self.adjacency.items():
            new_key = old_key if old_key < idx else old_key - 1
            new_neighbors = set()
            for n in old_neighbors:
                new_neighbors.add(n if n < idx else n - 1)
            new_adj[new_key] = new_neighbors
        self.adjacency = new_adj

    def _rebuild_after_topology_change(self) -> None:
        """rebuild grid distances and update network_shape after a topology change."""
        self.grid_distances = self._rebuild_graph_distances()

        # network_shape is kept for compatibility but is now approximate
        side = int(np.ceil(np.sqrt(self.n_neurons)))
        self.network_shape = [side, int(np.ceil(self.n_neurons / side))]

    #  Growth logic
    def grow(self) -> int:
        """
        evaluate every neuron's distortion and grow where needed.

        Growth strategy:
        - mean distortion per neuron (accumulated error / hits).
        - neurons above the ``growth_threshold`` quantile are candidates.
        - **Boundary** candidate: spawn a new neuron *outward*, with weights
          extrapolated from the candidate and the mean of its neighbors.
        - **Interior** candidate: spawn a new neuron *between* the candidate and
          its highest-distortion neighbor, with interpolated weights.

        Returns
        -------
        Number of neurons added.
        """
        if self.n_neurons >= self.max_neurons:
            return 0

        # Snapshot the original neuron count — mean_distortion is only valid for
        # indices [0, n_original).  Neurons added during this loop must not be
        # looked up in this array.
        n_original = self.n_neurons

        # Mean distortion per neuron (avoid /0)
        active_mask = self.hit_map[:n_original] > 0
        mean_distortion = np.zeros(n_original)
        mean_distortion[active_mask] = (
            self.distortion_map[:n_original][active_mask]
            / self.hit_map[:n_original][active_mask]
        )

        if mean_distortion.max() == 0:
            return 0

        threshold = np.quantile(mean_distortion[active_mask], self.growth_threshold)
        candidates = np.where(mean_distortion >= threshold)[0]

        added = 0
        for cand in candidates:
            if self.n_neurons >= self.max_neurons:
                break

            # Only consider original-topology neighbors (filter out any newly added)
            neighbors = [n for n in self._get_neighbors(cand) if n < n_original]
            if len(neighbors) == 0:
                continue

            neighbor_weights = self.weights[neighbors]
            neighbor_mean = np.mean(neighbor_weights, axis=0)

            if self._is_boundary_node(cand):
                # Extrapolate outward: new = candidate + (candidate - neighbor_mean)
                new_w = self.weights[cand] + 0.5 * (self.weights[cand] - neighbor_mean)
                self._add_neuron(new_w, connect_to=[cand])
            else:
                # Interior: insert between candidate and its worst neighbor
                neighbor_distortions = mean_distortion[neighbors]
                worst_neighbor = neighbors[int(np.argmax(neighbor_distortions))]
                new_w = 0.5 * (self.weights[cand] + self.weights[worst_neighbor])
                # Connect to both the candidate and the worst neighbor
                self._add_neuron(new_w, connect_to=[cand, worst_neighbor])
                # Break direct edge between cand <-> worst_neighbor
                # to keep the graph planar (the new node sits "between" them)
                self.adjacency[cand].discard(worst_neighbor)
                self.adjacency[worst_neighbor].discard(cand)

            added += 1

        if added > 0:
            self._rebuild_after_topology_change()
            self.topology_history.append(
                {"epoch": len(self.q_error_trace), "action": "grow", "count": added,
                 "n_neurons": self.n_neurons}
            )
            if self.verbose:
                print(f"[GPLSOM] Grew {added} neuron(s) → {self.n_neurons} total")

        return added

    # ------------------------------------------------------------------
    #  Shrink logic
    # ------------------------------------------------------------------
    def shrink(self) -> int:
        """
        Remove redundant neurons.

        A neuron is pruned when:
        1. Its hit count is below ``shrink_threshold × mean(hit_map)``  (rarely used).
        2. Its weight vector is close to all neighbors (redundant representation).

        Returns
        -------
        Number of neurons removed.
        """
        if self.n_neurons <= self.min_neurons:
            return 0

        mean_hits = np.mean(self.hit_map) if np.sum(self.hit_map) > 0 else 1.0
        hit_threshold = self.shrink_threshold * mean_hits

        # Compute per-neuron redundancy: mean weight-space distance to neighbors
        redundancy = np.full(self.n_neurons, np.inf)
        for idx in range(self.n_neurons):
            neighbors = self._get_neighbors(idx)
            if len(neighbors) == 0:
                continue
            neighbor_weights = self.weights[neighbors]
            dists = self.distance_function(
                self.weights[idx].reshape(1, -1), neighbor_weights
            )
            redundancy[idx] = np.mean(dists)

        # median neighbor distance as a scale reference
        finite_mask = np.isfinite(redundancy)
        if not np.any(finite_mask):
            return 0
        redundancy_threshold = np.median(redundancy[finite_mask]) * 0.33

        # Candidates: low hits AND close to neighbors
        candidates = np.where(
            (self.hit_map <= hit_threshold) & (redundancy <= redundancy_threshold)
        )[0]

        # Sort by hits ascending so we remove the least-used first
        candidates = candidates[np.argsort(self.hit_map[candidates])]

        removed = 0
        # Remove one at a time (indices shift after each removal)
        for cand in candidates:
            if self.n_neurons <= self.min_neurons:
                break
            # Re-check after prior removals may have shifted things
            if cand >= self.n_neurons:
                continue

            # Before removing, reconnect its neighbors to each other so the graph
            # doesn't fragment
            neighbors = self._get_neighbors(cand)
            for i, ni in enumerate(neighbors):
                for nj in neighbors[i + 1:]:
                    self.adjacency[ni].add(nj)
                    self.adjacency[nj].add(ni)

            self._remove_neuron(cand)
            removed += 1
            # After removal, all indices >= cand shifted down; adjust remaining candidates
            candidates = np.where(candidates > cand, candidates - 1, candidates)

        if removed > 0:
            self._rebuild_after_topology_change()
            self.topology_history.append(
                {"epoch": len(self.q_error_trace), "action": "shrink", "count": removed,
                 "n_neurons": self.n_neurons}
            )
            if self.verbose:
                print(f"[GPLSOM] Pruned {removed} neuron(s) → {self.n_neurons} total")

        return removed

    # ------------------------------------------------------------------
    #  Visualization (overrides to support non-rectangular topology)
    # ------------------------------------------------------------------
    def plot_grid(self, samples=0, highlight_idx=None):
        """Plot neuron positions (first 2 weight dims) with adjacency edges."""
        if self.weights.shape[1] < 2:
            return

        plt.figure(figsize=(15, 10))
        plt.title(f"GPLSOM Grid ({self.n_neurons} neurons)")

        if not isinstance(samples, int):
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=5)

        # Draw edges from adjacency
        drawn = set()
        for idx, neighbors in self.adjacency.items():
            for n in neighbors:
                edge = (min(idx, n), max(idx, n))
                if edge not in drawn:
                    drawn.add(edge)
                    plt.plot(
                        [self.weights[idx, 0], self.weights[n, 0]],
                        [self.weights[idx, 1], self.weights[n, 1]],
                        "b-", alpha=0.4, linewidth=0.8,
                    )

        plt.scatter(self.weights[:, 0], self.weights[:, 1], c="blue", s=30, zorder=5)

        if highlight_idx is not None and highlight_idx < self.n_neurons:
            plt.scatter(
                self.weights[highlight_idx, 0],
                self.weights[highlight_idx, 1],
                s=200, c="red", zorder=6,
            )
        plt.show()

    def plot_heatmap(self):
        """Draw the hit map as a bar chart (topology may not be rectangular)."""
        plt.figure(figsize=(12, 4))
        plt.title("GPLSOM Hit Map")
        plt.bar(range(self.n_neurons), self.hit_map)
        plt.xlabel("Neuron index")
        plt.ylabel("Hits")
        plt.show()

    def plot_topology_history(self):
        """Visualise the growth/shrink events over training."""
        if not self.topology_history:
            print("No topology changes recorded.")
            return
        epochs = [e["epoch"] for e in self.topology_history]
        sizes = [e["n_neurons"] for e in self.topology_history]
        colors = ["green" if e["action"] == "grow" else "red" for e in self.topology_history]
        plt.figure(figsize=(10, 4))
        plt.title("GPLSOM Topology Changes")
        plt.scatter(epochs, sizes, c=colors, s=40)
        plt.plot(epochs, sizes, "k--", alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Neuron count")
        plt.show()

    # ------------------------------------------------------------------
    #  Forward / predict / fit_predict — all inherited from PLSOM
    # ------------------------------------------------------------------

    @property
    def params(self):
        return self.weights


if __name__ == "__main__":
    # Example usage
    from ml_tools.generators import RandomDatasetGenerator
    from ml_tools.models.clustering.cluster_metrics import *

    feature_dim = 2
    gen = RandomDatasetGenerator(random_seed=123)
    x_clust, y_clust, meta = gen.generate(
        task="clustering",
        num_samples=1500,
        num_features=feature_dim,
        num_clusters=6,
        noise_scale=0.4,
    )
    plot_clusters(x_clust, y_clust, meta["centroids"])

    max_clusters = 10

    for dim in [6]:
        num_steps = dim * 8
        st = time.time()

        som = GPLSOM(
            width=dim,
            height=dim,
            input_dim=feature_dim,
            theta_min=0.01,
            theta_max=dim - 0.01,
            lock_seed=42,
            distance="euclidean",
            verbose=True,
            growth_threshold=0.80,
            shrink_threshold=0.05,
            grow_every=2,
            min_neurons=4,
            max_neurons=200,
        )

        predictions = som.fit_predict(
            x_data=x_clust,
            grid_dim=dim,
            num_iterations=num_steps,
            max_clusters=max_clusters,
            verbose=True,
        )

        print("GPLSOM predict took: ", time.time() - st)
        print(f"Final neuron count: {som.n_neurons}")
        print(f"Topology events: {len(som.topology_history)}")

        som.plot_topology_history()

        print("Predictions:", predictions[:20])
        print("Truth:", y_clust[:20])

        print(f"silhouette score (1 is best): {silhouette_score(x_clust, predictions)}")
        print(f"CH index (high): {calinski_harabasz_index(x_clust, predictions)}")
        print(f"DB index score (low): {davies_bouldin_index(x_clust, predictions)}")

        print(f"homogeneity: {homogeneity(y_clust, predictions)}")
        print(f"Mutual Information: {mutual_information_score(y_clust, predictions)}")
