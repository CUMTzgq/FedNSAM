from __future__ import annotations

import numpy as np


def build_dirichlet_partitions(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_size: int = 10,
    client_sizes: list[int] | None = None,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array.")

    if client_sizes is None:
        base = len(targets) // num_clients
        remainder = len(targets) % num_clients
        client_sizes = [base + (1 if client_id < remainder else 0) for client_id in range(num_clients)]

    if len(client_sizes) != num_clients:
        raise ValueError("client_sizes length must match num_clients.")
    if any(size < 0 for size in client_sizes):
        raise ValueError("client_sizes must be non-negative.")
    if sum(client_sizes) != len(targets):
        raise ValueError("client_sizes must sum to the dataset size.")
    if min(client_sizes) < min_size:
        raise ValueError("client_sizes create a client smaller than min_size.")

    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)
    class_indices = [np.where(targets == class_id)[0].tolist() for class_id in range(num_classes)]
    for indices in class_indices:
        rng.shuffle(indices)

    # Keep the old repo's intent: each client receives roughly the same number of
    # samples, but fix the original implementation so examples are assigned once.
    class_preferences = rng.dirichlet(np.full(num_classes, alpha), size=num_clients)
    client_indices = [[] for _ in range(num_clients)]
    remaining_per_class = np.asarray([len(indices) for indices in class_indices], dtype=np.int64)

    for client_id, target_size in enumerate(client_sizes):
        for _ in range(target_size):
            available = remaining_per_class > 0
            if not np.any(available):
                raise RuntimeError("Ran out of samples while building client partitions.")

            probs = class_preferences[client_id].copy()
            probs[~available] = 0.0
            if probs.sum() == 0:
                probs = available.astype(np.float64)
            probs /= probs.sum()

            class_id = int(rng.choice(num_classes, p=probs))
            client_indices[client_id].append(class_indices[class_id].pop())
            remaining_per_class[class_id] -= 1

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices
