from __future__ import annotations

import math
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


def build_dpfedsam_dirichlet_partitions(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array.")

    rng = np.random.RandomState(seed)
    num_classes = int(targets.max()) + 1
    avg_examples = len(targets) / num_clients
    client_sizes = rng.lognormal(mean=np.log(avg_examples), sigma=0.0, size=num_clients)
    client_sizes = (client_sizes / np.sum(client_sizes) * len(targets)).astype(int)

    class_priors = rng.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    class_indices = [np.where(targets == class_id)[0] for class_id in range(num_classes)]
    class_amounts = [len(indices) for indices in class_indices]
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    while np.sum(client_sizes) != 0:
        client_id = int(rng.randint(num_clients))
        if client_sizes[client_id] <= 0:
            continue

        client_sizes[client_id] -= 1
        client_prior = prior_cumsum[client_id]
        while True:
            class_id = int(np.argmax(rng.uniform() <= client_prior))
            if class_amounts[class_id] <= 0:
                continue
            class_amounts[class_id] -= 1
            client_indices[client_id].append(int(class_indices[class_id][class_amounts[class_id]]))
            break

    class_counts = record_net_data_stats(targets, {idx: indices for idx, indices in enumerate(client_indices)})
    return client_indices, class_counts


def record_net_data_stats(targets: np.ndarray, net_dataidx_map: dict[int, list[int]]) -> list[list[int]]:
    targets = np.asarray(targets)
    num_classes = int(targets.max()) + 1
    class_counts: list[list[int]] = []
    for indices in net_dataidx_map.values():
        subset_targets = targets[np.asarray(indices, dtype=np.int64)]
        unique_labels, counts = np.unique(subset_targets, return_counts=True)
        stats = [0] * num_classes
        for label, count in zip(unique_labels, counts):
            stats[int(label)] = int(count)
        class_counts.append(stats)
    return class_counts


def build_dpfedsam_client_test_partitions(
    test_targets: np.ndarray,
    train_class_counts: list[list[int]],
    seed: int = 0,
) -> list[list[int]]:
    test_targets = np.asarray(test_targets)
    num_clients = len(train_class_counts)
    if num_clients == 0:
        return []

    rng = np.random.RandomState(seed)
    num_classes = int(test_targets.max()) + 1
    indices_per_class = [np.where(test_targets == label)[0] for label in range(num_classes)]
    tmp_test_size = int(math.ceil(len(test_targets) / num_clients))
    client_test_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for client_id, class_counts in enumerate(train_class_counts):
        total_train = sum(class_counts)
        if total_train == 0:
            continue
        selected: list[int] = []
        for label in range(num_classes):
            label_num = int(math.ceil(class_counts[label] / total_train * tmp_test_size))
            perm = rng.permutation(len(indices_per_class[label]))
            selected.extend(indices_per_class[label][perm[:label_num]].tolist())
        client_test_indices[client_id] = selected

    return client_test_indices
