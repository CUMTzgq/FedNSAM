import math
import random
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dirichlet_data import build_dirichlet_partitions
from models import resnet18_cifar
from privacy import add_gaussian_noise, clip_model_update, compute_epsilon, solve_noise_multiplier
from sam import SAM

ExperimentHistory = dict[str, object]


@dataclass
class FedNSAMConfig:
    algorithm: str = "fednsam"
    dataset: str = "cifar100"
    data_dir: str = "./data"
    rounds: int = 300
    num_clients: int = 100
    client_fraction: float = 0.1
    local_epochs: int = 5
    local_steps: int = 50
    batch_size: int = 50
    lr: float = 0.1
    min_lr: float = 0.0
    weight_decay: float = 1e-3
    rho: float = 0.05
    gamma: float = 0.85
    alpha: float = 0.1
    grad_clip: float | None = 10.0
    dp: bool = False
    dp_clip_norm: float | None = None
    dp_noise_multiplier: float | None = None
    dp_target_epsilon: float | None = None
    dp_delta: float | None = None
    eval_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    save_json: str | None = None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state_dict.items())


def zero_update_like(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, torch.zeros_like(tensor))
        for name, tensor in state_dict.items()
        if torch.is_floating_point(tensor)
    )


def apply_update(
    base_state: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
    alpha: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    new_state = clone_state_dict(base_state)
    for name, delta in update.items():
        new_state[name] = new_state[name] + alpha * delta
    return new_state


def average_updates(updates: list[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    avg = OrderedDict((name, torch.zeros_like(delta)) for name, delta in updates[0].items())
    scale = 1.0 / len(updates)
    for update in updates:
        for name, delta in update.items():
            avg[name].add_(delta, alpha=scale)
    return avg


def update_global_momentum(
    momentum: OrderedDict[str, torch.Tensor],
    avg_delta: OrderedDict[str, torch.Tensor],
    gamma: float,
) -> OrderedDict[str, torch.Tensor]:
    if not momentum:
        return clone_state_dict(avg_delta)
    return OrderedDict((name, gamma * momentum[name] + avg_delta[name]) for name in avg_delta)


def build_equal_client_sizes(num_examples: int, num_clients: int) -> list[int]:
    base = num_examples // num_clients
    remainder = num_examples % num_clients
    return [base + (1 if client_id < remainder else 0) for client_id in range(num_clients)]


def selected_clients_per_round(config: FedNSAMConfig) -> int:
    return max(1, int(config.num_clients * config.client_fraction))


def cosine_lr(round_idx: int, config: FedNSAMConfig) -> float:
    if config.rounds <= 1:
        return config.lr
    # Match the old training script's cosine schedule, while still guarding the
    # single-round case above.
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1.0 + math.cos(math.pi * round_idx / config.rounds)
    )


def build_cifar_datasets(config: FedNSAMConfig):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_cls = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
    train_dataset = dataset_cls(config.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = dataset_cls(config.data_dir, train=False, download=True, transform=test_transform)
    num_classes = 10 if config.dataset == "cifar10" else 100
    return train_dataset, test_dataset, num_classes


def build_client_selection_schedule(config: FedNSAMConfig) -> list[list[int]]:
    clients_per_round = selected_clients_per_round(config)
    rng = np.random.default_rng(config.seed + 7)
    return [
        rng.choice(config.num_clients, clients_per_round, replace=False).tolist()
        for _ in range(config.rounds)
    ]


def build_client_loaders(
    train_dataset,
    partitions: list[list[int]],
    config: FedNSAMConfig,
    device: torch.device,
) -> list[DataLoader]:
    loaders: list[DataLoader] = []
    pin_memory = device.type == "cuda"
    for client_id, indices in enumerate(partitions):
        generator = torch.Generator().manual_seed(config.seed + client_id)
        subset = Subset(train_dataset, indices)
        loaders.append(
            DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=pin_memory,
                generator=generator,
            )
        )
    return loaders


def build_model(num_classes: int) -> nn.Module:
    return resnet18_cifar(num_classes=num_classes)


def to_device(state_dict: OrderedDict[str, torch.Tensor], device: torch.device) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.to(device)) for name, tensor in state_dict.items())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)
    return total_correct / total_examples, total_loss / total_examples


def run_local_sgd(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    config: FedNSAMConfig,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.0,
        weight_decay=config.weight_decay,
    )
    model.train()
    total_loss = 0.0
    total_steps = 0
    stop = False
    for _ in range(config.local_epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1
            if total_steps >= config.local_steps:
                stop = True
                break
        if stop:
            break
    return total_loss / max(total_steps, 1)


def run_local_sam(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    config: FedNSAMConfig,
    device: torch.device,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=lr,
        momentum=0.0,
        weight_decay=config.weight_decay,
        rho=config.rho,
    )

    model.train()
    total_loss = 0.0
    total_steps = 0
    stop = False
    for _ in range(config.local_epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(images), targets).backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            total_steps += 1
            if total_steps >= config.local_steps:
                stop = True
                break
        if stop:
            break
    return total_loss / max(total_steps, 1)


def state_delta(
    reference_state: OrderedDict[str, torch.Tensor],
    model_state: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, model_state[name] - reference_state[name])
        for name, tensor in model_state.items()
        if torch.is_floating_point(tensor)
    )


def local_fedavg_update(
    global_state: OrderedDict[str, torch.Tensor],
    loader: DataLoader,
    num_classes: int,
    lr: float,
    config: FedNSAMConfig,
    device: torch.device,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    model = build_model(num_classes).to(device)
    model.load_state_dict(to_device(global_state, device))
    local_loss = run_local_sgd(model, loader, lr, config, device)
    local_state = clone_state_dict(model.state_dict())
    return state_delta(global_state, local_state), local_loss


def local_fedsam_update(
    global_state: OrderedDict[str, torch.Tensor],
    loader: DataLoader,
    num_classes: int,
    lr: float,
    config: FedNSAMConfig,
    device: torch.device,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    model = build_model(num_classes).to(device)
    model.load_state_dict(to_device(global_state, device))
    local_loss = run_local_sam(model, loader, lr, config, device)
    local_state = clone_state_dict(model.state_dict())
    return state_delta(global_state, local_state), local_loss


def local_fednsam_update(
    global_state: OrderedDict[str, torch.Tensor],
    global_momentum: OrderedDict[str, torch.Tensor],
    loader: DataLoader,
    num_classes: int,
    lr: float,
    config: FedNSAMConfig,
    device: torch.device,
) -> tuple[OrderedDict[str, torch.Tensor], float]:
    extrapolated_state = apply_update(global_state, global_momentum, alpha=config.gamma)
    model = build_model(num_classes).to(device)
    model.load_state_dict(to_device(extrapolated_state, device))
    local_loss = run_local_sam(model, loader, lr, config, device)
    local_state = clone_state_dict(model.state_dict())
    return state_delta(extrapolated_state, local_state), local_loss


def format_client_selection(selected_clients: Iterable[int]) -> str:
    sample = list(selected_clients)
    if len(sample) <= 8:
        return str(sample)
    return f"{sample[:8]} ... ({len(sample)} clients)"


def normalize_algorithm_name(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "avg": "fedavg",
        "fedavg": "fedavg",
        "sam": "fedsam",
        "fedsam": "fedsam",
        "nsam": "fednsam",
        "fednsam": "fednsam",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported algorithm: {name}")
    return aliases[normalized]


def get_local_update_fn(algorithm: str):
    if algorithm == "fedavg":
        return local_fedavg_update
    if algorithm == "fedsam":
        return local_fedsam_update
    if algorithm == "fednsam":
        return local_fednsam_update
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def resolve_privacy_settings(config: FedNSAMConfig, train_examples: int) -> dict[str, float | str | bool | None]:
    if not config.dp:
        return {
            "enabled": False,
            "clip_norm": None,
            "noise_multiplier": None,
            "target_epsilon": None,
            "delta": None,
            "sample_rate": None,
            "source": None,
            "final_epsilon": None,
            "optimal_order": None,
        }

    if config.dp_clip_norm is None:
        raise ValueError("DP is enabled but no clip norm was provided.")

    delta = config.dp_delta if config.dp_delta is not None else 1.0 / train_examples
    sample_rate = selected_clients_per_round(config) / config.num_clients
    if config.dp_noise_multiplier is not None:
        noise_multiplier = config.dp_noise_multiplier
        source = "sigma"
    elif config.dp_target_epsilon is not None:
        noise_multiplier = solve_noise_multiplier(
            target_epsilon=config.dp_target_epsilon,
            sample_rate=sample_rate,
            steps=config.rounds,
            delta=delta,
        )
        source = "eps"
    else:
        raise ValueError("DP is enabled but neither sigma nor eps was provided.")

    final_epsilon, optimal_order = compute_epsilon(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        steps=config.rounds,
        delta=delta,
    )
    return {
        "enabled": True,
        "clip_norm": config.dp_clip_norm,
        "noise_multiplier": noise_multiplier,
        "target_epsilon": config.dp_target_epsilon,
        "delta": delta,
        "sample_rate": sample_rate,
        "source": source,
        "final_epsilon": final_epsilon,
        "optimal_order": optimal_order,
    }


def run_single_experiment(
    config: FedNSAMConfig,
    algorithm: str,
    initial_state: OrderedDict[str, torch.Tensor],
    train_dataset,
    partitions: list[list[int]],
    selection_schedule: list[list[int]],
    test_dataset,
    num_classes: int,
    privacy_settings: dict[str, float | str | bool | None],
) -> ExperimentHistory:
    algorithm = normalize_algorithm_name(algorithm)
    set_random_seed(config.seed)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    client_loaders = build_client_loaders(train_dataset, partitions, config, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    global_model = build_model(num_classes).to(device)
    global_state = clone_state_dict(initial_state)
    global_model.load_state_dict(to_device(global_state, device))
    global_momentum = zero_update_like(global_state)
    history: ExperimentHistory = {
        "algorithm": algorithm,
        "round": [],
        "accuracy": [],
        "loss": [],
        "epsilon": [],
        "dp_enabled": privacy_settings["enabled"],
        "dp_clip_norm": privacy_settings["clip_norm"],
        "dp_noise_multiplier": privacy_settings["noise_multiplier"],
        "dp_target_epsilon": privacy_settings["target_epsilon"],
        "dp_delta": privacy_settings["delta"],
        "dp_sample_rate": privacy_settings["sample_rate"],
        "dp_source": privacy_settings["source"],
    }

    print(
        f"{algorithm.upper()} | dataset={config.dataset} | clients={config.num_clients} | "
        f"selected={len(selection_schedule[0])} | device={device}"
    )

    local_update_fn = get_local_update_fn(algorithm)
    for round_idx in range(config.rounds):
        lr = cosine_lr(round_idx, config)
        selected_clients = selection_schedule[round_idx]
        local_updates: list[OrderedDict[str, torch.Tensor]] = []
        local_losses: list[float] = []

        for client_id in selected_clients:
            if algorithm == "fednsam":
                update, local_loss = local_update_fn(
                    global_state=global_state,
                    global_momentum=global_momentum,
                    loader=client_loaders[client_id],
                    num_classes=num_classes,
                    lr=lr,
                    config=config,
                    device=device,
                )
            else:
                update, local_loss = local_update_fn(
                    global_state=global_state,
                    loader=client_loaders[client_id],
                    num_classes=num_classes,
                    lr=lr,
                    config=config,
                    device=device,
                )
            if privacy_settings["enabled"]:
                update, _, _ = clip_model_update(update, float(privacy_settings["clip_norm"]))
            local_updates.append(update)
            local_losses.append(local_loss)

        avg_delta = average_updates(local_updates)
        if privacy_settings["enabled"]:
            noise_std = (
                float(privacy_settings["noise_multiplier"]) * float(privacy_settings["clip_norm"]) / len(selected_clients)
            )
            avg_delta = add_gaussian_noise(avg_delta, noise_std)
        if algorithm == "fednsam":
            global_momentum = update_global_momentum(global_momentum, avg_delta, config.gamma)
            global_state = apply_update(global_state, global_momentum)
        else:
            global_state = apply_update(global_state, avg_delta)
        global_model.load_state_dict(to_device(global_state, device))

        should_eval = round_idx == 0 or (round_idx + 1) % config.eval_every == 0 or round_idx == config.rounds - 1
        if should_eval:
            accuracy, test_loss = evaluate(global_model, test_loader, device)
            history["round"].append(round_idx + 1)
            history["accuracy"].append(accuracy)
            history["loss"].append(test_loss)
            if privacy_settings["enabled"]:
                epsilon, _ = compute_epsilon(
                    noise_multiplier=float(privacy_settings["noise_multiplier"]),
                    sample_rate=float(privacy_settings["sample_rate"]),
                    steps=round_idx + 1,
                    delta=float(privacy_settings["delta"]),
                )
                history["epsilon"].append(epsilon)
            print(
                f"{algorithm.upper()} round={round_idx + 1:03d} | lr={lr:.5f} | "
                f"local_loss={np.mean(local_losses):.4f} | test_loss={test_loss:.4f} | "
                f"test_acc={accuracy * 100:.2f}%"
                + (
                    f" | eps={history['epsilon'][-1]:.4f}"
                    if privacy_settings["enabled"] and history["epsilon"]
                    else ""
                )
                + f" | clients={format_client_selection(selected_clients)}"
            )

    return history


def save_histories(histories: dict[str, ExperimentHistory], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(histories, fp, indent=2)


def compare_histories(config: FedNSAMConfig, algorithms: list[str]) -> dict[str, ExperimentHistory]:
    set_random_seed(config.seed)
    train_dataset, test_dataset, num_classes = build_cifar_datasets(config)
    client_sizes = build_equal_client_sizes(len(train_dataset), config.num_clients)
    partitions = build_dirichlet_partitions(
        targets=np.asarray(train_dataset.targets),
        num_clients=config.num_clients,
        alpha=config.alpha,
        seed=config.seed,
        client_sizes=client_sizes,
    )
    selection_schedule = build_client_selection_schedule(config)
    privacy_settings = resolve_privacy_settings(config, len(train_dataset))

    set_random_seed(config.seed)
    initial_state = clone_state_dict(build_model(num_classes).state_dict())

    if privacy_settings["enabled"]:
        print(
            "DP enabled | "
            f"clip={float(privacy_settings['clip_norm']):.4f} | "
            f"sigma={float(privacy_settings['noise_multiplier']):.6f} | "
            f"delta={float(privacy_settings['delta']):.6g} | "
            f"q={float(privacy_settings['sample_rate']):.4f}"
            + (
                f" | target_eps={float(privacy_settings['target_epsilon']):.4f}"
                if privacy_settings["target_epsilon"] is not None
                else ""
            )
            + f" | final_eps={float(privacy_settings['final_epsilon']):.4f}"
        )

    histories: dict[str, ExperimentHistory] = {}
    for algorithm in algorithms:
        histories[algorithm] = run_single_experiment(
            config=config,
            algorithm=algorithm,
            initial_state=initial_state,
            train_dataset=train_dataset,
            partitions=partitions,
            selection_schedule=selection_schedule,
            test_dataset=test_dataset,
            num_classes=num_classes,
            privacy_settings=privacy_settings,
        )

    print("\nSummary")
    for algorithm in algorithms:
        history = histories[algorithm]
        accuracy_history = history["accuracy"]
        assert isinstance(accuracy_history, list)
        best_acc = max(accuracy_history) if accuracy_history else float("nan")
        final_acc = accuracy_history[-1] if accuracy_history else float("nan")
        summary = f"{algorithm.upper():8s} | best_acc={best_acc * 100:.2f}% | final_acc={final_acc * 100:.2f}%"
        epsilon_history = history["epsilon"]
        assert isinstance(epsilon_history, list)
        if epsilon_history:
            summary += f" | final_eps={epsilon_history[-1]:.4f}"
        print(summary)

    if config.save_json:
        save_histories(histories, config.save_json)
        print(f"Saved results to {config.save_json}")

    return histories


def run_fednsam(config: FedNSAMConfig) -> ExperimentHistory:
    history = compare_histories(config, [normalize_algorithm_name(config.algorithm)])
    return history[normalize_algorithm_name(config.algorithm)]
