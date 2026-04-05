import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dirichlet_data import build_dirichlet_partitions
from models import resnet18_cifar
from sam import SAM


@dataclass
class FedNSAMConfig:
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
    eval_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"


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

    local_state = clone_state_dict(model.state_dict())
    update = OrderedDict(
        (name, local_state[name] - extrapolated_state[name])
        for name, tensor in local_state.items()
        if torch.is_floating_point(tensor)
    )
    return update, total_loss / max(total_steps, 1)


def format_client_selection(selected_clients: Iterable[int]) -> str:
    sample = list(selected_clients)
    if len(sample) <= 8:
        return str(sample)
    return f"{sample[:8]} ... ({len(sample)} clients)"


def run_fednsam(config: FedNSAMConfig) -> dict[str, list[float]]:
    set_random_seed(config.seed)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, num_classes = build_cifar_datasets(config)
    client_sizes = build_equal_client_sizes(len(train_dataset), config.num_clients)
    partitions = build_dirichlet_partitions(
        targets=np.asarray(train_dataset.targets),
        num_clients=config.num_clients,
        alpha=config.alpha,
        seed=config.seed,
        client_sizes=client_sizes,
    )
    client_loaders = build_client_loaders(train_dataset, partitions, config, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    global_model = build_model(num_classes).to(device)
    global_state = clone_state_dict(global_model.state_dict())
    global_momentum = zero_update_like(global_state)
    history = {"round": [], "accuracy": [], "loss": []}

    clients_per_round = max(1, int(config.num_clients * config.client_fraction))
    print(
        f"FedNSAM | dataset={config.dataset} | clients={config.num_clients} | "
        f"selected={clients_per_round} | device={device}"
    )

    for round_idx in range(config.rounds):
        lr = cosine_lr(round_idx, config)
        selected_clients = np.random.choice(config.num_clients, clients_per_round, replace=False).tolist()
        local_updates: list[OrderedDict[str, torch.Tensor]] = []
        local_losses: list[float] = []

        for client_id in selected_clients:
            update, local_loss = local_fednsam_update(
                global_state=global_state,
                global_momentum=global_momentum,
                loader=client_loaders[client_id],
                num_classes=num_classes,
                lr=lr,
                config=config,
                device=device,
            )
            local_updates.append(update)
            local_losses.append(local_loss)

        avg_delta = average_updates(local_updates)
        global_momentum = update_global_momentum(global_momentum, avg_delta, config.gamma)
        global_state = apply_update(global_state, global_momentum)
        global_model.load_state_dict(to_device(global_state, device))

        should_eval = round_idx == 0 or (round_idx + 1) % config.eval_every == 0 or round_idx == config.rounds - 1
        if should_eval:
            accuracy, test_loss = evaluate(global_model, test_loader, device)
            history["round"].append(round_idx + 1)
            history["accuracy"].append(accuracy)
            history["loss"].append(test_loss)
            print(
                f"round={round_idx + 1:03d} | lr={lr:.5f} | "
                f"local_loss={np.mean(local_losses):.4f} | test_loss={test_loss:.4f} | "
                f"test_acc={accuracy * 100:.2f}% | clients={format_client_selection(selected_clients)}"
            )

    return history
