import copy
import json
import math
import os
import queue
import random
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dirichlet_data import (
    build_dirichlet_partitions,
    build_dpfedsam_client_test_partitions,
    build_dpfedsam_dirichlet_partitions,
)
from models import resnet18_cifar
from privacy import (
    add_tensorwise_gaussian_noise_,
    clip_tensor_updates_,
    compute_epsilon,
    solve_noise_multiplier,
)
from sam import SAM

ExperimentHistory = dict[str, object]
ProgressHook = Callable[[dict[str, object]], None]
RUNTIME_OVERRIDE_FIELDS = {"device", "devices", "compare_parallel", "fast_cuda", "amp", "save_json", "ckpt_dir"}


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
    lr_schedule: str = "auto"
    lr_decay: float = 1.0
    min_lr: float = 0.0
    momentum: float = 0.0
    weight_decay: float = 1e-3
    rho: float = 0.05
    gamma: float = 0.85
    alpha: float = 0.1
    grad_clip: float | None = 10.0
    dp: bool = False
    dp_clip_norm: float | None = None
    dp_clip_decay: float = 1.0
    dp_clip_min: float | None = None
    dp_noise_multiplier: float | None = None
    dp_target_epsilon: float | None = None
    dp_delta: float | None = None
    eval_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    devices: tuple[str, ...] = ()
    compare_parallel: bool = False
    fast_cuda: bool = False
    amp: str = "off"
    save_json: str | None = None
    ckpt_dir: str | None = None
    resume: str | None = None
    explicit_cli_fields: tuple[str, ...] = ()
    explicit_compare: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    amp_dtype: torch.dtype | None
    channels_last: bool
    use_grad_scaler: bool


CONFIG_DEFAULTS = FedNSAMConfig()
DP_FEDSAM_DEFAULTS: dict[str, object] = {
    "rounds": 300,
    "num_clients": 500,
    "client_fraction": 0.1,
    "local_epochs": 30,
    "batch_size": 50,
    "rho": 0.5,
    "momentum": 0.5,
    "weight_decay": 5e-4,
    "lr_decay": 0.998,
    "alpha": 0.6,
}


def field_was_explicitly_set(config: FedNSAMConfig, field_name: str) -> bool:
    if config.explicit_cli_fields:
        return field_name in config.explicit_cli_fields
    return getattr(config, field_name) != getattr(CONFIG_DEFAULTS, field_name)


def resolve_effective_config(config: FedNSAMConfig) -> FedNSAMConfig:
    if not config.dp:
        return copy.deepcopy(config)

    resolved = copy.deepcopy(config)
    for field_name, value in DP_FEDSAM_DEFAULTS.items():
        if not field_was_explicitly_set(config, field_name):
            setattr(resolved, field_name, value)
    if not field_was_explicitly_set(config, "grad_clip"):
        resolved.grad_clip = None
    return resolved


def dp_uses_local_step_cap(config: FedNSAMConfig) -> bool:
    if not config.dp:
        return True
    return field_was_explicitly_set(config, "local_steps")


def effective_local_step_limit(config: FedNSAMConfig) -> int | None:
    if dp_uses_local_step_cap(config):
        return config.local_steps
    return None


def effective_grad_clip(config: FedNSAMConfig) -> float | None:
    if not config.dp:
        return config.grad_clip
    if field_was_explicitly_set(config, "grad_clip"):
        return config.grad_clip
    return None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def configure_torch_runtime(config: FedNSAMConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = not config.fast_cuda
    torch.backends.cudnn.benchmark = config.fast_cuda
    torch.backends.cuda.matmul.allow_tf32 = config.fast_cuda
    torch.backends.cudnn.allow_tf32 = config.fast_cuda
    if config.fast_cuda:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")


def resolve_amp_dtype(config: FedNSAMConfig, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or config.amp == "off":
        return None
    if config.amp == "bf16":
        return torch.bfloat16
    if config.amp == "fp16":
        return torch.float16
    if config.amp == "auto":
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16_supported else torch.float16
    raise ValueError(f"Unsupported AMP mode: {config.amp}")


def build_runtime_config(config: FedNSAMConfig) -> RuntimeConfig:
    device = resolve_device(config.device)
    configure_torch_runtime(config, device)
    amp_dtype = resolve_amp_dtype(config, device)
    channels_last = config.fast_cuda and device.type == "cuda"
    return RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        use_grad_scaler=device.type == "cuda" and amp_dtype == torch.float16,
    )


def resolve_compare_devices(config: FedNSAMConfig) -> tuple[str, ...]:
    if config.devices:
        return tuple(config.devices)
    return (config.device,)


def compare_parallel_thread_limit(config: FedNSAMConfig) -> int:
    cpu_count = os.cpu_count() or 1
    device_count = max(1, len(resolve_compare_devices(config)))
    return max(1, cpu_count // device_count)


def configure_parallel_worker_threads(config: FedNSAMConfig) -> None:
    limit = compare_parallel_thread_limit(config)
    torch.set_num_threads(limit)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, limit)))
        except RuntimeError:
            pass


def clone_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    cloned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        detached = tensor.detach()
        if device is not None:
            detached = detached.to(device, non_blocking=device.type == "cuda")
        cloned[name] = detached.clone()
    return cloned


def zero_update_like(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, torch.zeros_like(tensor))
        for name, tensor in state_dict.items()
        if torch.is_floating_point(tensor)
    )


@torch.no_grad()
def add_update_(
    base_state: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
    alpha: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    for name, delta in update.items():
        base_state[name].add_(delta, alpha=alpha)
    return base_state


def apply_update(
    base_state: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
    alpha: float = 1.0,
) -> OrderedDict[str, torch.Tensor]:
    new_state = clone_state_dict(base_state)
    return add_update_(new_state, update, alpha=alpha)


@torch.no_grad()
def accumulate_update_(
    average_update: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
    scale: float,
) -> OrderedDict[str, torch.Tensor]:
    for name, delta in update.items():
        average_update[name].add_(delta, alpha=scale)
    return average_update


@torch.no_grad()
def update_global_momentum_(
    momentum: OrderedDict[str, torch.Tensor],
    avg_delta: OrderedDict[str, torch.Tensor],
    gamma: float,
) -> OrderedDict[str, torch.Tensor]:
    for name, delta in avg_delta.items():
        momentum[name].mul_(gamma).add_(delta)
    return momentum


def build_equal_client_sizes(num_examples: int, num_clients: int) -> list[int]:
    base = num_examples // num_clients
    remainder = num_examples % num_clients
    return [base + (1 if client_id < remainder else 0) for client_id in range(num_clients)]


def selected_clients_per_round(config: FedNSAMConfig) -> int:
    return max(1, int(config.num_clients * config.client_fraction))


def cosine_lr(round_idx: int, config: FedNSAMConfig) -> float:
    if config.rounds <= 1:
        return config.lr
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1.0 + math.cos(math.pi * round_idx / config.rounds)
    )


def build_eval_rounds(config: FedNSAMConfig) -> set[int]:
    eval_rounds = {1, config.rounds}
    eval_rounds.update(range(config.eval_every, config.rounds + 1, config.eval_every))
    return eval_rounds


def cifar_mean_std(dataset: str, *, dp_aligned: bool) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if dataset == "cifar10":
        if dp_aligned:
            return (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if dp_aligned:
        return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def build_cifar_datasets(config: FedNSAMConfig):
    mean, std = cifar_mean_std(config.dataset, dp_aligned=config.dp)
    train_transforms: list[object] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if config.dp:
        train_transforms.extend(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(),
            ]
        )
    train_transform = transforms.Compose(
        train_transforms
        + [
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
    if config.dp:
        schedule: list[list[int]] = []
        for round_idx in range(config.rounds):
            if config.num_clients == clients_per_round:
                schedule.append(list(range(config.num_clients)))
                continue
            rng = np.random.RandomState(round_idx)
            chosen = rng.choice(range(config.num_clients), clients_per_round, replace=False)
            schedule.append(chosen.tolist())
        return schedule
    rng = np.random.default_rng(config.seed + 7)
    return [
        rng.choice(config.num_clients, clients_per_round, replace=False).tolist()
        for _ in range(config.rounds)
    ]


def build_loader_kwargs(config: FedNSAMConfig, device: torch.device) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_client_loaders(
    train_dataset,
    partitions: list[list[int]],
    config: FedNSAMConfig,
    device: torch.device,
) -> list[DataLoader]:
    loaders: list[DataLoader] = []
    loader_kwargs = build_loader_kwargs(config, device)
    for client_id, indices in enumerate(partitions):
        generator = torch.Generator().manual_seed(config.seed + client_id)
        subset = Subset(train_dataset, indices)
        loaders.append(
            DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                **loader_kwargs,
            )
        )
    return loaders


def build_client_eval_loaders(
    test_dataset,
    partitions: list[list[int]],
    config: FedNSAMConfig,
    device: torch.device,
) -> list[DataLoader]:
    loaders: list[DataLoader] = []
    loader_kwargs = build_loader_kwargs(config, device)
    for indices in partitions:
        subset = Subset(test_dataset, indices)
        loaders.append(
            DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=False,
                **loader_kwargs,
            )
        )
    return loaders


def set_round_loader_seed(
    loaders: list[DataLoader],
    seed: int,
    round_idx: int,
    selected_clients: Iterable[int],
) -> None:
    round_seed_base = seed + (round_idx + 1) * 100_000
    for client_id in selected_clients:
        generator = getattr(loaders[client_id], "generator", None)
        if generator is not None:
            generator.manual_seed(round_seed_base + client_id)


def algorithm_seed_offset(algorithm: str) -> int:
    offset = 0
    for index, byte in enumerate(algorithm.encode("utf-8"), start=1):
        offset = (offset * 257 + index * byte) % (2**63 - 1)
    return offset


def build_round_noise_generator(
    seed: int,
    algorithm: str,
    round_idx: int,
    device: torch.device,
) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    round_seed = seed + 1_000_000_007 + algorithm_seed_offset(algorithm) + (round_idx + 1) * 1_000_003
    generator.manual_seed(round_seed)
    return generator


def build_client_noise_generator(
    seed: int,
    algorithm: str,
    round_idx: int,
    client_id: int,
    device: torch.device,
) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    client_seed = (
        seed
        + 2_000_000_033
        + algorithm_seed_offset(algorithm)
        + (round_idx + 1) * 1_000_003
        + (client_id + 1) * 9_973
    )
    generator.manual_seed(client_seed)
    return generator


def build_test_loader(test_dataset, config: FedNSAMConfig, device: torch.device) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        **build_loader_kwargs(config, device),
    )


def build_model(num_classes: int, runtime: RuntimeConfig) -> nn.Module:
    model = resnet18_cifar(num_classes=num_classes).to(runtime.device)
    if runtime.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


@torch.no_grad()
def copy_state_into_model(model: nn.Module, state_dict: OrderedDict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    for name, tensor in model_state.items():
        tensor.copy_(state_dict[name], non_blocking=tensor.device.type == "cuda")


def state_delta(
    reference_state: OrderedDict[str, torch.Tensor],
    model_state: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach() - reference_state[name])
        for name, tensor in model_state.items()
        if torch.is_floating_point(tensor)
    )


def prepare_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    runtime: RuntimeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = runtime.device.type == "cuda"
    if runtime.channels_last and images.ndim == 4:
        images = images.to(
            runtime.device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    else:
        images = images.to(runtime.device, non_blocking=non_blocking)
    targets = targets.to(runtime.device, non_blocking=non_blocking)
    return images, targets


def autocast_context(runtime: RuntimeConfig):
    if runtime.amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=runtime.amp_dtype)


def gradients_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def evaluate(model: nn.Module, loader: DataLoader, runtime: RuntimeConfig) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = prepare_batch(images, targets, runtime)
            with autocast_context(runtime):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)
    return total_correct / total_examples, total_loss / total_examples


def evaluate_client_average(
    model: nn.Module,
    loaders: list[DataLoader],
    runtime: RuntimeConfig,
) -> tuple[float, float]:
    client_accuracies: list[float] = []
    client_losses: list[float] = []
    for loader in loaders:
        if len(loader.dataset) == 0:
            continue
        accuracy, loss = evaluate(model, loader, runtime)
        client_accuracies.append(accuracy)
        client_losses.append(loss)
    if not client_accuracies:
        return 0.0, 0.0
    return float(np.mean(client_accuracies)), float(np.mean(client_losses))


def run_local_sgd(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    config: FedNSAMConfig,
    runtime: RuntimeConfig,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime.use_grad_scaler else None
    step_limit = effective_local_step_limit(config)
    grad_clip = effective_grad_clip(config)

    model.train()
    total_loss = 0.0
    total_steps = 0
    stop = False
    for _ in range(config.local_epochs):
        for images, targets in loader:
            images, targets = prepare_batch(images, targets, runtime)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                loss = criterion(model(images), targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1
            if step_limit is not None and total_steps >= step_limit:
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
    runtime: RuntimeConfig,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        rho=config.rho,
        adaptive=config.dp,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime.use_grad_scaler else None
    step_limit = effective_local_step_limit(config)
    grad_clip = effective_grad_clip(config)

    model.train()
    total_loss = 0.0
    total_steps = 0
    stop = False
    for _ in range(config.local_epochs):
        for images, targets in loader:
            images, targets = prepare_batch(images, targets, runtime)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                loss = criterion(model(images), targets)
            loss.backward()
            if not gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.first_step(zero_grad=True)

            with autocast_context(runtime):
                sharpness_loss = criterion(model(images), targets)

            if scaler is not None:
                scaler.scale(sharpness_loss).backward()
                scaler.unscale_(optimizer.base_optimizer)
                if not gradients_are_finite(model):
                    optimizer.restore_step(zero_grad=True)
                    scaler.update()
                    continue
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.restore_step(zero_grad=False)
                scaler.step(optimizer.base_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                sharpness_loss.backward()
                if not gradients_are_finite(model):
                    optimizer.restore_step(zero_grad=True)
                    continue
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)

            total_loss += loss.item()
            total_steps += 1
            if step_limit is not None and total_steps >= step_limit:
                stop = True
                break
        if stop:
            break
    return total_loss / max(total_steps, 1)


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


def normalize_algorithms(algorithms: list[str] | None, default_algorithm: str) -> list[str]:
    if algorithms is None:
        return [normalize_algorithm_name(default_algorithm)]
    return [normalize_algorithm_name(name) for name in algorithms]


def get_local_trainer(algorithm: str):
    if algorithm == "fedavg":
        return run_local_sgd
    if algorithm in {"fedsam", "fednsam"}:
        return run_local_sam
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def resolve_privacy_settings(config: FedNSAMConfig, train_examples: int) -> dict[str, float | str | bool | None]:
    if not config.dp:
        return {
            "enabled": False,
            "style": None,
            "clip_norm": None,
            "clip_schedule": None,
            "clip_decay": None,
            "clip_min": None,
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
    clip_min = config.dp_clip_norm if config.dp_clip_min is None else config.dp_clip_min

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
        "style": "dpfedsam",
        "clip_norm": config.dp_clip_norm,
        "clip_schedule": "exp",
        "clip_decay": config.dp_clip_decay,
        "clip_min": clip_min,
        "noise_multiplier": noise_multiplier,
        "target_epsilon": config.dp_target_epsilon,
        "delta": delta,
        "sample_rate": sample_rate,
        "source": source,
        "final_epsilon": final_epsilon,
        "optimal_order": optimal_order,
    }


def build_epsilon_trace(
    privacy_settings: dict[str, float | str | bool | None],
    eval_rounds: set[int],
) -> dict[int, float]:
    if not privacy_settings["enabled"]:
        return {}

    epsilon_trace: dict[int, float] = {}
    for step in sorted(eval_rounds):
        epsilon, _ = compute_epsilon(
            noise_multiplier=float(privacy_settings["noise_multiplier"]),
            sample_rate=float(privacy_settings["sample_rate"]),
            steps=step,
            delta=float(privacy_settings["delta"]),
        )
        epsilon_trace[step] = epsilon
    return epsilon_trace


def resolve_round_clip_norm(
    round_idx: int,
    privacy_settings: dict[str, float | str | bool | None],
) -> float:
    if not privacy_settings["enabled"]:
        raise ValueError("Round clip norm is only defined when DP is enabled.")
    clip_norm = float(privacy_settings["clip_norm"])
    clip_decay = float(privacy_settings.get("clip_decay") or 1.0)
    clip_min = float(privacy_settings.get("clip_min") or clip_norm)
    return max(clip_min, clip_norm * (clip_decay**round_idx))


def round_learning_rate(round_idx: int, config: FedNSAMConfig) -> float:
    if config.lr_schedule == "cosine":
        return cosine_lr(round_idx, config)
    if config.lr_schedule == "exp":
        return config.lr * (config.lr_decay**round_idx)
    if config.lr_schedule != "auto":
        raise ValueError(f"Unsupported lr schedule: {config.lr_schedule}")
    if config.dp:
        return config.lr * (config.lr_decay**round_idx)
    return cosine_lr(round_idx, config)


def copy_history(history: ExperimentHistory) -> ExperimentHistory:
    return copy.deepcopy(history)


def copy_histories(histories: dict[str, ExperimentHistory]) -> dict[str, ExperimentHistory]:
    return copy.deepcopy(histories)


def atomic_write_json(data: object, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_torch_save(obj: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def save_histories(histories: dict[str, ExperimentHistory], output_path: str) -> None:
    atomic_write_json(histories, output_path)


def serialize_config(config: FedNSAMConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized.pop("resume", None)
    serialized.pop("explicit_cli_fields", None)
    serialized.pop("explicit_compare", None)
    return serialized


def deserialize_config(data: dict[str, object]) -> FedNSAMConfig:
    return FedNSAMConfig(**data)


def checkpoint_path(config: FedNSAMConfig) -> Path | None:
    if config.ckpt_dir is not None:
        return Path(config.ckpt_dir) / "latest.pt"
    if config.resume is not None:
        return Path(config.resume).resolve().parent / "latest.pt"
    return None


def save_latest_checkpoint(
    config: FedNSAMConfig,
    algorithms: list[str],
    histories: dict[str, ExperimentHistory],
    current_algorithm_index: int,
    current_algorithm: str | None,
    current_round: int,
    current_state: dict[str, object] | None,
    shared_context: dict[str, object],
) -> None:
    path = checkpoint_path(config)
    if path is None:
        return

    checkpoint = {
        "version": 1,
        "config": serialize_config(config),
        "algorithms": list(algorithms),
        "histories": copy_histories(histories),
        "current_algorithm_index": current_algorithm_index,
        "current_algorithm": current_algorithm,
        "current_round": current_round,
        "current_state": current_state,
        "shared_context": shared_context,
        "completed": current_algorithm_index >= len(algorithms),
    }
    atomic_torch_save(checkpoint, path)


def load_checkpoint(checkpoint_file: str) -> dict[str, object]:
    return torch.load(checkpoint_file, map_location="cpu", weights_only=False)


def apply_resume_config(
    cli_config: FedNSAMConfig,
    checkpoint: dict[str, object],
    algorithms: list[str] | None,
) -> tuple[FedNSAMConfig, list[str]]:
    checkpoint_config = deserialize_config(dict(checkpoint["config"]))
    explicit_fields = set(cli_config.explicit_cli_fields)
    blocked_overrides = (explicit_fields - RUNTIME_OVERRIDE_FIELDS) - {"resume"}
    for field in blocked_overrides:
        cli_value = getattr(cli_config, field)
        checkpoint_value = getattr(checkpoint_config, field)
        if cli_value != checkpoint_value:
            raise ValueError(f"Cannot override '{field}' when resuming from a checkpoint.")

    resumed_config = deserialize_config(dict(checkpoint["config"]))
    resumed_config.resume = cli_config.resume
    resumed_config.explicit_cli_fields = cli_config.explicit_cli_fields
    resumed_config.explicit_compare = cli_config.explicit_compare
    if "device" in explicit_fields:
        resumed_config.device = cli_config.device
    if "fast_cuda" in explicit_fields:
        resumed_config.fast_cuda = cli_config.fast_cuda
    if "amp" in explicit_fields:
        resumed_config.amp = cli_config.amp
    if "save_json" in explicit_fields:
        resumed_config.save_json = cli_config.save_json
    if "ckpt_dir" in explicit_fields:
        resumed_config.ckpt_dir = cli_config.ckpt_dir
    elif resumed_config.ckpt_dir is None and cli_config.resume is not None:
        resumed_config.ckpt_dir = str(Path(cli_config.resume).resolve().parent)

    checkpoint_algorithms = [normalize_algorithm_name(name) for name in checkpoint["algorithms"]]
    if algorithms is not None:
        normalized_cli_algorithms = [normalize_algorithm_name(name) for name in algorithms]
        if normalized_cli_algorithms != checkpoint_algorithms:
            raise ValueError("Cannot override the compared algorithms when resuming from a checkpoint.")
    return resumed_config, checkpoint_algorithms


def build_shared_context(
    config: FedNSAMConfig,
    train_dataset,
    test_dataset,
    num_classes: int,
) -> dict[str, object]:
    targets = np.asarray(train_dataset.targets)
    if config.dp:
        partitions, train_class_counts = build_dpfedsam_dirichlet_partitions(
            targets=targets,
            num_clients=config.num_clients,
            alpha=config.alpha,
            seed=config.seed,
        )
        test_partitions = build_dpfedsam_client_test_partitions(
            test_targets=np.asarray(test_dataset.targets),
            train_class_counts=train_class_counts,
            seed=config.seed,
        )
        evaluation_mode = "client_average"
    else:
        client_sizes = build_equal_client_sizes(len(train_dataset), config.num_clients)
        partitions = build_dirichlet_partitions(
            targets=targets,
            num_clients=config.num_clients,
            alpha=config.alpha,
            seed=config.seed,
            client_sizes=client_sizes,
        )
        test_partitions = None
        evaluation_mode = "global"
    selection_schedule = build_client_selection_schedule(config)
    privacy_settings = resolve_privacy_settings(config, len(train_dataset))
    eval_rounds = build_eval_rounds(config)
    epsilon_trace = build_epsilon_trace(privacy_settings, eval_rounds)

    set_random_seed(config.seed)
    initial_state = clone_state_dict(resnet18_cifar(num_classes=num_classes).state_dict())
    return {
        "partitions": partitions,
        "selection_schedule": selection_schedule,
        "privacy_settings": privacy_settings,
        "eval_rounds": sorted(eval_rounds),
        "epsilon_trace": epsilon_trace,
        "initial_state": initial_state,
        "num_classes": num_classes,
        "test_partitions": test_partitions,
        "evaluation_mode": evaluation_mode,
    }


def normalize_shared_context(shared_context: dict[str, object]) -> dict[str, object]:
    normalized = dict(shared_context)
    normalized["partitions"] = [list(indices) for indices in shared_context["partitions"]]
    normalized["selection_schedule"] = [list(indices) for indices in shared_context["selection_schedule"]]
    normalized["privacy_settings"] = dict(shared_context["privacy_settings"])
    normalized["eval_rounds"] = [int(round_id) for round_id in shared_context["eval_rounds"]]
    normalized["epsilon_trace"] = {int(key): float(value) for key, value in shared_context["epsilon_trace"].items()}
    normalized["initial_state"] = clone_state_dict(shared_context["initial_state"])
    normalized["num_classes"] = int(shared_context["num_classes"])
    test_partitions = shared_context.get("test_partitions")
    normalized["test_partitions"] = None if test_partitions is None else [list(indices) for indices in test_partitions]
    normalized["evaluation_mode"] = str(shared_context.get("evaluation_mode", "global"))
    return normalized


def print_privacy_banner(privacy_settings: dict[str, float | str | bool | None]) -> None:
    if not privacy_settings["enabled"]:
        return
    print(
        "DP enabled | "
        f"clip={float(privacy_settings['clip_norm']):.4f} | "
        f"clip_decay={float(privacy_settings.get('clip_decay') or 1.0):.6f} | "
        f"clip_min={float(privacy_settings.get('clip_min') or privacy_settings['clip_norm']):.4f} | "
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


def run_single_experiment(
    config: FedNSAMConfig,
    algorithm: str,
    initial_state: OrderedDict[str, torch.Tensor],
    selection_schedule: list[list[int]],
    num_classes: int,
    privacy_settings: dict[str, float | str | bool | None],
    runtime: RuntimeConfig,
    client_loaders: list[DataLoader],
    test_loader: DataLoader | None,
    client_test_loaders: list[DataLoader] | None,
    evaluation_mode: str,
    eval_rounds: set[int],
    epsilon_trace: dict[int, float],
    start_round: int = 0,
    initial_global_state: OrderedDict[str, torch.Tensor] | None = None,
    initial_global_momentum: OrderedDict[str, torch.Tensor] | None = None,
    initial_history: ExperimentHistory | None = None,
    round_callback: ProgressHook | None = None,
) -> ExperimentHistory:
    algorithm = normalize_algorithm_name(algorithm)
    set_random_seed(config.seed)

    work_model = build_model(num_classes, runtime)
    global_state = (
        clone_state_dict(initial_global_state, device=runtime.device)
        if initial_global_state is not None
        else clone_state_dict(initial_state, device=runtime.device)
    )
    global_momentum = (
        clone_state_dict(initial_global_momentum, device=runtime.device)
        if initial_global_momentum is not None
        else zero_update_like(global_state)
    )
    history: ExperimentHistory = copy_history(initial_history) if initial_history is not None else {
        "algorithm": algorithm,
        "round": [],
        "accuracy": [],
        "loss": [],
        "epsilon": [],
        "dp_enabled": privacy_settings["enabled"],
        "dp_clip_norm": privacy_settings["clip_norm"],
        "dp_clip_schedule": privacy_settings.get("clip_schedule"),
        "dp_clip_decay": privacy_settings.get("clip_decay"),
        "dp_clip_min": privacy_settings.get("clip_min"),
        "clip_norm": [],
        "dp_noise_multiplier": privacy_settings["noise_multiplier"],
        "dp_target_epsilon": privacy_settings["target_epsilon"],
        "dp_delta": privacy_settings["delta"],
        "dp_sample_rate": privacy_settings["sample_rate"],
        "dp_source": privacy_settings["source"],
        "dp_style": privacy_settings["style"],
        "evaluation_mode": evaluation_mode,
    }
    history.setdefault("clip_norm", [])
    history.setdefault("dp_clip_schedule", privacy_settings.get("clip_schedule"))
    history.setdefault("dp_clip_decay", privacy_settings.get("clip_decay"))
    history.setdefault("dp_clip_min", privacy_settings.get("clip_min"))

    banner = (
        f"{algorithm.upper()} | dataset={config.dataset} | clients={config.num_clients} | "
        f"selected={len(selection_schedule[0])} | device={runtime.device}"
    )
    if start_round > 0:
        banner += f" | resume_from_round={start_round + 1}"
    print(banner)

    local_trainer = get_local_trainer(algorithm)
    for round_idx in range(start_round, config.rounds):
        lr = round_learning_rate(round_idx, config)
        selected_clients = selection_schedule[round_idx]
        set_round_loader_seed(client_loaders, config.seed, round_idx, selected_clients)
        avg_delta = zero_update_like(global_state)
        local_losses: list[float] = []
        round_clip_norm = resolve_round_clip_norm(round_idx, privacy_settings) if privacy_settings["enabled"] else None
        reference_state = (
            apply_update(global_state, global_momentum, alpha=config.gamma)
            if algorithm == "fednsam"
            else global_state
        )
        if privacy_settings["enabled"]:
            total_client_examples = sum(len(client_loaders[client_id].dataset) for client_id in selected_clients)
        else:
            total_client_examples = len(selected_clients)

        for client_id in selected_clients:
            copy_state_into_model(work_model, reference_state)
            local_loss = local_trainer(work_model, client_loaders[client_id], lr, config, runtime)
            update = state_delta(reference_state, work_model.state_dict())
            if privacy_settings["enabled"]:
                clip_tensor_updates_(update, float(round_clip_norm))
                noise_generator = build_client_noise_generator(
                    config.seed,
                    algorithm,
                    round_idx,
                    client_id,
                    runtime.device,
                )
                add_tensorwise_gaussian_noise_(
                    update,
                    clip_norm=float(round_clip_norm),
                    noise_multiplier=float(privacy_settings["noise_multiplier"]),
                    client_count=len(selected_clients),
                    generator=noise_generator,
                )
                scale = len(client_loaders[client_id].dataset) / max(total_client_examples, 1)
            else:
                scale = 1.0 / len(selected_clients)
            accumulate_update_(avg_delta, update, scale)
            local_losses.append(local_loss)

        if algorithm == "fednsam":
            update_global_momentum_(global_momentum, avg_delta, config.gamma)
            add_update_(global_state, global_momentum)
        else:
            add_update_(global_state, avg_delta)

        current_round = round_idx + 1
        did_eval = current_round in eval_rounds
        if did_eval:
            copy_state_into_model(work_model, global_state)
            if evaluation_mode == "client_average":
                if client_test_loaders is None:
                    raise ValueError("Client-average evaluation requires per-client test loaders.")
                accuracy, test_loss = evaluate_client_average(work_model, client_test_loaders, runtime)
            else:
                if test_loader is None:
                    raise ValueError("Global evaluation requires a test loader.")
                accuracy, test_loss = evaluate(work_model, test_loader, runtime)
            history["round"].append(current_round)
            history["accuracy"].append(accuracy)
            history["loss"].append(test_loss)
            if privacy_settings["enabled"]:
                history["clip_norm"].append(float(round_clip_norm))
                history["epsilon"].append(epsilon_trace[current_round])
            print(
                f"{algorithm.upper()} round={current_round:03d} | lr={lr:.5f} | "
                f"local_loss={np.mean(local_losses):.4f} | test_loss={test_loss:.4f} | "
                f"test_acc={accuracy * 100:.2f}%"
                + (
                    f" | clip={history['clip_norm'][-1]:.4f}"
                    if privacy_settings["enabled"] and history["clip_norm"]
                    else ""
                )
                + (
                    f" | eps={history['epsilon'][-1]:.4f}"
                    if privacy_settings["enabled"] and history["epsilon"]
                    else ""
                )
                + f" | clients={format_client_selection(selected_clients)}"
            )

        if round_callback is not None:
            round_callback(
                {
                    "algorithm": algorithm,
                    "round": current_round,
                    "history": history,
                    "global_state": global_state,
                    "global_momentum": global_momentum,
                    "did_eval": did_eval,
                }
            )

    return history


def build_json_snapshot(
    completed_histories: dict[str, ExperimentHistory],
    current_algorithm: str | None = None,
    current_history: ExperimentHistory | None = None,
) -> dict[str, ExperimentHistory]:
    snapshot = copy_histories(completed_histories)
    if current_algorithm is not None and current_history is not None:
        snapshot[current_algorithm] = copy_history(current_history)
    return snapshot


def summarize_histories(histories: dict[str, ExperimentHistory], algorithms: list[str]) -> None:
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


def order_histories(
    histories: dict[str, ExperimentHistory],
    algorithms: list[str],
) -> dict[str, ExperimentHistory]:
    return {algorithm: copy_history(histories[algorithm]) for algorithm in algorithms if algorithm in histories}


def build_ordered_json_snapshot(
    algorithms: list[str],
    completed_histories: dict[str, ExperimentHistory],
    active_histories: dict[str, ExperimentHistory] | None = None,
) -> dict[str, ExperimentHistory]:
    snapshot: dict[str, ExperimentHistory] = {}
    for algorithm in algorithms:
        if active_histories is not None and algorithm in active_histories:
            snapshot[algorithm] = copy_history(active_histories[algorithm])
        elif algorithm in completed_histories:
            snapshot[algorithm] = copy_history(completed_histories[algorithm])
    return snapshot


def parallel_compare_disable_reason(
    config: FedNSAMConfig,
    algorithms: list[str],
) -> str | None:
    if not config.compare_parallel:
        return "parallel compare not requested"
    if config.resume is not None:
        return "resume mode uses the sequential checkpoint path"
    if config.ckpt_dir is not None:
        return "--ckpt-dir currently supports only sequential compare"
    if len(algorithms) < 2:
        return "fewer than two algorithms were requested"
    if len(resolve_compare_devices(config)) < 2:
        return "fewer than two devices were provided"
    return None


def _parallel_compare_worker(
    result_queue: mp.Queue,
    config: FedNSAMConfig,
    algorithm: str,
    device_name: str,
    shared_context: dict[str, object],
    dataset_payload: tuple[object, object, int] | None,
) -> None:
    try:
        configure_parallel_worker_threads(config)
        worker_config = copy.deepcopy(config)
        worker_config.device = device_name
        worker_config.devices = ()
        worker_config.compare_parallel = False
        worker_config.resume = None
        worker_config.ckpt_dir = None
        worker_config.save_json = None

        runtime = build_runtime_config(worker_config)
        normalized_context = normalize_shared_context(shared_context)
        if dataset_payload is None:
            train_dataset, test_dataset, _ = build_cifar_datasets(worker_config)
        else:
            train_dataset, test_dataset, _ = dataset_payload
        client_loaders = build_client_loaders(
            train_dataset,
            list(normalized_context["partitions"]),
            worker_config,
            runtime.device,
        )
        evaluation_mode = str(normalized_context.get("evaluation_mode", "global"))
        if evaluation_mode == "client_average":
            client_test_loaders = build_client_eval_loaders(
                test_dataset,
                list(normalized_context["test_partitions"]),
                worker_config,
                runtime.device,
            )
            test_loader = None
        else:
            client_test_loaders = None
            test_loader = build_test_loader(test_dataset, worker_config, runtime.device)
        privacy_settings = dict(normalized_context["privacy_settings"])
        eval_rounds = set(int(round_id) for round_id in normalized_context["eval_rounds"])
        epsilon_trace = {
            int(key): float(value) for key, value in dict(normalized_context["epsilon_trace"]).items()
        }
        initial_state = clone_state_dict(normalized_context["initial_state"])
        num_classes = int(normalized_context["num_classes"])

        def emit_progress(event: dict[str, object]) -> None:
            if event["did_eval"]:
                result_queue.put(
                    {
                        "type": "progress",
                        "algorithm": algorithm,
                        "round": int(event["round"]),
                        "did_eval": True,
                        "history": copy_history(event["history"]),
                    }
                )

        history = run_single_experiment(
            config=worker_config,
            algorithm=algorithm,
            initial_state=initial_state,
            selection_schedule=list(normalized_context["selection_schedule"]),
            num_classes=num_classes,
            privacy_settings=privacy_settings,
            runtime=runtime,
            client_loaders=client_loaders,
            test_loader=test_loader,
            client_test_loaders=client_test_loaders,
            evaluation_mode=evaluation_mode,
            eval_rounds=eval_rounds,
            epsilon_trace=epsilon_trace,
            round_callback=emit_progress,
        )
        result_queue.put(
            {
                "type": "result",
                "algorithm": algorithm,
                "history": copy_history(history),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "algorithm": algorithm,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def terminate_parallel_workers(active_workers: dict[str, dict[str, object]]) -> None:
    for worker in active_workers.values():
        process = worker["process"]
        if process.is_alive():
            process.terminate()
    for worker in active_workers.values():
        process = worker["process"]
        process.join(timeout=5)


def compare_histories_parallel(
    config: FedNSAMConfig,
    algorithms: list[str],
    *,
    progress_hook: ProgressHook | None = None,
) -> dict[str, ExperimentHistory]:
    train_dataset, test_dataset, num_classes = build_cifar_datasets(config)
    shared_context = build_shared_context(config, train_dataset, test_dataset, num_classes)
    privacy_settings = dict(shared_context["privacy_settings"])
    print_privacy_banner(privacy_settings)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    pending_algorithms = list(algorithms)
    idle_devices = list(resolve_compare_devices(config))
    all_cpu_devices = all(resolve_device(device_name).type == "cpu" for device_name in idle_devices)
    dataset_payload = (train_dataset, test_dataset, num_classes) if all_cpu_devices else None
    if not all_cpu_devices:
        del train_dataset
        del test_dataset
    active_workers: dict[str, dict[str, object]] = {}
    completed_histories: dict[str, ExperimentHistory] = {}
    active_histories: dict[str, ExperimentHistory] = {}

    def launch_worker(algorithm: str, device_name: str) -> None:
        process = ctx.Process(
            target=_parallel_compare_worker,
            args=(
                result_queue,
                copy.deepcopy(config),
                algorithm,
                device_name,
                normalize_shared_context(shared_context),
                dataset_payload,
            ),
        )
        process.start()
        active_workers[algorithm] = {
            "process": process,
            "device": device_name,
        }

    try:
        while pending_algorithms and idle_devices:
            launch_worker(pending_algorithms.pop(0), idle_devices.pop(0))

        while active_workers:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed_worker = None
                for algorithm, worker in active_workers.items():
                    process = worker["process"]
                    if process.exitcode not in (None, 0):
                        failed_worker = (algorithm, process.exitcode)
                        break
                if failed_worker is not None:
                    terminate_parallel_workers(active_workers)
                    algorithm, exitcode = failed_worker
                    raise RuntimeError(
                        f"Parallel compare worker for {algorithm} exited unexpectedly with code {exitcode}."
                    )
                continue

            message_type = str(message["type"])
            algorithm = str(message["algorithm"])

            if message_type == "progress":
                history = copy_history(message["history"])
                active_histories[algorithm] = history
                if config.save_json:
                    save_histories(
                        build_ordered_json_snapshot(algorithms, completed_histories, active_histories),
                        config.save_json,
                    )
                if progress_hook is not None:
                    progress_hook(
                        {
                            "algorithm": algorithm,
                            "round": int(message["round"]),
                            "history": history,
                            "global_state": None,
                            "global_momentum": None,
                            "did_eval": True,
                        }
                    )
                continue

            if message_type == "error":
                terminate_parallel_workers(active_workers)
                raise RuntimeError(
                    f"Parallel compare worker for {algorithm} failed.\n{message['traceback']}"
                )

            if message_type != "result":
                terminate_parallel_workers(active_workers)
                raise RuntimeError(f"Unexpected parallel compare message type: {message_type}")

            history = copy_history(message["history"])
            completed_histories[algorithm] = history
            active_histories.pop(algorithm, None)
            worker = active_workers.pop(algorithm)
            process = worker["process"]
            process.join(timeout=5)
            idle_devices.append(str(worker["device"]))

            if config.save_json:
                save_histories(
                    build_ordered_json_snapshot(algorithms, completed_histories, active_histories),
                    config.save_json,
                )

            if pending_algorithms and idle_devices:
                launch_worker(pending_algorithms.pop(0), idle_devices.pop(0))
    except BaseException:
        terminate_parallel_workers(active_workers)
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()

    ordered_histories = order_histories(completed_histories, algorithms)
    summarize_histories(ordered_histories, algorithms)

    if config.save_json:
        save_histories(ordered_histories, config.save_json)
        print(f"Saved results to {config.save_json}")

    return ordered_histories


def compare_histories(
    config: FedNSAMConfig,
    algorithms: list[str] | None,
    *,
    progress_hook: ProgressHook | None = None,
) -> dict[str, ExperimentHistory]:
    checkpoint: dict[str, object] | None = None
    if config.resume is not None:
        checkpoint = load_checkpoint(config.resume)
        config, algorithms = apply_resume_config(config, checkpoint, algorithms)
    else:
        config = resolve_effective_config(config)
    resolved_algorithms = normalize_algorithms(algorithms, config.algorithm)

    disable_reason = parallel_compare_disable_reason(config, resolved_algorithms)
    if checkpoint is None and disable_reason is None:
        return compare_histories_parallel(config, resolved_algorithms, progress_hook=progress_hook)
    if config.compare_parallel and disable_reason is not None:
        print(f"Compare-parallel disabled: {disable_reason}. Falling back to sequential compare.")

    set_random_seed(config.seed)
    runtime = build_runtime_config(config)
    train_dataset, test_dataset, num_classes = build_cifar_datasets(config)

    if checkpoint is None:
        shared_context = build_shared_context(config, train_dataset, test_dataset, num_classes)
        histories: dict[str, ExperimentHistory] = {}
        current_algorithm_index = 0
        current_algorithm = None
        current_round = 0
        current_state = None
    else:
        shared_context = normalize_shared_context(dict(checkpoint["shared_context"]))
        histories = copy_histories(dict(checkpoint["histories"]))
        current_algorithm_index = int(checkpoint["current_algorithm_index"])
        current_algorithm = checkpoint["current_algorithm"]
        current_round = int(checkpoint["current_round"])
        current_state = checkpoint["current_state"]
        num_classes = int(shared_context["num_classes"])

    client_loaders = build_client_loaders(
        train_dataset,
        list(shared_context["partitions"]),
        config,
        runtime.device,
    )
    evaluation_mode = str(shared_context.get("evaluation_mode", "global"))
    if evaluation_mode == "client_average":
        client_test_loaders = build_client_eval_loaders(
            test_dataset,
            list(shared_context["test_partitions"]),
            config,
            runtime.device,
        )
        test_loader = None
    else:
        client_test_loaders = None
        test_loader = build_test_loader(test_dataset, config, runtime.device)
    selection_schedule = list(shared_context["selection_schedule"])
    privacy_settings = dict(shared_context["privacy_settings"])
    eval_rounds = set(int(round_id) for round_id in shared_context["eval_rounds"])
    epsilon_trace = {int(key): float(value) for key, value in dict(shared_context["epsilon_trace"]).items()}
    initial_state = clone_state_dict(shared_context["initial_state"])

    print_privacy_banner(privacy_settings)

    if config.save_json and histories:
        save_histories(build_json_snapshot(histories), config.save_json)

    for algorithm_index in range(current_algorithm_index, len(resolved_algorithms)):
        algorithm = resolved_algorithms[algorithm_index]
        if algorithm_index == current_algorithm_index and current_state is not None:
            state_algorithm = str(current_state["algorithm"])
            if state_algorithm != algorithm:
                raise ValueError("Checkpoint algorithm state does not match the requested resume position.")
            start_round = int(current_state["round"])
            resumed_history = copy_history(current_state["history"])
            resumed_global_state = clone_state_dict(current_state["global_state"])
            resumed_global_momentum = clone_state_dict(current_state["global_momentum"])
        else:
            start_round = 0
            resumed_history = None
            resumed_global_state = None
            resumed_global_momentum = None

        def persist_progress(event: dict[str, object]) -> None:
            current_history = copy_history(event["history"])
            checkpoint_state = {
                "algorithm": event["algorithm"],
                "round": event["round"],
                "history": current_history,
                "global_state": clone_state_dict(event["global_state"], device=torch.device("cpu")),
                "global_momentum": clone_state_dict(event["global_momentum"], device=torch.device("cpu")),
            }
            save_latest_checkpoint(
                config=config,
                algorithms=resolved_algorithms,
                histories=build_json_snapshot(histories),
                current_algorithm_index=algorithm_index,
                current_algorithm=str(event["algorithm"]),
                current_round=int(event["round"]),
                current_state=checkpoint_state,
                shared_context=shared_context,
            )
            if event["did_eval"] and config.save_json:
                snapshot = build_json_snapshot(histories, str(event["algorithm"]), current_history)
                save_histories(snapshot, config.save_json)
            if progress_hook is not None:
                progress_hook(event)

        histories[algorithm] = run_single_experiment(
            config=config,
            algorithm=algorithm,
            initial_state=initial_state,
            selection_schedule=selection_schedule,
            num_classes=num_classes,
            privacy_settings=privacy_settings,
            runtime=runtime,
            client_loaders=client_loaders,
            test_loader=test_loader,
            client_test_loaders=client_test_loaders,
            evaluation_mode=evaluation_mode,
            eval_rounds=eval_rounds,
            epsilon_trace=epsilon_trace,
            start_round=start_round,
            initial_global_state=resumed_global_state,
            initial_global_momentum=resumed_global_momentum,
            initial_history=resumed_history,
            round_callback=persist_progress,
        )

        current_state = None
        current_round = config.rounds
        save_latest_checkpoint(
            config=config,
            algorithms=resolved_algorithms,
            histories=histories,
            current_algorithm_index=algorithm_index + 1,
            current_algorithm=None,
            current_round=current_round,
            current_state=None,
            shared_context=shared_context,
        )
        if config.save_json:
            save_histories(build_json_snapshot(histories), config.save_json)

    ordered_histories = order_histories(histories, resolved_algorithms)
    summarize_histories(ordered_histories, resolved_algorithms)

    if config.save_json:
        save_histories(ordered_histories, config.save_json)
        print(f"Saved results to {config.save_json}")

    return ordered_histories


def run_fednsam(config: FedNSAMConfig) -> ExperimentHistory:
    history = compare_histories(config, [normalize_algorithm_name(config.algorithm)])
    return history[normalize_algorithm_name(config.algorithm)]
