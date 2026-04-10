import copy
import json
import math
import os
import random
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dirichlet_data import build_dirichlet_partitions
from models import resnet18_cifar
from privacy import add_gaussian_noise_, clip_model_update_, compute_epsilon, solve_noise_multiplier
from sam import SAM

ExperimentHistory = dict[str, object]
ProgressHook = Callable[[dict[str, object]], None]
RUNTIME_OVERRIDE_FIELDS = {"device", "fast_cuda", "amp", "save_json", "ckpt_dir"}


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
        momentum=0.0,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime.use_grad_scaler else None

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
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
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
    runtime: RuntimeConfig,
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
    scaler = torch.amp.GradScaler("cuda", enabled=True) if runtime.use_grad_scaler else None

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
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.restore_step(zero_grad=False)
                scaler.step(optimizer.base_optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                sharpness_loss.backward()
                if not gradients_are_finite(model):
                    optimizer.restore_step(zero_grad=True)
                    continue
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
    num_classes: int,
) -> dict[str, object]:
    targets = np.asarray(train_dataset.targets)
    client_sizes = build_equal_client_sizes(len(train_dataset), config.num_clients)
    partitions = build_dirichlet_partitions(
        targets=targets,
        num_clients=config.num_clients,
        alpha=config.alpha,
        seed=config.seed,
        client_sizes=client_sizes,
    )
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
    return normalized


def print_privacy_banner(privacy_settings: dict[str, float | str | bool | None]) -> None:
    if not privacy_settings["enabled"]:
        return
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


def run_single_experiment(
    config: FedNSAMConfig,
    algorithm: str,
    initial_state: OrderedDict[str, torch.Tensor],
    selection_schedule: list[list[int]],
    num_classes: int,
    privacy_settings: dict[str, float | str | bool | None],
    runtime: RuntimeConfig,
    client_loaders: list[DataLoader],
    test_loader: DataLoader,
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
        "dp_noise_multiplier": privacy_settings["noise_multiplier"],
        "dp_target_epsilon": privacy_settings["target_epsilon"],
        "dp_delta": privacy_settings["delta"],
        "dp_sample_rate": privacy_settings["sample_rate"],
        "dp_source": privacy_settings["source"],
    }

    banner = (
        f"{algorithm.upper()} | dataset={config.dataset} | clients={config.num_clients} | "
        f"selected={len(selection_schedule[0])} | device={runtime.device}"
    )
    if start_round > 0:
        banner += f" | resume_from_round={start_round + 1}"
    print(banner)

    local_trainer = get_local_trainer(algorithm)
    for round_idx in range(start_round, config.rounds):
        lr = cosine_lr(round_idx, config)
        selected_clients = selection_schedule[round_idx]
        set_round_loader_seed(client_loaders, config.seed, round_idx, selected_clients)
        avg_delta = zero_update_like(global_state)
        local_losses: list[float] = []
        reference_state = (
            apply_update(global_state, global_momentum, alpha=config.gamma)
            if algorithm == "fednsam"
            else global_state
        )
        scale = 1.0 / len(selected_clients)

        for client_id in selected_clients:
            copy_state_into_model(work_model, reference_state)
            local_loss = local_trainer(work_model, client_loaders[client_id], lr, config, runtime)
            update = state_delta(reference_state, work_model.state_dict())
            if privacy_settings["enabled"]:
                clip_model_update_(update, float(privacy_settings["clip_norm"]))
            accumulate_update_(avg_delta, update, scale)
            local_losses.append(local_loss)

        if privacy_settings["enabled"]:
            noise_std = (
                float(privacy_settings["noise_multiplier"]) * float(privacy_settings["clip_norm"]) / len(selected_clients)
            )
            noise_generator = build_round_noise_generator(config.seed, algorithm, round_idx, runtime.device)
            add_gaussian_noise_(avg_delta, noise_std, generator=noise_generator)

        if algorithm == "fednsam":
            update_global_momentum_(global_momentum, avg_delta, config.gamma)
            add_update_(global_state, global_momentum)
        else:
            add_update_(global_state, avg_delta)

        current_round = round_idx + 1
        did_eval = current_round in eval_rounds
        if did_eval:
            copy_state_into_model(work_model, global_state)
            accuracy, test_loss = evaluate(work_model, test_loader, runtime)
            history["round"].append(current_round)
            history["accuracy"].append(accuracy)
            history["loss"].append(test_loss)
            if privacy_settings["enabled"]:
                history["epsilon"].append(epsilon_trace[current_round])
            print(
                f"{algorithm.upper()} round={current_round:03d} | lr={lr:.5f} | "
                f"local_loss={np.mean(local_losses):.4f} | test_loss={test_loss:.4f} | "
                f"test_acc={accuracy * 100:.2f}%"
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
    resolved_algorithms = normalize_algorithms(algorithms, config.algorithm)

    set_random_seed(config.seed)
    runtime = build_runtime_config(config)
    train_dataset, test_dataset, num_classes = build_cifar_datasets(config)

    if checkpoint is None:
        shared_context = build_shared_context(config, train_dataset, num_classes)
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

    summarize_histories(histories, resolved_algorithms)

    if config.save_json:
        save_histories(histories, config.save_json)
        print(f"Saved results to {config.save_json}")

    return histories


def run_fednsam(config: FedNSAMConfig) -> ExperimentHistory:
    history = compare_histories(config, [normalize_algorithm_name(config.algorithm)])
    return history[normalize_algorithm_name(config.algorithm)]
