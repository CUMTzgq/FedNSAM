import math
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
from scipy import special


def clone_update(update: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in update.items())


def update_l2_norm(update: OrderedDict[str, torch.Tensor]) -> float:
    total_sq = 0.0
    for tensor in update.values():
        if torch.is_floating_point(tensor):
            total_sq += tensor.pow(2).sum().item()
    return math.sqrt(total_sq)


def clip_model_update(
    update: OrderedDict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[OrderedDict[str, torch.Tensor], float, float]:
    clipped = clone_update(update)
    total_norm, scale = clip_model_update_(clipped, clip_norm)
    return clipped, total_norm, scale


def clip_model_update_(
    update: OrderedDict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[float, float]:
    total_norm = update_l2_norm(update)
    if total_norm == 0.0 or total_norm <= clip_norm:
        return total_norm, 1.0

    scale = clip_norm / (total_norm + 1e-12)
    for tensor in update.values():
        if torch.is_floating_point(tensor):
            tensor.mul_(scale)
    return total_norm, scale


def per_tensor_l2_norms(update: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, float]:
    norms: OrderedDict[str, float] = OrderedDict()
    for name, tensor in update.items():
        if torch.is_floating_point(tensor):
            norms[name] = float(tensor.norm(2).item())
    return norms


def clip_tensor_updates(
    update: OrderedDict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, float], OrderedDict[str, float]]:
    clipped = clone_update(update)
    norms, scales = clip_tensor_updates_(clipped, clip_norm)
    return clipped, norms, scales


def clip_tensor_updates_(
    update: OrderedDict[str, torch.Tensor],
    clip_norm: float,
) -> tuple[OrderedDict[str, float], OrderedDict[str, float]]:
    norms: OrderedDict[str, float] = OrderedDict()
    scales: OrderedDict[str, float] = OrderedDict()
    for name, tensor in update.items():
        if not torch.is_floating_point(tensor):
            continue
        tensor_norm = float(tensor.norm(2).item())
        norms[name] = tensor_norm
        if tensor_norm == 0.0 or tensor_norm <= clip_norm:
            scales[name] = 1.0
            continue
        scale = clip_norm / (tensor_norm + 1e-12)
        tensor.mul_(scale)
        scales[name] = scale
    return norms, scales


def add_gaussian_noise_(
    update: OrderedDict[str, torch.Tensor],
    noise_std: float,
    generator: torch.Generator | None = None,
) -> OrderedDict[str, torch.Tensor]:
    if noise_std <= 0:
        return update

    for tensor in update.values():
        if torch.is_floating_point(tensor):
            tensor.add_(torch.randn_like(tensor, generator=generator), alpha=noise_std)
    return update


def add_gaussian_noise(
    update: OrderedDict[str, torch.Tensor],
    noise_std: float,
    generator: torch.Generator | None = None,
) -> OrderedDict[str, torch.Tensor]:
    noised = clone_update(update)
    return add_gaussian_noise_(noised, noise_std, generator=generator)


def add_tensorwise_gaussian_noise_(
    update: OrderedDict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    client_count: int,
    generator: torch.Generator | None = None,
) -> OrderedDict[str, torch.Tensor]:
    if noise_multiplier <= 0 or client_count <= 0:
        return update

    noise_std = noise_multiplier * clip_norm / math.sqrt(client_count)
    return add_gaussian_noise_(update, noise_std=noise_std, generator=generator)


def _log_add(logx: float, logy: float) -> float:
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf
    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    return math.log(2) + special.log_ndtr(-x * 2**0.5)


def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.log(special.binom(alpha, i))
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * i - i) / (2 * (sigma**2))
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0
    z0 = sigma**2 * math.log(1 / q - 1) + 0.5

    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
    if q == 0:
        return 0.0
    if sigma == 0:
        return np.inf
    if q == 1.0:
        return alpha / (2 * sigma**2)
    if np.isinf(alpha):
        return np.inf
    return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: Sequence[float],
) -> np.ndarray:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order) for order in orders], dtype=float)
    return rdp * steps


def get_privacy_spent(
    orders: Sequence[float],
    rdp: Sequence[float],
    delta: float,
) -> tuple[float, float]:
    orders_vec = np.atleast_1d(np.asarray(orders, dtype=float))
    rdp_vec = np.atleast_1d(np.asarray(rdp, dtype=float))
    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            "Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = int(np.nanargmin(eps))
    return float(max(eps[idx_opt], 0.0)), float(orders_vec[idx_opt])


def default_rdp_orders() -> np.ndarray:
    return np.arange(1.01, 100.0, 0.05)


def compute_epsilon_alpha(
    noise_multiplier: float,
    num_steps: int,
    q: float,
    delta: float,
) -> tuple[float, float, float]:
    alpha_list = default_rdp_orders()
    rdp_list = compute_rdp(
        q=q,
        noise_multiplier=noise_multiplier,
        steps=num_steps,
        orders=alpha_list,
    )
    epsilon, alpha = get_privacy_spent(orders=alpha_list, rdp=rdp_list, delta=delta)
    temp_rdp = float(rdp_list[list(alpha_list).index(alpha)])
    return float(epsilon), float(alpha), temp_rdp


def compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: Sequence[float] | None = None,
) -> tuple[float, float]:
    if steps <= 0:
        return 0.0, float("nan")
    if noise_multiplier <= 0:
        raise ValueError("Noise multiplier must be positive.")

    if orders is not None:
        rdp_orders = np.asarray(orders, dtype=float)
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=noise_multiplier,
            steps=steps,
            orders=rdp_orders,
        )
        return get_privacy_spent(orders=rdp_orders, rdp=rdp, delta=delta)

    epsilon, alpha, _ = compute_epsilon_alpha(
        noise_multiplier=noise_multiplier,
        num_steps=steps,
        q=sample_rate,
        delta=delta,
    )
    return epsilon, alpha


def solve_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    tolerance: float = 1e-4,
    max_iterations: int = 80,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("Target epsilon must be positive.")

    lower = 0.0
    upper = 1.0
    epsilon, _ = compute_epsilon(
        noise_multiplier=upper,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )
    while epsilon > target_epsilon:
        lower = upper
        upper *= 2.0
        if upper > 1e6:
            raise ValueError("Could not find a feasible noise multiplier for the requested epsilon.")
        epsilon, _ = compute_epsilon(
            noise_multiplier=upper,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )

    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        epsilon, _ = compute_epsilon(
            noise_multiplier=middle,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        if epsilon <= target_epsilon:
            upper = middle
        else:
            lower = middle
        if upper - lower <= tolerance * max(upper, 1.0):
            break

    return upper
