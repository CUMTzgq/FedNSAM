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
    total_norm = update_l2_norm(clipped)
    if total_norm == 0.0 or total_norm <= clip_norm:
        return clipped, total_norm, 1.0

    scale = clip_norm / (total_norm + 1e-12)
    for name, tensor in clipped.items():
        if torch.is_floating_point(tensor):
            clipped[name] = tensor * scale
    return clipped, total_norm, scale


def add_gaussian_noise(
    update: OrderedDict[str, torch.Tensor],
    noise_std: float,
) -> OrderedDict[str, torch.Tensor]:
    noised = clone_update(update)
    if noise_std <= 0:
        return noised

    for name, tensor in noised.items():
        if torch.is_floating_point(tensor):
            noised[name] = tensor + torch.randn_like(tensor) * noise_std
    return noised


def _log_add(logx: float, logy: float) -> float:
    lower, upper = min(logx, logy), max(logx, logy)
    if lower == -np.inf:
        return upper
    return math.log1p(math.exp(lower - upper)) + upper


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
    return math.log(2) + special.log_ndtr(-x * math.sqrt(2))


def _compute_log_a_for_int_alpha(sample_rate: float, noise_multiplier: float, alpha: int) -> float:
    log_a = -np.inf
    for index in range(alpha + 1):
        log_coef = (
            math.log(special.binom(alpha, index))
            + index * math.log(sample_rate)
            + (alpha - index) * math.log(1 - sample_rate)
        )
        log_term = log_coef + (index * index - index) / (2 * (noise_multiplier**2))
        log_a = _log_add(log_a, log_term)
    return float(log_a)


def _compute_log_a_for_frac_alpha(sample_rate: float, noise_multiplier: float, alpha: float) -> float:
    log_a0, log_a1 = -np.inf, -np.inf
    boundary = noise_multiplier**2 * math.log(1 / sample_rate - 1) + 0.5
    index = 0

    while True:
        coef = special.binom(alpha, index)
        log_coef = math.log(abs(coef))
        complement = alpha - index

        log_t0 = log_coef + index * math.log(sample_rate) + complement * math.log(1 - sample_rate)
        log_t1 = log_coef + complement * math.log(sample_rate) + index * math.log(1 - sample_rate)
        log_e0 = math.log(0.5) + _log_erfc((index - boundary) / (math.sqrt(2) * noise_multiplier))
        log_e1 = math.log(0.5) + _log_erfc((boundary - complement) / (math.sqrt(2) * noise_multiplier))
        log_s0 = log_t0 + (index * index - index) / (2 * (noise_multiplier**2)) + log_e0
        log_s1 = log_t1 + (complement * complement - complement) / (2 * (noise_multiplier**2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        index += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _compute_log_a(sample_rate: float, noise_multiplier: float, alpha: float) -> float:
    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(sample_rate, noise_multiplier, int(alpha))
    return _compute_log_a_for_frac_alpha(sample_rate, noise_multiplier, alpha)


def _compute_rdp(sample_rate: float, noise_multiplier: float, alpha: float) -> float:
    if sample_rate == 0:
        return 0.0
    if noise_multiplier == 0:
        return np.inf
    if sample_rate == 1.0:
        return alpha / (2 * noise_multiplier**2)
    if np.isinf(alpha):
        return np.inf
    return _compute_log_a(sample_rate, noise_multiplier, alpha) / (alpha - 1)


def compute_rdp(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    orders: Sequence[float],
) -> np.ndarray:
    rdp = np.array([_compute_rdp(sample_rate, noise_multiplier, order) for order in orders], dtype=float)
    return rdp * steps


def get_privacy_spent(
    orders: Sequence[float],
    rdp: Sequence[float],
    delta: float,
) -> tuple[float, float]:
    orders_arr = np.asarray(orders, dtype=float)
    rdp_arr = np.asarray(rdp, dtype=float)
    eps = (
        rdp_arr
        - (np.log(delta) + np.log(orders_arr)) / (orders_arr - 1)
        + np.log((orders_arr - 1) / orders_arr)
    )
    if np.isnan(eps).all():
        return np.inf, np.nan

    optimal_index = np.nanargmin(eps)
    epsilon = float(max(eps[optimal_index], 0.0))
    return epsilon, float(orders_arr[optimal_index])


def default_rdp_orders() -> np.ndarray:
    return np.arange(1.01, 100.0, 0.05)


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

    rdp_orders = default_rdp_orders() if orders is None else np.asarray(orders, dtype=float)
    rdp = compute_rdp(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=rdp_orders,
    )
    return get_privacy_spent(rdp=rdp, orders=rdp_orders, delta=delta)


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
