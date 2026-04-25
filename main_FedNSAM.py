import argparse
import sys
from pathlib import Path

from fednsam import (
    FedNSAMConfig,
    compare_histories,
    configure_run_logging,
    finalize_run_logging,
    format_command_for_log,
    normalize_algorithm_name,
    run_fednsam,
)


def parse_args() -> tuple[FedNSAMConfig, list[str] | None]:
    parser = argparse.ArgumentParser(description="Minimal FedNSAM training entrypoint.")
    parser.add_argument("--algorithm", default="fednsam", help="fedavg, fedsam, or fednsam")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple algorithms fairly, e.g. --compare fedavg fedsam fednsam",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "emnist"], default="cifar100")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--client-fraction", type=float, default=0.1)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--local-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--lr-schedule",
        choices=["auto", "cosine", "exp"],
        default="auto",
        help="Learning-rate schedule. 'auto' keeps the current default: cosine for non-DP, exponential for DP.",
    )
    parser.add_argument("--lr-decay", type=float, default=1.0, help="Per-round exponential decay factor used by DP-FedSAM-style local training.")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0, help="Local SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument(
        "--rho-mode",
        choices=["fixed", "dp_algorithm"],
        default="fixed",
        help="Optional DP SAM rho/adaptive preset by algorithm.",
    )
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument(
        "--gamma-strategy",
        choices=["fixed", "cosine_gate"],
        default="fixed",
        help="Server momentum gamma strategy for FedNSAM.",
    )
    parser.add_argument("--gamma-min", type=float, default=0.0, help="Lower bound for cosine-gated gamma.")
    parser.add_argument(
        "--gamma-zero-round",
        type=int,
        default=None,
        help="Set gamma to 0 starting from this round (inclusive).",
    )
    parser.add_argument(
        "--gamma-zero-lr-multiplier",
        "--restart-factor",
        dest="gamma_zero_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning-rate restart factor applied on --gamma-zero-round.",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet non-IID coefficient.")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--dp", action="store_true", help="Enable client-level differential privacy.")
    parser.add_argument(
        "--dp-clip",
        dest="dp_clip_norm",
        type=float,
        default=None,
        help="client-level DP clip norm / clipping threshold C",
    )
    parser.add_argument(
        "--dp-clip-decay",
        dest="dp_clip_decay",
        type=float,
        default=1.0,
        help="per-round exponential decay factor for the DP clip threshold",
    )
    parser.add_argument(
        "--dp-clip-min",
        dest="dp_clip_min",
        type=float,
        default=None,
        help="minimum DP clip threshold reached by the decay schedule",
    )
    dp_group = parser.add_mutually_exclusive_group()
    dp_group.add_argument(
        "--sigma",
        dest="dp_noise_multiplier",
        type=float,
        default=None,
        help="client-level DP noise multiplier",
    )
    dp_group.add_argument(
        "--eps",
        dest="dp_target_epsilon",
        type=float,
        default=None,
        help="target privacy budget epsilon; resolves sigma before training",
    )
    parser.add_argument(
        "--delta",
        dest="dp_delta",
        type=float,
        default=None,
        help="target delta for RDP accountant",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional device list for compare-parallel mode, e.g. --devices cuda:0 cuda:1",
    )
    parser.add_argument(
        "--compare-parallel",
        action="store_true",
        help="Run compared algorithms in parallel across multiple devices.",
    )
    parser.add_argument(
        "--fast-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable non-deterministic CUDA speed features such as TF32 and cuDNN benchmark.",
    )
    parser.add_argument(
        "--amp",
        choices=["off", "auto", "fp16", "bf16"],
        default="off",
        help="Automatic mixed precision mode: off, auto, fp16, or bf16.",
    )
    parser.add_argument("--ckpt-dir", default=None, help="Save rolling round-level checkpoints to this directory.")
    parser.add_argument("--resume", default=None, help="Resume training from a saved checkpoint file.")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    argv = sys.argv[1:]
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    explicit_fields: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", maxsplit=1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_fields.add(dest)

    dp_args_provided = any(
        value is not None
        for value in (
            args.dp_clip_norm,
            args.dp_noise_multiplier,
            args.dp_target_epsilon,
            args.dp_delta,
            args.dp_clip_min,
        )
    )
    if args.dp:
        if args.dp_clip_norm is None:
            parser.error("--dp requires --dp-clip.")
        if args.dp_noise_multiplier is None and args.dp_target_epsilon is None:
            parser.error("--dp requires exactly one of --sigma or --eps.")
    elif dp_args_provided or args.dp_clip_decay != 1.0:
        parser.error("DP options require --dp.")

    if args.dp_clip_norm is not None and args.dp_clip_norm <= 0:
        parser.error("--dp-clip must be positive.")
    if args.gamma_min < 0:
        parser.error("--gamma-min must be non-negative.")
    if args.gamma_zero_round is not None and args.gamma_zero_round <= 0:
        parser.error("--gamma-zero-round must be positive.")
    if args.gamma_zero_lr_multiplier <= 0:
        parser.error("--gamma-zero-lr-multiplier must be positive.")
    if args.gamma_zero_round is None and args.gamma_zero_lr_multiplier != 1.0:
        parser.error("--gamma-zero-lr-multiplier requires --gamma-zero-round.")
    if args.dp_clip_decay <= 0:
        parser.error("--dp-clip-decay must be positive.")
    if args.dp_clip_min is not None and args.dp_clip_min <= 0:
        parser.error("--dp-clip-min must be positive.")
    if args.dp_clip_norm is not None and args.dp_clip_min is not None and args.dp_clip_min > args.dp_clip_norm:
        parser.error("--dp-clip-min cannot exceed --dp-clip.")
    if args.dp_noise_multiplier is not None and args.dp_noise_multiplier <= 0:
        parser.error("--sigma must be positive.")
    if args.dp_target_epsilon is not None and args.dp_target_epsilon <= 0:
        parser.error("--eps must be positive.")
    if args.dp_delta is not None and not 0 < args.dp_delta < 1:
        parser.error("--delta must be between 0 and 1.")

    config = FedNSAMConfig(
        algorithm=normalize_algorithm_name(args.algorithm),
        dataset=args.dataset,
        data_dir=args.data_dir,
        rounds=args.rounds,
        num_clients=args.num_clients,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        rho=args.rho,
        rho_mode=args.rho_mode,
        gamma=args.gamma,
        gamma_strategy=args.gamma_strategy,
        gamma_min=args.gamma_min,
        gamma_zero_round=args.gamma_zero_round,
        gamma_zero_lr_multiplier=args.gamma_zero_lr_multiplier,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        dp=args.dp,
        dp_clip_norm=args.dp_clip_norm,
        dp_clip_decay=args.dp_clip_decay,
        dp_clip_min=args.dp_clip_min,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_delta=args.dp_delta,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        devices=() if args.devices is None else tuple(args.devices),
        compare_parallel=args.compare_parallel,
        fast_cuda=args.fast_cuda,
        amp=args.amp,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
        explicit_cli_fields=tuple(sorted(field for field in explicit_fields if field != "compare")),
        explicit_compare="compare" in explicit_fields,
        save_json=args.save_json,
    )
    compare = None if args.compare is None else [normalize_algorithm_name(name) for name in args.compare]
    return config, compare


if __name__ == "__main__":
    config, compare = parse_args()
    command = format_command_for_log(Path(sys.argv[0]).name, sys.argv[1:])
    configure_run_logging(config, command)
    try:
        if compare or config.resume:
            compare_histories(config, compare)
        else:
            run_fednsam(config)
    except BaseException:
        finalize_run_logging("failed")
        raise
    else:
        finalize_run_logging("completed")
