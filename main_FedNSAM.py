import argparse

from fednsam import FedNSAMConfig, compare_histories, normalize_algorithm_name, run_fednsam


def parse_args() -> tuple[FedNSAMConfig, list[str] | None]:
    parser = argparse.ArgumentParser(description="Minimal FedNSAM training entrypoint.")
    parser.add_argument("--algorithm", default="fednsam", help="fedavg, fedsam, or fednsam")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple algorithms fairly, e.g. --compare fedavg fedsam fednsam",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--client-fraction", type=float, default=0.1)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--local-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.85)
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
        "--fast-cuda",
        action="store_true",
        help="Enable non-deterministic CUDA speed features such as TF32 and cuDNN benchmark.",
    )
    parser.add_argument(
        "--amp",
        choices=["off", "auto", "fp16", "bf16"],
        default="off",
        help="Automatic mixed precision mode: off, auto, fp16, or bf16.",
    )
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    dp_args_provided = any(
        value is not None
        for value in (args.dp_clip_norm, args.dp_noise_multiplier, args.dp_target_epsilon, args.dp_delta)
    )
    if args.dp:
        if args.dp_clip_norm is None:
            parser.error("--dp requires --dp-clip.")
        if args.dp_noise_multiplier is None and args.dp_target_epsilon is None:
            parser.error("--dp requires exactly one of --sigma or --eps.")
    elif dp_args_provided:
        parser.error("DP options require --dp.")

    if args.dp_clip_norm is not None and args.dp_clip_norm <= 0:
        parser.error("--dp-clip must be positive.")
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
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        rho=args.rho,
        gamma=args.gamma,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        dp=args.dp,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_delta=args.dp_delta,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        fast_cuda=args.fast_cuda,
        amp=args.amp,
        save_json=args.save_json,
    )
    compare = None if args.compare is None else [normalize_algorithm_name(name) for name in args.compare]
    return config, compare


if __name__ == "__main__":
    config, compare = parse_args()
    if compare:
        compare_histories(config, compare)
    else:
        run_fednsam(config)
