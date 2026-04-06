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
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()
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
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
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
