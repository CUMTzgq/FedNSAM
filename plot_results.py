import argparse
import json
import os
from pathlib import Path


COLOR_MAP = {
    "fedavg": "#1f77b4",
    "fedsam": "#ff7f0e",
    "fednsam": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy curves from saved FedNSAM result JSON files.")
    parser.add_argument("inputs", nargs="+", help="One or more result JSON files.")
    parser.add_argument("--output-dir", default="figures", help="Directory to save generated plots.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    parser.add_argument("--title", default=None, help="Optional custom title for single-file plots.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    return parser.parse_args()


def load_result_file(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} does not contain a non-empty result mapping.")

    normalized: dict[str, dict[str, object]] = {}
    for algorithm, payload in data.items():
        if str(algorithm).startswith("_"):
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"{path} -> {algorithm} is not a JSON object.")

        if "round" not in payload:
            raise ValueError(f"{path} -> {algorithm} is missing 'round'.")
        if "accuracy" not in payload:
            raise ValueError(f"{path} -> {algorithm} is missing 'accuracy'.")

        rounds = payload["round"]
        accuracy = payload["accuracy"]
        if not isinstance(rounds, list) or not isinstance(accuracy, list):
            raise ValueError(f"{path} -> {algorithm} requires list values for 'round' and 'accuracy'.")
        if len(rounds) != len(accuracy):
            raise ValueError(f"{path} -> {algorithm} has mismatched 'round' and 'accuracy' lengths.")

        normalized[str(algorithm)] = {
            "round": [int(value) for value in rounds],
            "accuracy": [float(value) * 100.0 for value in accuracy],
        }

    if not normalized:
        raise ValueError(f"{path} does not contain any algorithm result entries.")

    return normalized


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_accuracy.png"


def default_title(input_path: Path) -> str:
    return input_path.stem.replace("_", " ")


def plot_accuracy_curves(
    input_path: Path,
    results: dict[str, dict[str, object]],
    output_dir: Path,
    *,
    show: bool,
    title: str | None,
    dpi: int,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fednsam-mplconfig")
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for algorithm, payload in results.items():
        color = COLOR_MAP.get(algorithm.lower())
        axis.plot(
            payload["round"],
            payload["accuracy"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=algorithm,
            color=color,
        )

    axis.set_xlabel("Round")
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_title(title if title is not None else default_title(input_path))
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend()
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(input_path, output_dir)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for raw_path in args.inputs:
        input_path = Path(raw_path)
        results = load_result_file(input_path)
        custom_title = args.title if len(args.inputs) == 1 else None
        output_path = plot_accuracy_curves(
            input_path,
            results,
            output_dir,
            show=args.show,
            title=custom_title,
            dpi=args.dpi,
        )
        print(f"Saved accuracy plot to {output_path}")


if __name__ == "__main__":
    main()
