import argparse
import json
import os
from pathlib import Path


COLOR_MAP = {
    "fedavg": "#1f77b4",
    "fedsam": "#ff7f0e",
    "fednsam": "#2ca02c",
}
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy curves from saved FedNSAM result JSON files.")
    parser.add_argument("inputs", nargs="+", help="One or more result JSON files.")
    parser.add_argument("--output-dir", default="figures", help="Directory to save generated plots.")
    parser.add_argument("--output-name", default=None, help="Filename for combined multi-file plots.")
    parser.add_argument("--separate", action="store_true", help="Save one plot per input JSON instead of one combined plot.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    parser.add_argument("--title", default=None, help="Optional custom plot title.")
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


def combined_output_path(input_paths: list[Path], output_dir: Path, output_name: str | None) -> Path:
    if output_name is not None:
        output_path = Path(output_name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        if output_path.is_absolute():
            return output_path
        return output_dir / output_path

    if len(input_paths) <= 3:
        stem = "_vs_".join(path.stem for path in input_paths)
    else:
        stem = "combined"
    return output_dir / f"{stem}_accuracy.png"


def default_title(input_path: Path) -> str:
    return input_path.stem.replace("_", " ")


def default_combined_title(input_paths: list[Path]) -> str:
    if len(input_paths) <= 3:
        return " vs ".join(default_title(path) for path in input_paths)
    return "Combined accuracy comparison"


def build_series_label(input_path: Path, algorithm: str, *, include_file: bool) -> str:
    if not include_file:
        return algorithm
    return f"{input_path.stem} / {algorithm}"


def plot_accuracy_curves(
    series: list[tuple[Path, str, dict[str, object], int]],
    output_path: Path,
    *,
    show: bool,
    title: str | None,
    dpi: int,
    include_file_in_label: bool,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fednsam-mplconfig")
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for input_path, algorithm, payload, file_index in series:
        color = COLOR_MAP.get(algorithm.lower())
        line_style = LINE_STYLES[file_index % len(LINE_STYLES)]
        marker = MARKERS[file_index % len(MARKERS)]
        axis.plot(
            payload["round"],
            payload["accuracy"],
            marker=marker,
            linestyle=line_style,
            linewidth=2.0,
            markersize=4.5,
            label=build_series_label(input_path, algorithm, include_file=include_file_in_label),
            color=color,
        )

    axis.set_xlabel("Round")
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_title(title)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_paths = [Path(raw_path) for raw_path in args.inputs]

    if args.separate:
        for input_path in input_paths:
            results = load_result_file(input_path)
            series = [
                (input_path, algorithm, payload, 0)
                for algorithm, payload in results.items()
            ]
            output_path = plot_accuracy_curves(
                series,
                output_path_for(input_path, output_dir),
                show=args.show,
                title=args.title if len(input_paths) == 1 else default_title(input_path),
                dpi=args.dpi,
                include_file_in_label=False,
            )
            print(f"Saved accuracy plot to {output_path}")
        return

    combined_series: list[tuple[Path, str, dict[str, object], int]] = []
    for file_index, input_path in enumerate(input_paths):
        results = load_result_file(input_path)
        for algorithm, payload in results.items():
            combined_series.append((input_path, algorithm, payload, file_index))

    output_path = plot_accuracy_curves(
        combined_series,
        combined_output_path(input_paths, output_dir, args.output_name),
        show=args.show,
        title=args.title if args.title is not None else default_combined_title(input_paths),
        dpi=args.dpi,
        include_file_in_label=len(input_paths) > 1,
    )
    print(f"Saved accuracy plot to {output_path}")


if __name__ == "__main__":
    main()
