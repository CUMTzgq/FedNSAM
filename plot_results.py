import argparse
import json
import os
from pathlib import Path
from typing import Any


COLOR_MAP = {
    "fedavg": "#1f77b4",
    "fedsam": "#ff7f0e",
    "fednsam": "#2ca02c",
}
DISPLAY_LABELS = {
    "avg": "FedAvg+DP",
    "fedavg": "FedAvg+DP",
    "sam": "FedSAM+DP",
    "fedsam": "FedSAM+DP",
    "nsam": "NSDC-FL",
    "fednsam": "NSDC-FL",
}
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy curves from saved FedNSAM/APC-DPFL result JSON files.")
    parser.add_argument("inputs", nargs="+", help="One or more result JSON files.")
    parser.add_argument("--output-dir", default="figures", help="Directory to save generated plots.")
    parser.add_argument("--output-name", default=None, help="Filename for combined multi-file plots.")
    parser.add_argument("--separate", action="store_true", help="Save one plot per input JSON instead of one combined plot.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    parser.add_argument("--title", default=None, help="Optional custom plot title.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="ALGORITHM=DISPLAY_NAME",
        help=(
            "Override legend labels, for example --label fednsam=NSDC-FL. "
            "Defaults are fednsam=NSDC-FL, fedavg=FedAvg+DP, fedsam=FedSAM+DP."
        ),
    )
    return parser.parse_args()


def parse_label_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_override in raw_overrides:
        if "=" not in raw_override:
            raise ValueError(f"Invalid --label value: {raw_override!r}. Expected ALGORITHM=DISPLAY_NAME.")
        algorithm, label = raw_override.split("=", maxsplit=1)
        algorithm = algorithm.strip().lower()
        label = label.strip()
        if not algorithm or not label:
            raise ValueError(f"Invalid --label value: {raw_override!r}. Expected ALGORITHM=DISPLAY_NAME.")
        overrides[algorithm] = label
    return overrides


def display_label_for(algorithm: str, label_overrides: dict[str, str]) -> str:
    key = algorithm.strip().lower()
    return label_overrides.get(key) or DISPLAY_LABELS.get(key) or algorithm


def normalize_accuracy(values: list[Any]) -> list[float]:
    accuracy = [float(value) for value in values]
    if not accuracy:
        return accuracy
    finite_values = [value for value in accuracy if value == value]
    if finite_values and max(finite_values) <= 1.5:
        return [value * 100.0 for value in accuracy]
    return accuracy


def normalize_rounds(values: list[Any] | None, length: int) -> list[int]:
    if values is None:
        return list(range(1, length + 1))
    rounds = [int(value) for value in values]
    if len(rounds) != length:
        raise ValueError("mismatched 'round' and accuracy lengths.")
    return rounds


def get_config_value(config: object, key: str) -> object | None:
    if not isinstance(config, dict):
        return None
    value = config.get(key)
    if isinstance(value, str):
        return value.strip()
    return value


def infer_single_file_algorithm(path: Path, data: dict[str, Any]) -> str:
    config = data.get("config")
    for key in ("algorithm", "method", "name"):
        value = get_config_value(config, key)
        if value:
            return str(value)

    stem = path.stem.lower()
    for algorithm in ("fednsam", "nsam", "fedsam", "sam", "fedavg", "avg"):
        if algorithm in stem:
            return algorithm
    return path.stem


def extract_accuracy_payload(payload: dict[str, Any], *, path: Path, algorithm: str) -> dict[str, object]:
    if "accuracy" in payload:
        accuracy_values = payload["accuracy"]
    elif "test_accs" in payload:
        accuracy_values = payload["test_accs"]
    elif "accuracies" in payload:
        accuracy_values = payload["accuracies"]
    else:
        raise ValueError(f"{path} -> {algorithm} is missing 'accuracy', 'test_accs', or 'accuracies'.")

    if not isinstance(accuracy_values, list):
        raise ValueError(f"{path} -> {algorithm} requires a list accuracy value.")

    rounds = payload.get("round")
    if rounds is not None and not isinstance(rounds, list):
        raise ValueError(f"{path} -> {algorithm} requires a list value for 'round'.")

    return {
        "round": normalize_rounds(rounds, len(accuracy_values)),
        "accuracy": normalize_accuracy(accuracy_values),
    }


def looks_like_algorithm_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in ("accuracy", "accuracies", "test_accs"))


def load_result_file(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} does not contain a non-empty result object.")

    normalized: dict[str, dict[str, object]] = {}

    # FedNSAM compare JSON: {"fedavg": {"round": [...], "accuracy": [...]}, ... , "_meta": {...}}
    for algorithm, payload in data.items():
        if str(algorithm).startswith("_"):
            continue
        if not looks_like_algorithm_payload(payload):
            continue
        if not isinstance(payload, dict):
            continue
        normalized[str(algorithm)] = extract_accuracy_payload(payload, path=path, algorithm=str(algorithm))

    if normalized:
        return normalized

    # APC-DPFL-style single-run JSON: {"accuracies": [...]} or {"test_accs": [...], "train_accs": [...]}
    if looks_like_algorithm_payload(data):
        algorithm = infer_single_file_algorithm(path, data)
        return {algorithm: extract_accuracy_payload(data, path=path, algorithm=algorithm)}

    raise ValueError(f"{path} does not contain any plottable accuracy series.")


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


def build_series_label(
    input_path: Path,
    algorithm: str,
    *,
    include_file: bool,
    label_overrides: dict[str, str],
) -> str:
    algorithm_label = display_label_for(algorithm, label_overrides)
    if not include_file:
        return algorithm_label
    return f"{input_path.stem} / {algorithm_label}"


def plot_accuracy_curves(
    series: list[tuple[Path, str, dict[str, object], int]],
    output_path: Path,
    *,
    show: bool,
    title: str | None,
    dpi: int,
    include_file_in_label: bool,
    label_overrides: dict[str, str],
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
            label=build_series_label(
                input_path,
                algorithm,
                include_file=include_file_in_label,
                label_overrides=label_overrides,
            ),
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
    label_overrides = parse_label_overrides(args.label)

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
                label_overrides=label_overrides,
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
        label_overrides=label_overrides,
    )
    print(f"Saved accuracy plot to {output_path}")


if __name__ == "__main__":
    main()
