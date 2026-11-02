"""Command line interface for every pipeline stage.

Each DVC stage calls the same entry point with a different sub-command, so the
tests can exercise the whole pipeline the same way a user does.
"""

import argparse
from pathlib import Path

from mlops.config import DEFAULT_PARAMS, Paths, load_params
from mlops.data import prepare_dataset
from mlops.evaluate import evaluate_model
from mlops.generate import generate_dataset
from mlops.train import train_model


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mlops",
        description=(
            "Synthetic regression pipeline: generate, prepare, train, evaluate."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="pipeline working directory (default: current directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="create the raw synthetic dataset")
    subparsers.add_parser("prepare", help="clean, split and scale the dataset")
    subparsers.add_parser("train", help="fit the ridge regression model")
    subparsers.add_parser("evaluate", help="score the model and write metrics")

    all_stages = subparsers.add_parser("all", help="run every stage in order")
    all_stages.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="override test_size from params.yaml",
    )
    return parser


def run_generate(paths: Paths) -> Path:
    """Run the generate stage."""
    params = load_params(paths.params)
    return generate_dataset(params, paths.raw)


def run_prepare(paths: Paths) -> tuple[Path, Path, Path]:
    """Run the prepare stage."""
    params = load_params(paths.params)
    return prepare_dataset(
        raw_path=paths.raw,
        params_seed=params["seed"],
        test_size=params["test_size"],
        data_dir=paths.data_dir,
    )


def run_train(paths: Paths) -> list[float]:
    """Run the train stage and return the learned weights."""
    params = load_params(paths.params)
    model = train_model(paths.train, paths.model, params["ridge_alpha"])
    if model.weights is None:
        raise RuntimeError("model was not fitted")
    return model.weights


def run_evaluate(paths: Paths) -> dict[str, float]:
    """Run the evaluate stage."""
    return evaluate_model(paths.test, paths.model, paths.metrics)


def run_all(paths: Paths, test_size: float | None = None) -> dict[str, float]:
    """Run every stage in order and return the final metrics."""
    run_generate(paths)
    params = load_params(paths.params)
    if test_size is not None:
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        params = {**params, "test_size": test_size}
        paths.params.write_text(
            "\n".join(f"{key}: {value}" for key, value in params.items()) + "\n",
            encoding="utf-8",
        )
    run_prepare(paths)
    run_train(paths)
    return run_evaluate(paths)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m mlops.cli``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = Paths(root=args.root)

    match args.command:
        case "generate":
            run_generate(paths)
        case "prepare":
            run_prepare(paths)
        case "train":
            run_train(paths)
        case "evaluate":
            metrics = run_evaluate(paths)
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}")
        case "all":
            metrics = run_all(paths, test_size=args.test_size)
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}")
        case _:  # pragma: no cover - argparse rejects unknown commands first
            parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def default_params_text() -> str:
    """Return ``params.yaml`` content for a fresh checkout."""
    return "\n".join(f"{key}: {value}" for key, value in DEFAULT_PARAMS.items()) + "\n"
