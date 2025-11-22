"""Baseline CAFA evaluation pipeline for the GCN stacker model.

This version expects predictions produced by protnn/scripts/predict_gcn.py
and aggregated in the same fashion as protlib/scripts/postproc/collect_ttas.py.
It then evaluates those predictions with the cafaeval package while loading
ground-truth labels from the stacker's temp directories (mirroring the
post-processing scripts).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
from cafaeval.evaluation import cafa_eval
from colorama import Fore, Style, init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate precomputed GCN stacker predictions on CAFA ground truth."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used for training (e.g. config.yaml)",
    )
    parser.add_argument(
        "--ontologies",
        nargs="+",
        default=["bp", "mf", "cc"],
        choices=["bp", "mf", "cc"],
        help="Which ontologies to evaluate (default: all).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of CPU threads for cafaeval (default: 0 = use all).",
    )
    parser.add_argument(
        "--output_dir",
        default="baseline_eval_results",
        help="Where to write predictions and evaluation artifacts.",
    )
    parser.add_argument(
        "--th_step",
        type=float,
        default=0.01,
        help="Threshold step passed to cafaeval (default: 0.01).",
    )
    parser.add_argument(
        "--tta_count",
        type=int,
        help="Override the number of TTA prediction files to aggregate. Defaults to the config value.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def collect_tta_predictions(models_root: Path, tta_count: int) -> pd.DataFrame:
    """Aggregate TTA submission files into a single prediction dataframe."""
    if tta_count <= 0:
        raise ValueError("tta_count must be positive to collect predictions")

    prediction_files = []
    for idx in range(tta_count):
        path = models_root / "gcn" / f"pred_tta_{idx}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing TTA prediction file {path}. Run predict_gcn.py before evaluation."
            )
        df = pd.read_csv(path, sep="\t", header=None, names=["EntryID", "term", f"prob{idx}"])
        prediction_files.append(df)

    combined = prediction_files[0]
    for df in prediction_files[1:]:
        combined = combined.merge(df, how="outer", on=["EntryID", "term"]).fillna(0)

    prob_cols = [f"prob{idx}" for idx in range(tta_count)]
    combined["prob"] = combined[prob_cols].mean(axis=1)
    return combined[["EntryID", "term", "prob"]]


def infer_tta_count(config: Dict, ontologies: Sequence[str]) -> int:
    """Determine how many TTA submissions were generated from the config."""

    for ontology in ontologies:
        ontology_cfg = config.get("gcn", {}).get(ontology)
        if ontology_cfg and ontology_cfg.get("tta"):
            return len(ontology_cfg["tta"])

    for ontology_cfg in config.get("gcn", {}).values():
        if ontology_cfg.get("tta"):
            return len(ontology_cfg["tta"])

    raise ValueError(
        "Could not infer TTA count from config. Ensure gcn.<ontology>.tta is defined."
    )


def load_labels_from_temp(models_root: Path, ontologies: Sequence[str]) -> pd.DataFrame:
    """Load ground-truth labels from the GCN temp folders (train_gcn output)."""

    label_frames: List[pd.DataFrame] = []
    searched_paths: List[Path] = []
    for ontology in ontologies:
        candidate = models_root / "gcn" / ontology / "temp" / "labels.tsv"
        searched_paths.append(candidate)
        if candidate.exists():
            df = pd.read_csv(candidate, sep="\t")
            if not {"EntryID", "term"}.issubset(df.columns):
                raise ValueError(
                    f"Label file {candidate} is missing required columns EntryID and term."
                )
            label_frames.append(df[["EntryID", "term"]])

    if not label_frames:
        joined = "\n".join(str(path) for path in searched_paths)
        raise FileNotFoundError(
            "Could not find labels.tsv in any ontology temp directory. "
            "Ensure train_gcn.py has been run. Checked:\n" + joined
        )

    labels = pd.concat(label_frames).drop_duplicates().reset_index(drop=True)
    return labels


def write_ground_truth(labels: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for entry_id, term in labels[["EntryID", "term"]].itertuples(index=False):
            handle.write(f"{entry_id}\t{term}\n")


def run_cafa_evaluation(
    ontology_path: Path,
    predictions_dir: Path,
    ground_truth_path: Path,
    ia_file_path: Path,
    n_cpu: int,
    th_step: float,
):
    print("Running CAFA evaluation via cafaeval...")
    if ia_file_path and ia_file_path.exists():
        results = cafa_eval(
            str(ontology_path),
            str(predictions_dir),
            str(ground_truth_path),
            str(ia_file_path),
            th_step=th_step,
        )
    else:
        results = cafa_eval(
            str(ontology_path),
            str(predictions_dir),
            str(ground_truth_path),
            th_step=th_step,
        )
    return results


def extract_metrics_summary(results) -> Dict[str, float]:
    evaluation_df, best_scores_dict = results
    metrics: Dict[str, float] = {}
    df = best_scores_dict.get("f", best_scores_dict.get("f_w"))
    if df is None:
        return metrics
    df = df.reset_index()
    namespaces = df["ns"].unique()
    for ns in namespaces:
        ns_row = df[df["ns"] == ns].iloc[0]
        metrics[f"{ns}_f1"] = ns_row["f"]
        metrics[f"{ns}_weighted_f1"] = ns_row["f_w"]
    metrics["overall_mean_f1"] = np.mean([metrics[f"{ns}_f1"] for ns in namespaces])
    metrics["overall_mean_weighted_f1"] = np.mean(
        [metrics[f"{ns}_weighted_f1"] for ns in namespaces]
    )
    return metrics


def print_results_summary(metrics: Dict[str, float]) -> None:
    if not metrics:
        print(f"{Fore.RED}No metrics were produced by cafaeval.{Style.RESET_ALL}")
        return
    print("\n" + "=" * 60)
    print("BASELINE CAFA EVALUATION SUMMARY")
    print("=" * 60)
    print("\nF1 SCORES:")
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        key = f"{aspect}_f1"
        if key in metrics:
            print(f"{aspect:25}: {metrics[key]:.4f}")
    print(f"{'OVERALL AVERAGE':25}: {metrics.get('overall_mean_f1', float('nan')):.4f}")
    print("\nWEIGHTED F1 SCORES:")
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        key = f"{aspect}_weighted_f1"
        if key in metrics:
            print(f"{aspect:25}: {metrics[key]:.4f}")
    print(
        f"{'OVERALL AVERAGE':25}: {metrics.get('overall_mean_weighted_f1', float('nan')):.4f}"
    )
    print("=" * 60)


def main():
    init()
    args = parse_args()

    config = load_config(args.config)
    base_path = Path(config["base_path"]).resolve()
    models_root = base_path / config["models_path"]
    ontology_path = base_path / "Train" / "go-basic.obo"
    ia_file_path = base_path / "IA.txt"

    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / "baseline_predictions.tsv"

    tta_count = args.tta_count or infer_tta_count(config, args.ontologies)
    print(f"{Fore.CYAN}Aggregating {tta_count} TTA prediction files...{Style.RESET_ALL}")
    predictions = collect_tta_predictions(models_root, tta_count)
    predictions.to_csv(predictions_file, sep="\t", header=False, index=False)

    print(f"{Fore.CYAN}Loading ground truth labels from stacker temp directories...{Style.RESET_ALL}")
    labels = load_labels_from_temp(models_root, args.ontologies)
    ground_truth_file = output_dir / "ground_truth.tsv"
    write_ground_truth(labels, ground_truth_file)

    results = run_cafa_evaluation(
        ontology_path,
        predictions_dir,
        ground_truth_file,
        ia_file_path,
        args.threads,
        args.th_step,
    )

    metrics = extract_metrics_summary(results)
    print_results_summary(metrics)

    evaluation_df, best_scores_dict = results
    evaluation_df.to_csv(output_dir / "evaluation_results.tsv", sep="\t", index=False)
    for metric_name, df in best_scores_dict.items():
        df.to_csv(output_dir / f"best_{metric_name}.tsv", sep="\t")
    with open(output_dir / "metrics_summary.json", "w") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
