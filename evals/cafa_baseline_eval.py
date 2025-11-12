"""Baseline CAFA evaluation pipeline for the GCN stacker model.

This script mirrors the high-level workflow of evals/cafa_evals.py but
runs inference with the protein baseline defined in protnn/scripts/train_gcn.py
and evaluates the resulting predictions with the cafaeval package.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from cafaeval.evaluation import cafa_eval
from colorama import Fore, Style, init

from protnn.dataset import StackDataLoader, StackDataset
from protnn.stacker import GCNStacker
from protnn.utils import get_goa_data, make_submission
from protlib.metric import Graph, get_topk_targets, ia_parser, obo_parser

NAMESPACE_TO_KEY = {
    "biological_process": "bp",
    "molecular_function": "mf",
    "cellular_component": "cc",
}
ONTOLOGY_INDEX = {"bp": 0, "mf": 1, "cc": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the GCN protein baseline on CAFA ground truth."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config used for training (e.g. config.yaml)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="checkpoint.pth",
        help=(
            "Checkpoint filename inside models/gcn/<ontology>/ (default: checkpoint.pth)."
        ),
    )
    parser.add_argument(
        "--ontologies",
        nargs="+",
        default=["bp", "mf", "cc"],
        choices=["bp", "mf", "cc"],
        help="Which ontologies to evaluate (default: all).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the evaluation dataloader (default: 128).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for the dataloader (default: 8).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=500,
        help="Top-K predictions to retain per protein when writing submissions (default: 500).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Minimum probability threshold for predictions (default: 0.01).",
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
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def load_graphs(graph_path: Path, ia_path: Path) -> Dict[str, Graph]:
    ia_dict = ia_parser(str(ia_path))
    graphs: Dict[str, Graph] = {}
    for namespace, terms_dict in obo_parser(str(graph_path)).items():
        key = NAMESPACE_TO_KEY.get(namespace)
        if key is None:
            continue
        graphs[key] = Graph(namespace, terms_dict, ia_dict, True)
    return graphs


def load_test_labels(temporal_path: Path) -> pd.DataFrame:
    label_files = [
        temporal_path / "labels" / "prop_test_leak_no_dup.tsv",
        temporal_path / "labels" / "prop_quickgo51.tsv",
    ]
    frames = []
    for file in label_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing label file: {file}")
        frames.append(pd.read_csv(file, sep="\t"))
    labels = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    return labels


def align_entry_ids(helpers_path: Path, entry_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    seq_path = helpers_path / "fasta" / "test_seq.feather"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing FASTA metadata: {seq_path}")
    seq_df = pd.read_feather(seq_path, columns=["EntryID"]).reset_index().set_index("EntryID")
    try:
        subset = seq_df.loc[entry_ids]
    except KeyError as exc:  # provide explicit context on missing IDs
        missing = set(entry_ids) - set(seq_df.index)
        raise KeyError(
            f"Could not locate {len(missing)} EntryID values in {seq_path}: {sorted(missing)[:5]}"
        ) from exc
    ordered_ids = subset.index.to_numpy()
    positional_index = subset["index"].to_numpy().astype(int)
    return ordered_ids, positional_index


def write_ground_truth(labels: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for entry_id, term in labels[["EntryID", "term"]].itertuples(index=False):
            handle.write(f"{entry_id}\t{term}\n")


def build_prediction_sources(
    config: Dict,
    nn_cfg: Dict,
    ontology_key: str,
    graph: Graph,
    models_root: Path,
    helpers_path: Path,
    temporal_path: Path,
    positional_index: np.ndarray,
    ordered_ids: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, bool]], List, np.ndarray, np.ndarray, List[List[int]]]:
    nout = ONTOLOGY_INDEX[ontology_key]

    models_config = []
    for model_name in nn_cfg["preds"]:
        base_cfg = config["base_models"][model_name]
        models_config.append(
            (
                models_root / model_name,
                [base_cfg["bp"], base_cfg["mf"], base_cfg["cc"]],
                base_cfg["conditional"],
            )
        )

    preds: List[Tuple[np.ndarray, np.ndarray, bool]] = []
    idx_lookup: List[List[int]] = []
    for folder, split, conditional in models_config:
        pred_path = folder / "test_pred.pkl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing base prediction file: {pred_path}")
        raw_pred = joblib.load(pred_path)
        # select proteins in the order expected by the model
        raw_pred = raw_pred[positional_index]
        start = sum(split[:nout])
        stop = start + split[nout]
        slice_pred = raw_pred[:, start:stop]
        idx = get_topk_targets(graph, split[nout], train_path=str(Path(config["base_path"]) / "Train"))
        preds.append((slice_pred, idx, conditional))
        idx_lookup.append(idx)

    for side_name in nn_cfg.get("side_preds", []):
        public_cfg = config["public_models"][side_name]
        side_path = models_root / side_name / public_cfg["source"]
        if not side_path.exists():
            raise FileNotFoundError(f"Missing side prediction file: {side_path}")
        side_pred = joblib.load(side_path)
        split = side_pred["borders"]
        start = sum(split[:nout])
        stop = start + split[nout]
        arr = side_pred["test_pred"][positional_index][:, start:stop]
        idx = side_pred["idx"][start:stop]
        preds.append((arr, idx, False))
        idx_lookup.append(idx)

    prior_cnd = joblib.load(helpers_path / f"real_targets/{graph.namespace}/prior.pkl")
    nulls = joblib.load(helpers_path / f"real_targets/{graph.namespace}/nulls.pkl")
    prior_raw = prior_cnd * (1 - nulls)

    goa_data = get_goa_data(str(temporal_path), "test", ordered_ids, graph)
    return preds, goa_data, prior_raw, prior_cnd, idx_lookup


def build_dataloader(
    preds: List[Tuple[np.ndarray, np.ndarray, bool]],
    goa_data: List[pd.Series],
    prior_raw: np.ndarray,
    prior_cnd: np.ndarray,
    graph: Graph,
    batch_size: int,
    num_workers: int,
) -> StackDataLoader:
    dataset = StackDataset(
        preds,
        graph.idxs,
        prior_raw,
        prior_cnd,
        graph,
        goa_list=goa_data,
        p_goa=1,
        targets=None,
    )
    return StackDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_inference_for_ontology(
    ontology: str,
    config: Dict,
    nn_cfg: Dict,
    graph: Graph,
    models_root: Path,
    helpers_path: Path,
    temporal_path: Path,
    positional_index: np.ndarray,
    ordered_ids: np.ndarray,
    checkpoint_name: str,
    batch_size: int,
    num_workers: int,
    predictions_path: Path,
    mode: str,
    topk: int,
    tau: float,
) -> None:
    preds, goa_data, prior_raw, prior_cnd, _ = build_prediction_sources(
        config,
        nn_cfg,
        ontology,
        graph,
        models_root,
        helpers_path,
        temporal_path,
        positional_index,
        ordered_ids,
    )

    dataloader = build_dataloader(
        preds,
        goa_data,
        prior_raw,
        prior_cnd,
        graph,
        batch_size,
        num_workers,
    )

    model = GCNStacker(
        len(preds),
        len(goa_data),
        graph.idxs,
        hidden_size=nn_cfg["hidden_size"],
        n_layers=nn_cfg["n_layers"],
        embed_size=nn_cfg["embed_size"],
    )
    checkpoint_path = models_root / "gcn" / ontology / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.cuda()
    make_submission(
        model,
        dataloader,
        graph,
        ordered_ids,
        str(predictions_path),
        mode=mode,
        topk=topk,
        tau=tau,
    )


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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config = load_config(args.config)
    base_path = Path(config["base_path"]).resolve()
    models_root = base_path / config["models_path"]
    helpers_path = base_path / config["helpers_path"]
    temporal_path = base_path / config["temporal_path"]
    ontology_path = base_path / "Train" / "go-basic.obo"
    ia_file_path = base_path / "IA.txt"

    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / "baseline_predictions.tsv"

    labels = load_test_labels(temporal_path)
    ordered_ids, positional_index = align_entry_ids(
        helpers_path, labels["EntryID"].drop_duplicates().values
    )
    ground_truth_file = output_dir / "ground_truth.tsv"
    write_ground_truth(labels, ground_truth_file)

    graphs = load_graphs(ontology_path, ia_file_path)

    for idx, ontology in enumerate(args.ontologies):
        if ontology not in graphs:
            print(f"{Fore.YELLOW}Skipping unknown ontology {ontology}{Style.RESET_ALL}")
            continue
        print(
            f"{Fore.CYAN}Running inference for ontology {ontology.upper()} using {args.checkpoint_name}{Style.RESET_ALL}"
        )
        nn_cfg = config["gcn"][ontology]
        mode = "w" if idx == 0 else "a"
        run_inference_for_ontology(
            ontology,
            config,
            nn_cfg,
            graphs[ontology],
            models_root,
            helpers_path,
            temporal_path,
            positional_index,
            ordered_ids,
            args.checkpoint_name,
            args.batch_size,
            args.num_workers,
            predictions_file,
            mode,
            args.topk,
            args.tau,
        )

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
