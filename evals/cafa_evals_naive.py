# cafa_naive_baseline.py
# This script computes CAFA naive baseline metrics by assigning GO term frequencies
# from the training set as prediction scores for all proteins in the test set.

import numpy as np
import os
from typing import Set, List, Tuple, Dict
from collections import Counter
from tqdm import tqdm
from colorama import init, Fore, Style
import argparse

# Import functions from existing modules
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from cafa_evals import (
    create_cafa_ground_truth_file,
    run_cafa_evaluation,
    extract_metrics_summary,
    print_results_summary,
)


def compute_go_term_frequencies(train_dataset, min_frequency=0.001) -> Dict[str, float]:
    """
    Compute the frequency of each GO term in the training dataset.
    Only returns GO terms with frequency >= min_frequency.
    """
    print(f"{Fore.CYAN}Computing GO term frequencies from training data...{Style.RESET_ALL}")

    # Count occurrences of each GO term
    go_counter = Counter()
    total_proteins = 0

    for sample in tqdm(train_dataset, desc="Processing training proteins"):
        total_proteins += 1

        # Extract GO terms from go_ids column (numpy string array)
        go_ids = sample.get("go_ids", [])
        if go_ids is not None and len(go_ids) > 0:
            for go_term in go_ids:
                if go_term and go_term.startswith("GO:"):
                    go_counter[go_term] += 1

    # Convert counts to frequencies and filter
    go_frequencies = {}
    filtered_count = 0
    total_count = len(go_counter)

    for go_term, count in go_counter.items():
        freq = count / total_proteins
        if freq >= min_frequency:
            go_frequencies[go_term] = freq
        else:
            filtered_count += 1

    print(f"{Fore.GREEN}Found {total_count} total unique GO terms{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Kept {len(go_frequencies)} GO terms with frequency >= {min_frequency}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Filtered out {filtered_count} rare GO terms (below threshold){Style.RESET_ALL}")

    # Print statistics
    if go_frequencies:
        freq_values = list(go_frequencies.values())
        print(f"{Fore.CYAN}Frequency statistics (after filtering):{Style.RESET_ALL}")
        print(f"  - Min frequency: {min(freq_values):.6f}")
        print(f"  - Max frequency: {max(freq_values):.6f}")
        print(f"  - Mean frequency: {np.mean(freq_values):.6f}")
        print(f"  - Median frequency: {np.median(freq_values):.6f}")

    return go_frequencies


def create_naive_predictions(test_dataset, go_frequencies: Dict[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    """
    Create naive predictions by assigning all GO terms with their frequencies
    to every protein in the test set.
    Returns list of (protein_id, {go_term: score}) tuples.
    """
    print(f"{Fore.CYAN}Creating naive predictions for test set...{Style.RESET_ALL}")

    predictions = []

    for idx, sample in enumerate(tqdm(test_dataset, desc="Creating predictions")):
        # Use index as protein ID to ensure uniqueness
        protein_id = f"protein_{idx:06d}"

        # Every protein gets ALL GO terms with their frequency scores
        predictions.append((protein_id, go_frequencies.copy()))

    print(f"{Fore.GREEN}Created predictions for {len(predictions)} proteins{Style.RESET_ALL}")

    return predictions


def extract_ground_truth_from_dataset(test_dataset) -> List[Tuple[str, Set[str]]]:
    """
    Extract ground truth GO terms from test dataset.
    Returns list of (protein_id, set of GO terms) tuples.
    """
    print(f"{Fore.CYAN}Extracting ground truth from test set...{Style.RESET_ALL}")

    ground_truth = []

    for idx, sample in enumerate(tqdm(test_dataset, desc="Extracting ground truth")):
        # Use same protein ID scheme as predictions
        protein_id = f"protein_{idx:06d}"

        # Extract GO terms from go_ids column
        go_ids = sample.get("go_ids", [])
        go_terms = set()

        if go_ids is not None and len(go_ids) > 0:
            for go_term in go_ids:
                if go_term and go_term.startswith("GO:"):
                    go_terms.add(go_term)

        if go_terms:  # Only add if protein has GO terms
            ground_truth.append((protein_id, go_terms))

    print(f"{Fore.GREEN}Extracted ground truth for {len(ground_truth)} proteins{Style.RESET_ALL}")

    return ground_truth


def create_naive_cafa_prediction_file(predictions: List[Tuple[str, Dict[str, float]]], output_path: str):
    """
    Create CAFA-format prediction file with frequency scores.
    Format: target_id, term_id, score
    """
    print(f"Creating naive prediction file: {output_path}")

    with open(output_path, "w") as f:
        for target_id, go_scores in predictions:
            for go_term, score in go_scores.items():
                f.write(f"{target_id}\t{go_term}\t{score:.6f}\n")


def main():
    """Main pipeline for CAFA naive baseline evaluation."""

    # Initialize colorama for colored output
    init()

    parser = argparse.ArgumentParser(description="CAFA Naive Baseline Evaluation")
    parser.add_argument(
        "--ontology",
        "-o",
        required=True,
        help="Path to GO ontology file (go-basic.obo)",
    )
    parser.add_argument(
        "--ia-file",
        "-a",
        required=True,
        help="Path to Information Accretion file (IA.txt)",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default="naive_baseline_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=0,
        help="Number of CPU threads to use (0 = all available)",
    )
    parser.add_argument(
        "--min-frequency",
        "-f",
        type=float,
        default=0.001,
        help="Minimum GO term frequency to include (default: 0.001 = 0.1%)",
    )
    parser.add_argument(
        "--threshold-step",
        "-s",
        type=float,
        default=0.01,
        help="Threshold step size for CAFA evaluation (default: 0.01)",
    )
    parser.add_argument("--cache-dir", "-c", default="cache", help="Cache directory for dataset")

    args = parser.parse_args()

    print(f"{Fore.CYAN}Starting CAFA Naive Baseline Evaluation{Style.RESET_ALL}")
    print("-" * 40)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load datasets
    print(f"{Fore.CYAN}Loading CAFA5 dataset...{Style.RESET_ALL}")
    train_dataset, val_dataset, test_dataset = load_cafa5_dataset(
        dataset="wanglab/cafa5",
        dataset_subset=None,
        max_length=2048,
        val_split_ratio=0.1,
        seed=23,
        return_as_chat_template=False,
        cache_dir=args.cache_dir,
        include_go_defs=False,
    )

    # Verify we're using validation as test (as mentioned)
    print(f"{Fore.YELLOW}Note: Using validation set ({len(val_dataset)} samples) as test set{Style.RESET_ALL}")

    # Step 2: Compute GO term frequencies from training data
    go_frequencies = compute_go_term_frequencies(train_dataset, min_frequency=args.min_frequency)

    # Step 3: Create naive predictions for test set (val_dataset)
    predictions = create_naive_predictions(val_dataset, go_frequencies)

    # Step 4: Extract ground truth from test set
    ground_truth = extract_ground_truth_from_dataset(val_dataset)

    if not predictions:
        print(f"{Fore.RED}ERROR: No predictions created!{Style.RESET_ALL}")
        return

    if not ground_truth:
        print(f"{Fore.RED}ERROR: No ground truth data found!{Style.RESET_ALL}")
        return

    # Step 5: Create CAFA format files
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    prediction_file = os.path.join(predictions_dir, "naive_predictions.tsv")
    ground_truth_file = os.path.join(args.output_dir, "ground_truth.tsv")

    # Use the naive prediction file creator for frequency scores
    create_naive_cafa_prediction_file(predictions, prediction_file)
    create_cafa_ground_truth_file(ground_truth, ground_truth_file)

    # Step 6: Run CAFA evaluation
    try:
        results = run_cafa_evaluation(
            args.ontology,
            predictions_dir,
            ground_truth_file,
            ia_file_path=args.ia_file,
            n_cpu=args.threads,
            th_step=args.threshold_step,
        )

        # Step 7: Extract and display metrics
        metrics = extract_metrics_summary(results)
        print_results_summary(metrics)

        # Step 8: Save detailed results
        evaluation_df, best_scores_dict = results

        # Save main evaluation results
        evaluation_df.to_csv(os.path.join(args.output_dir, "evaluation_results.tsv"), sep="\t")
        print(f"\nEvaluation results saved to: {os.path.join(args.output_dir, 'evaluation_results.tsv')}")

        # Save best scores for each metric
        for metric, df in best_scores_dict.items():
            metric_path = os.path.join(args.output_dir, f"best_{metric}.tsv")
            df.to_csv(metric_path, sep="\t")
        print(f"Best score files saved to: {args.output_dir}")

    except Exception as e:
        print(f"{Fore.RED}ERROR during evaluation: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
        print("Please check that all input files exist and are in the correct format.")


if __name__ == "__main__":
    main()
