import pandas as pd
import re
import json
import os
from pathlib import Path
from typing import Set, List, Tuple, Dict
from tqdm import tqdm
from cafaeval.evaluation import cafa_eval
from colorama import init, Fore, Style
import argparse

# GO aspect classification constants
NAMESPACE_TO_ASPECT = {
    'molecular_function': 'MF',
    'biological_process': 'BP', 
    'cellular_component': 'CC'
}


def extract_go_terms(text: str) -> Set[str]:
    """
    Extract all GO terms from text in format GO:XXXXXXX.
    Removes duplicates by returning a set.
    """
    # Find all GO terms in format GO:XXXXXXX (7 digits)
    go_pattern = r"GO:\d{7}"
    go_terms = set(re.findall(go_pattern, text))

    return go_terms


def parse_ground_truth_format(text: str) -> Set[str]:
    """
    Extract GO terms from ground truth format with GO_SUMMARY tags.
    Format: <|GO_SUMMARY_START|>\nMF: GO:XXXX, GO:YYYY\nBP: GO:ZZZZ\n<|GO_SUMMARY_END|>
    """
    # First try to extract content between GO_SUMMARY tags
    summary_pattern = r"<\|GO_SUMMARY_START\|>(.*?)<\|GO_SUMMARY_END\|>"
    summary_match = re.search(summary_pattern, text, re.DOTALL)

    if summary_match:
        summary_content = summary_match.group(1)
        # Extract all GO terms from the summary content
        return extract_go_terms(summary_content)
    else:
        # Fallback to regular GO term extraction if no summary tags found
        return extract_go_terms(text)


def extract_reasoning_ground_truth(sample: Dict) -> Tuple[Set[str], Set[str]]:
    """
    Extract ground truth GO terms from reasoning data leaf columns.
    
    Args:
        sample: Dictionary containing go_bp_leaf, go_mf_leaf, go_cc_leaf fields
        
    Returns:
        Tuple of (all_gt_terms, present_aspects) where:
        - all_gt_terms: Set of all GO terms from present aspects
        - present_aspects: Set of aspect codes that have non-empty ground truth
    """
    all_gt_terms = set()
    present_aspects = set()
    
    leaf_columns = {
        "MF": "go_mf_leaf",
        "BP": "go_bp_leaf", 
        "CC": "go_cc_leaf"
    }
    
    for aspect, column in leaf_columns.items():
        leaf_data = sample.get(column)
        
        # Check if aspect has actual GO terms (not None, not "Not known", not empty)
        if leaf_data and leaf_data != "Not known" and leaf_data.strip():
            # Extract GO terms from the leaf data
            go_terms = extract_go_terms(leaf_data)
            if go_terms:  # Only add if we actually found GO terms
                all_gt_terms.update(go_terms)
                present_aspects.add(aspect)
    
    return all_gt_terms, present_aspects


def classify_go_term_by_aspect(go_term: str, go_dag) -> str:
    """
    Classify a GO term into its aspect (MF/BP/CC) using the GO ontology namespace.
    
    Args:
        go_term: GO term ID (e.g., "GO:0008150")
        go_dag: GO DAG object from goatools
        
    Returns:
        Aspect code ("MF", "BP", "CC") or None if not found
    """
    if not go_dag or go_term not in go_dag:
        return None
    
    # Use the namespace attribute for efficient classification
    namespace = go_dag[go_term].namespace
    
    return NAMESPACE_TO_ASPECT.get(namespace, None)


def filter_predictions_by_aspects(predicted_terms: Set[str], present_aspects: Set[str], go_dag) -> Set[str]:
    """
    Filter predicted GO terms to only include those belonging to aspects with ground truth.
    
    Args:
        predicted_terms: Set of predicted GO term IDs
        present_aspects: Set of aspect codes that have ground truth ("MF", "BP", "CC")
        go_dag: GO DAG object for classification
        
    Returns:
        Filtered set of GO terms belonging to present aspects
    """
    if not go_dag:
        # Fallback: return all predictions if no GO DAG available
        return predicted_terms
    
    filtered_terms = set()
    
    for go_term in predicted_terms:
        aspect = classify_go_term_by_aspect(go_term, go_dag)
        if aspect and aspect in present_aspects:
            filtered_terms.add(go_term)
    
    return filtered_terms


def parse_prediction_format(text: str, summary_only: bool = False) -> Set[str]:
    """
    Extract GO terms from prediction text, optionally only from summary sections.

    Args:
        text: The prediction text to parse
        summary_only: If True, only extract GO terms from GO_SUMMARY tags.
                     If False, extract from all text.
    """
    if summary_only:
        # Extract only from GO_SUMMARY tags
        summary_pattern = r"<\|GO_SUMMARY_START\|>(.*?)<\|GO_SUMMARY_END\|>"
        summary_match = re.search(summary_pattern, text, re.DOTALL)

        if summary_match:
            summary_content = summary_match.group(1)
            return extract_go_terms(summary_content)
        else:
            # If no summary found and summary_only is True, return empty set
            return set()
    else:
        # Extract from all text (original behavior)
        return extract_go_terms(text)


def load_json_files_from_directory(directory: str) -> List[Dict]:
    """
    Load all JSON files from a single chunk directory.
    Excludes errors.jsonl files.
    """
    json_files = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"{Fore.YELLOW}Warning: Directory {directory} does not exist{Style.RESET_ALL}")
        return json_files

    # Find all .json files, excluding errors.jsonl
    for json_file in dir_path.glob("*.json"):
        if json_file.name != "errors.jsonl":
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        json_files.extend(data)
                    else:
                        json_files.append(data)
            except Exception as e:
                print(f"{Fore.RED}Error loading {json_file}: {e}{Style.RESET_ALL}")

    return json_files


def process_json_data(
    base_dir: str, summary_only: bool = False, reasoning_mode: bool = False, go_dag=None
) -> Tuple[List[Tuple[str, Set[str]]], List[Tuple[str, Set[str]]]]:
    """
    Process JSON files from directory and extract predictions and ground truth.
    Args:
        base_dir: Directory containing JSON files or chunk directories with JSON files
        summary_only: If True, extract GO terms only from summary sections for predictions
        reasoning_mode: If True, use leaf columns for ground truth and filter predictions by aspects
        go_dag: GO DAG object for aspect classification (required for reasoning_mode)
    Returns: (predictions_list, ground_truth_list)
    """
    print(f"{Fore.CYAN}Loading and processing JSON data from {base_dir}...{Style.RESET_ALL}")
    if reasoning_mode:
        print(f"{Fore.YELLOW}Using reasoning evaluation mode with aspect filtering{Style.RESET_ALL}")

    predictions = []
    ground_truth = []
    total_samples = 0
    successful_samples = 0
    aspect_stats = {"MF": 0, "BP": 0, "CC": 0}  # Track aspect presence
    proteins_with_filtering = 0
    total_predictions_filtered = 0

    base_path = Path(base_dir)

    # Try loading JSON files directly first
    json_data = load_json_files_from_directory(base_dir)

    # If no JSON files found directly, look for chunk directories
    if not json_data:
        chunk_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        for chunk_dir in chunk_dirs:
            json_data.extend(load_json_files_from_directory(chunk_dir))

    for sample in tqdm(json_data, desc="Processing samples"):
        total_samples += 1
        if not sample.get("success", False):
            continue

        successful_samples += 1
        target_id = sample.get("protein_id", f"unknown_protein_{successful_samples}")

        if reasoning_mode:
            # Extract ground truth from leaf columns
            gt_terms, present_aspects = extract_reasoning_ground_truth(sample)
            
            # Track aspect statistics
            for aspect in present_aspects:
                aspect_stats[aspect] += 1
            
            # Extract predicted GO terms and filter by present aspects
            generated_text = sample.get("generated_response", "")
            all_predicted_terms = parse_prediction_format(generated_text, summary_only)
            
            if all_predicted_terms and go_dag:
                predicted_terms = filter_predictions_by_aspects(all_predicted_terms, present_aspects, go_dag)
                if len(predicted_terms) < len(all_predicted_terms):
                    proteins_with_filtering += 1
                    total_predictions_filtered += (len(all_predicted_terms) - len(predicted_terms))
            else:
                predicted_terms = all_predicted_terms
            
            # Only add if we have data
            if predicted_terms:
                predictions.append((target_id, predicted_terms))
            if gt_terms:
                ground_truth.append((target_id, gt_terms))
                
        else:
            # Original mode: extract from text
            # Extract predicted GO terms
            generated_text = sample.get("generated_response", "")
            predicted_terms = parse_prediction_format(generated_text, summary_only)
            if predicted_terms:
                predictions.append((target_id, predicted_terms))

            # Extract ground truth GO terms
            gt_text = sample.get("ground_truth", "")
            gt_terms = parse_ground_truth_format(gt_text)
            if gt_terms:
                ground_truth.append((target_id, gt_terms))

    # Count total annotations
    total_predicted_annotations = sum(len(terms) for _, terms in predictions)
    total_gt_annotations = sum(len(terms) for _, terms in ground_truth)

    print(f"{Fore.CYAN}Total samples processed: {total_samples}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Successful samples: {successful_samples}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Failed samples: {total_samples - successful_samples}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}Processed {len(predictions)} proteins with {total_predicted_annotations} predicted annotations{Style.RESET_ALL}"
    )
    print(
        f"{Fore.CYAN}Processed {len(ground_truth)} proteins with {total_gt_annotations} ground truth annotations{Style.RESET_ALL}"
    )
    
    if reasoning_mode:
        print(f"{Fore.MAGENTA}Aspect presence in ground truth:{Style.RESET_ALL}")
        for aspect, count in aspect_stats.items():
            print(f"  {aspect}: {count} proteins")
        if go_dag:
            print(f"{Fore.MAGENTA}Proteins with filtered predictions: {proteins_with_filtering}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Total GO terms filtered out: {total_predictions_filtered}{Style.RESET_ALL}")

    return predictions, ground_truth


def create_cafa_prediction_file(predictions: List[Tuple[str, Set[str]]], output_path: str):
    """Create CAFA-format prediction file: target_id, term_id, score"""
    print(f"Creating prediction file: {output_path}")

    with open(output_path, "w") as f:
        for target_id, go_terms in predictions:
            for go_term in go_terms:
                # Assign score=1.0 to all predicted GO terms
                f.write(f"{target_id}\t{go_term}\t1.0\n")


def create_cafa_ground_truth_file(ground_truth: List[Tuple[str, Set[str]]], output_path: str):
    """Create CAFA-format ground truth file: target_id, term_id"""
    print(f"Creating ground truth file: {output_path}")

    with open(output_path, "w") as f:
        for target_id, go_terms in ground_truth:
            for go_term in go_terms:
                f.write(f"{target_id}\t{go_term}\n")


def run_cafa_evaluation(
    ontology_path: str,
    predictions_dir: str,
    ground_truth_path: str,
    ia_file_path: str = None,
    n_cpu: int = 0,
    th_step: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run CAFA evaluation using cafaeval package.
    """
    print("Running CAFA evaluation...")
    print(f"Ontology: {ontology_path}")
    print(f"Predictions: {predictions_dir}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Information Accretion file: {ia_file_path if ia_file_path else 'Not provided'}")
    print(f"Using {n_cpu if n_cpu > 0 else 'all available'} CPU cores for evaluation")

    # Run evaluation
    if ia_file_path and os.path.exists(ia_file_path):
        results = cafa_eval(
            ontology_path,
            predictions_dir,
            ground_truth_path,
            ia_file_path,
            th_step=th_step,
        )
    else:
        results = cafa_eval(ontology_path, predictions_dir, ground_truth_path, th_step=th_step)

    return results


def extract_metrics_summary(results) -> Dict[str, float]:
    """
    Extract F1 and weighted F1 scores for each subontology and compute overall means.
    """
    evaluation_df, best_scores_dict = results
    metrics = {}

    # Use any of the best score dataframes (they all have the same structure)
    df = best_scores_dict.get("f", best_scores_dict.get("f_w"))
    if df is None:
        print("ERROR: No valid dataframe found in best_scores_dict")
        return metrics

    # Reset index to access namespace column
    df = df.reset_index()

    # Extract metrics for each namespace
    for ns in df["ns"].unique():
        ns_data = df[df["ns"] == ns].iloc[0]  # Get the row for this namespace
        metrics[f"{ns}_f1"] = ns_data["f"]
        metrics[f"{ns}_weighted_f1"] = ns_data["f_w"]

    # Compute overall means
    f1_values = [metrics[f"{ns}_f1"] for ns in df["ns"].unique()]
    fw1_values = [metrics[f"{ns}_weighted_f1"] for ns in df["ns"].unique()]

    metrics["overall_mean_f1"] = sum(f1_values) / len(f1_values)
    metrics["overall_mean_weighted_f1"] = sum(fw1_values) / len(fw1_values)

    # Print summary
    print("\nF1 scores by aspect:")
    for ns in df["ns"].unique():
        print(f"  {ns}: {metrics[f'{ns}_f1']:.4f}")
    print(f"  Overall mean: {metrics['overall_mean_f1']:.4f}")

    print("\nWeighted F1 scores by aspect:")
    for ns in df["ns"].unique():
        print(f"  {ns}: {metrics[f'{ns}_weighted_f1']:.4f}")
    print(f"  Overall mean: {metrics['overall_mean_weighted_f1']:.4f}")

    return metrics


def print_results_summary(metrics: Dict[str, float]):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("CAFA EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    # Print F1 scores
    print("\nF1 SCORES:")
    print("-" * 30)
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        if f"{aspect}_f1" in metrics:
            print(f"{aspect:25}: {metrics[f'{aspect}_f1']:.4f}")
    print(f"{'OVERALL AVERAGE':25}: {metrics['overall_mean_f1']:.4f}")

    # Print weighted F1 scores
    print("\nWEIGHTED F1 SCORES:")
    print("-" * 35)
    for aspect in ["biological_process", "molecular_function", "cellular_component"]:
        if f"{aspect}_weighted_f1" in metrics:
            print(f"{aspect:25}: {metrics[f'{aspect}_weighted_f1']:.4f}")
    print(f"{'OVERALL AVERAGE':25}: {metrics['overall_mean_weighted_f1']:.4f}")

    print("=" * 60)


def main():
    """Main pipeline for CAFA GO term evaluation."""

    # Initialize colorama for colored output
    init()

    parser = argparse.ArgumentParser(description="CAFA GO Term Evaluation Pipeline")
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Input directory containing chunk directories with JSON files",
    )
    parser.add_argument(
        "--ontology",
        "-o",
        required=True,
        help="Path to GO ontology file (go-basic.obo)",
    )
    parser.add_argument(
        "--ia_file",
        "-a",
        required=True,
        help="Path to Information Accretion file (IA.txt)",
    )
    parser.add_argument(
        "--output_dir",
        "-d",
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=0,
        help="Number of CPU threads to use (0 = all available, default: 0)",
    )
    parser.add_argument(
        "--summary_only",
        "-s",
        action="store_true",
        help="Extract GO terms only from summary sections",
    )
    parser.add_argument(
        "--reasoning_mode",
        "-r",
        action="store_true",
        help="Use reasoning evaluation mode: extract ground truth from leaf columns and filter predictions by aspects",
    )
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    GO_ONTOLOGY_PATH = args.ontology
    IA_FILE_PATH = args.ia_file
    OUTPUT_DIR = args.output_dir
    NUM_THREADS = args.threads
    SUMMARY_ONLY = args.summary_only
    REASONING_MODE = args.reasoning_mode

    print(f"{Fore.CYAN}Starting CAFA GO Term Evaluation Pipeline{Style.RESET_ALL}")
    print("-" * 40)

    # Print extraction mode
    extraction_mode = "summary sections only" if SUMMARY_ONLY else "all text"
    print(f"{Fore.YELLOW}GO term extraction mode for predictions: {extraction_mode}{Style.RESET_ALL}")
    if REASONING_MODE:
        print(f"{Fore.YELLOW}Reasoning evaluation mode: using leaf columns for ground truth with aspect filtering{Style.RESET_ALL}")
    print()

    # Load GO DAG if reasoning mode is enabled
    go_dag = None
    if REASONING_MODE:
        try:
            from goatools.obo_parser import GODag
            print(f"{Fore.CYAN}Loading GO ontology for aspect classification...{Style.RESET_ALL}")
            go_dag = GODag(GO_ONTOLOGY_PATH, optional_attrs={'relationship'})
            print(f"{Fore.GREEN}✓ GO ontology loaded with {len(go_dag)} terms{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.RED}ERROR: goatools not available. Install with: pip install goatools{Style.RESET_ALL}")
            return
        except Exception as e:
            print(f"{Fore.RED}ERROR loading GO ontology: {e}{Style.RESET_ALL}")
            return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Process JSON data from chunk directories
    predictions, ground_truth = process_json_data(INPUT_DIR, SUMMARY_ONLY, REASONING_MODE, go_dag)

    if not predictions:
        print(f"{Fore.RED}ERROR: No predictions found in the data!{Style.RESET_ALL}")
        return

    if not ground_truth:
        print(f"{Fore.RED}ERROR: No ground truth data found!{Style.RESET_ALL}")
        return

    # Step 2: Create CAFA format files
    predictions_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    prediction_file = os.path.join(predictions_dir, "llm_predictions.tsv")
    ground_truth_file = os.path.join(OUTPUT_DIR, "ground_truth.tsv")

    create_cafa_prediction_file(predictions, prediction_file)
    create_cafa_ground_truth_file(ground_truth, ground_truth_file)

    # Step 3: Run CAFA evaluation
    try:
        results = run_cafa_evaluation(
            GO_ONTOLOGY_PATH,
            predictions_dir,
            ground_truth_file,
            ia_file_path=IA_FILE_PATH,
            n_cpu=NUM_THREADS,
            th_step=0.99,  # since every predicted GO term score=1.0
        )

        # Step 4: Extract and display metrics
        metrics = extract_metrics_summary(results)
        print_results_summary(metrics)

        # Step 5: Save detailed results
        evaluation_df, best_scores_dict = results

        # Save main evaluation results
        evaluation_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.tsv"), sep="\t")
        print(f"\nEvaluation results saved to: {os.path.join(OUTPUT_DIR, 'evaluation_results.tsv')}")

        # Save best scores for each metric
        for metric, df in best_scores_dict.items():
            metric_path = os.path.join(OUTPUT_DIR, f"best_{metric}.tsv")
            df.to_csv(metric_path, sep="\t")
        print(f"Best score files saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"{Fore.RED}ERROR during evaluation: {e}{Style.RESET_ALL}")
        print("Please check that all input files exist and are in the correct format.")


if __name__ == "__main__":
    main()
