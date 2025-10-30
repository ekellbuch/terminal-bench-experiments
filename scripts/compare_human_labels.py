#!/usr/bin/env python3
"""
Compare model judge predictions against human ground-truth labels.

This script loads human annotations and judge predictions, then calculates
classification metrics to evaluate judge accuracy.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
)
import sys

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from utils.loading import (
    load_config_with_env,
    load_human_labels,
    load_judge_predictions,
)


def get_majority_vote(rater_scores: Dict[str, int]) -> int:
    """Calculate majority vote from multiple rater scores."""
    scores = list(rater_scores.values())
    if not scores:
        return 0
    return 1 if np.mean(scores) >= 0.5 else 0


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate classification metrics."""

    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0.0,
            "kappa": 0.0,
        }

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Handle single-class y_true/y_pred safely
    if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
        # Avoid division-by-zero warnings or undefined scores
        acc = accuracy_score(y_true, y_pred)
        return {
            "accuracy": acc,
            "precision": 0.0,
            "recall": 0.0,
            "kappa": 0.0,
            "f1": 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "support": len(y_true),
        }

    # Normal case
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": len(y_true),
    }


def compare_judge_to_human(
    judge_df: pd.DataFrame,
    human_majority_labels: Dict,
    judge_model: str = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compare a judge's predictions to human labels.

    (1) Get trials evaluated by both the judge and failure mode.
    (2) Get the human labels for each trial and failure mode.
    (3) Calculate the metrics for each failure mode.
    (4) Calculate the overall metrics.

    judge_model: If None, expects judge_df to be a single-judge dataframe (e.g., after majority vote).
    """
    # Step 1: Get the set of trial IDs that are present in BOTH judge predictions and human labels
    judge_trial_ids = set(judge_df["trial_id"].unique())
    human_trial_ids = set(human_majority_labels.keys())
    valid_trial_ids = judge_trial_ids & human_trial_ids
    if not valid_trial_ids:
        return {}

    print(f"Valid trial IDs for both judge and human labels: {len(valid_trial_ids)}")
    # Filter judge_df to relevant trials only
    judge_df = judge_df[judge_df["trial_id"].isin(valid_trial_ids)]
    # Filter human labels to relevant trials only
    human_labels = {
        trial_id: human_majority_labels[trial_id] for trial_id in valid_trial_ids
    }

    results = {}
    all_y_true = []
    all_y_pred = []

    failure_modes = sorted(
        m for m in judge_df["failure_mode"].unique() if pd.notnull(m)
    )

    for mode in failure_modes:
        mode_data = judge_df[judge_df["failure_mode"] == mode]

        y_true = []
        y_pred = []

        for _, row in mode_data.iterrows():
            trial_id = row["trial_id"]
            judge_score_float = float(row["score"])
            judge_label = 1 if judge_score_float >= threshold else 0

            # Use precomputed human majority label for this trial/mode
            human_modes = human_labels.get(trial_id, {})
            if mode not in human_modes:
                continue
            human_label = int(human_modes[mode])

            y_true.append(human_label)
            y_pred.append(judge_label)
            all_y_true.append(human_label)
            all_y_pred.append(judge_label)

        if y_true:
            metrics = calculate_metrics(y_true, y_pred)
            metrics["n_trials"] = len(y_true)
            results[mode] = metrics

    # Calculate overall metrics
    if all_y_true:
        metrics = calculate_metrics(all_y_true, all_y_pred)
        metrics["n_trials"] = len(all_y_true)
        try:
            metrics["kappa"] = cohen_kappa_score(all_y_true, all_y_pred)
        except Exception:
            metrics["kappa"] = float("nan")
        results["OVERALL"] = metrics

    return results


def collect_disagreements(
    judge_df: pd.DataFrame,
    human_majority_labels: Dict[str, Dict[str, int]],
    judge_model: str,
    threshold: float,
) -> List[Dict[str, object]]:
    """Collect per-trial, per-mode disagreements between a judge and human labels.

    Returns a list of rows with: trial_id, failure_mode, judge_model, judge_score,
    judge_label, human_label.
    """
    disagreements: List[Dict[str, object]] = []

    # Only keep trials present in human labels
    valid_trial_ids = set(judge_df["trial_id"].unique()) & set(
        human_majority_labels.keys()
    )
    if not valid_trial_ids:
        return disagreements

    filtered = judge_df[judge_df["trial_id"].isin(valid_trial_ids)]

    for _, row in filtered.iterrows():
        trial_id = row["trial_id"]
        mode = row["failure_mode"]
        if pd.isna(mode):
            continue
        human_modes = human_majority_labels.get(trial_id, {})
        if mode not in human_modes:
            continue

        judge_score_float = float(row["score"])
        judge_label = 1 if judge_score_float >= threshold else 0
        human_label = int(human_modes[mode])

        if judge_label != human_label:
            disagreements.append(
                {
                    "trial_id": trial_id,
                    "failure_mode": mode,
                    "judge_model": judge_model,
                    "judge_score": judge_score_float,
                    "judge_label": judge_label,
                    "human_label": human_label,
                }
            )

    return disagreements


def compute_human_majority_labels(human_labels: Dict) -> Dict[str, Dict[str, int]]:
    """Compute per-trial, per-mode majority labels from raw human annotations.

    Input format: trial_id -> rater -> mode -> score
    Output format: trial_id -> mode -> majority_label (0/1)
    """
    majority: Dict[str, Dict[str, int]] = {}
    for trial_id, rater_map in human_labels.items():
        mode_to_scores: Dict[str, List[int]] = {}
        for _rater, modes in rater_map.items():
            for mode, score in modes.items():
                # if trial_id == '0b1249b7-84df-4604-95c0-94a8ac7f2cad': breakpoint()
                mode_to_scores.setdefault(mode, []).append(int(score))
        if mode_to_scores:
            majority[trial_id] = {}
            for mode, scores in mode_to_scores.items():
                majority[trial_id][mode] = 1 if np.mean(scores) >= 0.5 else 0
    return majority


def print_results(results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print comparison results in a formatted table."""

    for judge, metrics in results.items():
        # Special formatting for majority vote
        if judge == "JUDGE_MAJORITY_VOTE":
            print(f"\n{'=' * 60}")
            print("Judge Majority Vote vs Human Labels")
            print(f"{'=' * 60}")
        else:
            print(f"\n{'=' * 60}")
            print(f"Judge: {judge}")
            print(f"{'=' * 60}")

        # Print header
        print(
            f"{'Mode':<10} {'Acc':<7} {'Prec':<7} {'Rec':<7} {'F1':<7} {'Kappa':<7}{'Support':<8}"
        )
        print(f"{'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")

        # Sort modes, putting OVERALL last
        modes = sorted([m for m in metrics.keys() if m != "OVERALL"])
        if "OVERALL" in metrics:
            modes.append("OVERALL")

        for mode in modes:
            m = metrics[mode]
            if mode == "OVERALL":
                print(
                    f"{'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}"
                )

            print(
                f"{mode:<10} {m['accuracy']:<7.2f} {m['precision']:<7.2f} "
                f"{m['recall']:<7.2f} {m['f1']:<7.2f} {m['kappa']:<7.2f} {m['support']:<8}"
            )

        # Print confusion matrix for overall
        if "OVERALL" in metrics:
            m = metrics["OVERALL"]
            print("\nConfusion Matrix:")
            print("              Predicted")
            print("              No    Yes")
            print(f"Actual  No  [{m['tn']:4} {m['fp']:4}]")
            print(f"        Yes [{m['fn']:4} {m['tp']:4}]")

            if "kappa" in m:
                print(f"\nCohen's Kappa: {m['kappa']:.3f}")


def calculate_judge_majority_vote(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate majority vote across all judges for each trial and failure mode."""

    # Group by trial_id and failure_mode, then calculate majority vote
    majority_votes = []

    for (trial_id, failure_mode), group in judge_df.groupby(
        ["trial_id", "failure_mode"]
    ):
        scores = group["score"].tolist()

        # Calculate majority vote (1 if mean >= 0.5, else 0)
        majority_score = 1 if np.mean(scores) >= 0.5 else 0

        # Count votes
        vote_counts = group["score"].value_counts().to_dict()
        total_votes = len(scores)

        majority_votes.append(
            {
                "trial_id": trial_id,
                "failure_mode": failure_mode,
                "majority_score": majority_score,
                "total_votes": total_votes,
                "votes_for_1": vote_counts.get(1, 0),
                "votes_for_0": vote_counts.get(0, 0),
                "agreement": max(vote_counts.get(1, 0), vote_counts.get(0, 0))
                / total_votes,
            }
        )

    return pd.DataFrame(majority_votes)


def calculate_judge_agreement(judge_df: pd.DataFrame):
    """Calculate inter-judge agreement statistics."""

    print("\n" + "=" * 60)
    print("Inter-Judge Agreement Analysis")
    print("=" * 60)

    # Get unique judges
    judges = sorted(judge_df["judge_model"].unique())

    if len(judges) < 2:
        print("Insufficient judges for agreement analysis")
        return

    print(f"Number of judges: {len(judges)}")
    print(f"Judges: {', '.join(judges)}")

    # Calculate pairwise agreement
    print("\nPairwise Agreement (Cohen's Kappa):")
    print("-" * 40)

    for i, judge1 in enumerate(judges):
        for judge2 in judges[i + 1 :]:
            # Collect paired predictions
            paired_scores_1 = []
            paired_scores_2 = []

            # Group by trial and failure mode
            for (trial_id, failure_mode), group in judge_df.groupby(
                ["trial_id", "failure_mode"]
            ):
                judge1_scores = group[group["judge_model"] == judge1]["score"].tolist()
                judge2_scores = group[group["judge_model"] == judge2]["score"].tolist()

                if judge1_scores and judge2_scores:
                    # Take first score if multiple (shouldn't happen with proper data)
                    paired_scores_1.append(judge1_scores[0])
                    paired_scores_2.append(judge2_scores[0])

            if paired_scores_1:
                kappa = cohen_kappa_score(paired_scores_1, paired_scores_2)
                agreement = np.mean(
                    [s1 == s2 for s1, s2 in zip(paired_scores_1, paired_scores_2)]
                )

                print(
                    f"{judge1} - {judge2}: κ = {kappa:.3f} "
                    f"(Agreement: {agreement:.1%}, n={len(paired_scores_1)})"
                )

    # Calculate overall agreement statistics
    print("\nOverall Judge Agreement:")
    print("-" * 30)

    # Calculate majority vote agreement for each trial/mode
    majority_df = calculate_judge_majority_vote(judge_df)

    if not majority_df.empty:
        avg_agreement = majority_df["agreement"].mean()
        perfect_agreement = (majority_df["agreement"] == 1.0).sum()
        total_cases = len(majority_df)

        print(f"Average agreement: {avg_agreement:.1%}")
        print(
            f"Perfect agreement cases: {perfect_agreement}/{total_cases} ({perfect_agreement / total_cases:.1%})"
        )

        # Show cases with disagreement
        disagreement_cases = majority_df[majority_df["agreement"] < 1.0]
        if not disagreement_cases.empty:
            print("\nCases with disagreement:")
            for _, row in disagreement_cases.head(10).iterrows():
                print(
                    f"  {row['trial_id']} - {row['failure_mode']}: "
                    f"{row['votes_for_1']} for 1, {row['votes_for_0']} for 0 "
                    f"(agreement: {row['agreement']:.1%})"
                )


def _build_rater_entity_labels(human_labels: Dict) -> Dict[str, Dict[tuple, int]]:
    """Build mapping: rater -> {(trial_id, mode): label}."""
    entity_to_labels: Dict[str, Dict[tuple, int]] = {}
    for trial_id, by_rater in human_labels.items():
        for rater, modes in by_rater.items():
            target = entity_to_labels.setdefault(rater, {})
            for mode, score in modes.items():
                breakpoint()
                target[(trial_id, mode)] = int(score)
    return entity_to_labels


def _build_judge_entity_labels(
    judge_df: pd.DataFrame, threshold: float = 0.5
) -> Dict[str, Dict[tuple, int]]:
    """Build mapping: judge_model -> {(trial_id, mode): label}."""
    entity_to_labels: Dict[str, Dict[tuple, int]] = {}
    for judge, group in judge_df.groupby("judge_model"):
        target: Dict[tuple, int] = {}
        for _, row in group.iterrows():
            trial_id = row["trial_id"]
            mode = row["failure_mode"]
            if pd.isna(mode):
                continue
            score = float(row["score"])
            target[(trial_id, mode)] = 1 if score >= threshold else 0
        entity_to_labels[judge] = target
    return entity_to_labels


def calculate_pairwise_agreement(
    entity_to_labels: Dict[str, Dict[tuple, int]], title: str
):
    """Generic pairwise agreement (Cohen's kappa and raw agreement) across entities.

    entity_to_labels maps entity -> {(trial_id, mode): label}.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    entities = sorted(entity_to_labels.keys())
    if len(entities) < 2:
        print("Insufficient entities for agreement analysis")
        return

    print(f"Number of entities: {len(entities)}")
    print(f"Entities: {', '.join(entities)}")

    print("\nPairwise Agreement (Cohen's Kappa):")
    print("-" * 40)

    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            labels1 = entity_to_labels[e1]
            labels2 = entity_to_labels[e2]
            common_keys = list(set(labels1.keys()) & set(labels2.keys()))
            if not common_keys:
                continue
            y1 = [labels1[k] for k in common_keys]
            y2 = [labels2[k] for k in common_keys]
            kappa = cohen_kappa_score(y1, y2)
            agreement = float(np.mean(np.array(y1) == np.array(y2)))
            print(
                f"{e1} - {e2}: κ = {kappa:.3f} (Agreement: {agreement:.1%}, n={len(common_keys)})"
            )


def main(args):
    # Convert paths to absolute
    human_labels_path = Path(args.human_labels)
    if not human_labels_path.is_absolute():
        human_labels_path = BASE_DIR / human_labels_path

    # Determine judge database path from config
    judge_config_path = Path(args.judge_config)
    if not judge_config_path.is_absolute():
        judge_config_path = BASE_DIR / judge_config_path

    if not judge_config_path.exists():
        print(f"Error: Judge config file not found: {judge_config_path}")
        sys.exit(1)

    # load yamkl
    config = load_config_with_env(judge_config_path)

    # Extract database path from config
    judge_db_path = Path(config.get("database_path"))

    print("=" * 60)
    print("Human-Judge Comparison Analysis")
    print("=" * 60)
    print(f"Human labels: {human_labels_path.name}")
    print(f"Judge config: {args.judge_config}")
    print(f"Judge database: {judge_db_path.name}")
    if args.debug:
        print("Debug mode: ENABLED")

    # Load human annotations
    human_labels = load_human_labels(human_labels_path)
    trial_ids = list[str](human_labels.keys())

    print(f"\nHuman annotations: {len(trial_ids)} trials")

    # Count unique raters
    all_raters = set()
    for trial_data in human_labels.values():
        all_raters.update(trial_data.keys())
    print(f"Unique raters: {len(all_raters)}")

    # Load judge predictions
    judge_df = load_judge_predictions(judge_db_path, trial_ids)

    judges = judge_df["judge_model"].unique()
    print(f"Judges found: {', '.join(judges)}")

    if args.debug:
        print("\n=== DEBUG: Data Summary ===")
        print(f"Human trials labeled: {len(trial_ids)}")
        print(f"Judge predictions labeled: {judge_df.shape}")
        print(f"Judge columns: {list(judge_df.columns)}")
        print(
            f"Unique failure modes in judge data: {sorted(judge_df['failure_mode'].unique())}"
        )

    # Analyze inter-rater agreement if requested
    if args.analyze_rater_agreement:
        rater_map = _build_rater_entity_labels(human_labels)
        calculate_pairwise_agreement(rater_map, "Inter-Rater Agreement Analysis")

    # Analyze inter-judge agreement if requested
    if args.analyze_judge_agreement:
        judge_map = _build_judge_entity_labels(judge_df, threshold=args.threshold)
        calculate_pairwise_agreement(judge_map, "Inter-Judge Agreement Analysis")

    # Compute majority vote outside: per trial, per mode
    human_majority_labels = compute_human_majority_labels(human_labels)

    # Compare judge predictions to human labels
    results = {}
    for judge in judges:
        judge_df_single = judge_df[judge_df["judge_model"] == judge]
        results[judge] = compare_judge_to_human(
            judge_df_single, human_majority_labels, judge, threshold=args.threshold
        )

    # Compute judge majority vote if requested
    if args.include_majority_vote and len(judges) > 1:
        print("\n" + "=" * 60)
        print("Judge Majority Vote Analysis")
        print("=" * 60)
        # compute majority vote for each trial and failure mode among judges
        judge_df2 = calculate_judge_majority_vote(judge_df)
        # Rename majority_score to score to match compare_judge_to_human expectations
        if "majority_score" in judge_df2.columns and "score" not in judge_df2.columns:
            judge_df2 = judge_df2.rename(columns={"majority_score": "score"})
        majority_results = compare_judge_to_human(judge_df2, human_majority_labels)
        results["JUDGE_MAJORITY_VOTE"] = majority_results

    # Print results
    print_results(results)

    if args.debug:
        # Build disagreements across all judges
        all_disagreements: List[Dict[str, object]] = []
        for judge in judges:
            judge_df_single = judge_df[judge_df["judge_model"] == judge]
            all_disagreements.extend(
                collect_disagreements(
                    judge_df_single,
                    human_majority_labels,
                    judge,
                    threshold=args.threshold,
                )
            )

        if all_disagreements:
            dis_df = pd.DataFrame(all_disagreements)
            # Count disagreements per trial and sort desc
            trial_counts = (
                dis_df.groupby("trial_id")
                .size()
                .reset_index(name="disagreement_count")
                .sort_values("disagreement_count", ascending=False)
            )

            print("\n" + "=" * 60)
            print("DEBUG: Judge vs Human Disagreements (sorted by trial)")
            print("=" * 60)

            for _, row in trial_counts.iterrows():
                tid = row["trial_id"]
                count = int(row["disagreement_count"])
                print(f"\nTrial ID: {tid}")
                print(f"Total disagreements: {count}")
                print("-" * 60)

                trial_rows = dis_df[dis_df["trial_id"] == tid].sort_values(
                    ["failure_mode", "judge_model"]
                )
                for _, r in trial_rows.iterrows():
                    print(
                        f"  Mode {r['failure_mode']}: {r['judge_model']}={r['judge_score']:.1f}, HUMAN={r['human_label']}"
                    )
        else:
            print("\nNo judge-human disagreements found.")

    # Save to CSV if requested
    if args.output_csv:
        csv_data = []
        for judge, metrics in results.items():
            for mode, values in metrics.items():
                row = {"judge": judge, "failure_mode": mode}
                row.update(values)
                csv_data.append(row)

        df = pd.DataFrame(csv_data)
        output_path = BASE_DIR / args.output_csv
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare model judge predictions against human labels"
    )
    parser.add_argument(
        "--human-labels",
        type=str,
        default="config/human_labels_tb0.yaml",
        help="Path to YAML file with human annotations",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="config/failure_debug_t0_d1.yaml",
        help="Path to judge configuration file",
    )
    parser.add_argument(
        "--analyze-rater-agreement",
        action="store_true",
        help="Analyze inter-rater agreement",
    )
    parser.add_argument(
        "--analyze-judge-agreement",
        action="store_true",
        help="Analyze inter-judge agreement",
    )
    parser.add_argument(
        "--include-majority-vote",
        # action="store_true",
        default=True,
        help="Include judge majority vote analysis",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize judge scores (default: 0.5)",
    )
    parser.add_argument(
        "--output-csv", type=str, help="Optional: Save results to CSV file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show detailed analysis information",
    )

    args = parser.parse_args()
    main(args)
