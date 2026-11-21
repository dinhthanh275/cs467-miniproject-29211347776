import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import CourseRecommenderEngine


def train_test_split_per_user(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a per-user test split (~20% of each learner's ratings) for offline evaluation.
    """
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, group in ratings_df.groupby("user_id"):
        n_total = len(group)
        n_test = max(1, int(round(n_total * test_ratio)))
        shuffled_idx = rng.permutation(group.index.values)
        test_idx = shuffled_idx[:n_test]
        train_idx = shuffled_idx[n_test:]

        train_parts.append(ratings_df.loc[train_idx])
        test_parts.append(ratings_df.loc[test_idx])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def evaluate_models(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 4.0
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate Popularity Baseline, NLP Content-Based, and Hybrid models using Precision@5,
    along with RMSE and MAE for User-Based Collaborative Filtering rating predictions.
    """
    train_df, test_df = train_test_split_per_user(ratings_df, test_ratio=0.2, random_state=42)
    engine = CourseRecommenderEngine(items_df, train_df)

    # 1. Compute RMSE and MAE on held-out test ratings
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        pred = engine.predict_rating_cf(row["user_id"], row["item_id"])
        y_true.append(float(row["rating"]))
        y_pred.append(pred)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    # 2. Compute Precision@5 across all three recommendation models
    precisions: Dict[str, List[float]] = {
        "Popularity Baseline": [],
        "NLP Content-Based": [],
        "Hybrid (NLP + User-CF)": [],
    }

    for user_id, user_test in test_df.groupby("user_id"):
        relevant_items = set(
            user_test[user_test["rating"] >= relevance_threshold]["item_id"]
        )
        if not relevant_items:
            continue

        pop_recs = set(engine.recommend_popularity(user_id=user_id, top_k=k, min_ratings=2)["item_id"])
        cb_recs = set(engine.recommend_content_based_user(user_id=user_id, top_k=k)["item_id"])
        hyb_recs = set(engine.recommend_hybrid(user_id=user_id, top_k=k)["item_id"])

        precisions["Popularity Baseline"].append(len(pop_recs & relevant_items) / k)
        precisions["NLP Content-Based"].append(len(cb_recs & relevant_items) / k)
        precisions["Hybrid (NLP + User-CF)"].append(len(hyb_recs & relevant_items) / k)

    summary_rows = []
    for model_name, scores in precisions.items():
        summary_rows.append({
            "Model": model_name,
            "Precision@5": round(float(np.mean(scores)), 4),
            "Evaluated_Users": len(scores),
        })

    metrics_df = pd.DataFrame(summary_rows)
    rating_errors = {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}
    return metrics_df, rating_errors


def generate_visualizations(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: str = "results"
) -> None:
    """
    Generate and save dataset distribution and model evaluation comparison plots in results/.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Rating distribution & course difficulty level distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    rating_counts = ratings_df["rating"].value_counts().sort_index()
    bars1 = axes[0].bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color="#3b82f6",
        edgecolor="black",
        width=0.55
    )
    axes[0].set_title("Learner Rating Distribution (1-5 Scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rating Score")
    axes[0].set_ylabel("Number of Ratings")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", fontsize=10)

    level_order = ["Beginner", "Intermediate", "Advanced"]
    level_counts = items_df["level"].value_counts().reindex(level_order).fillna(0)
    bars2 = axes[1].bar(
        level_counts.index,
        level_counts.values,
        color=["#10b981", "#f59e0b", "#ef4444"],
        edgecolor="black",
        width=0.5
    )
    axes[1].set_title("Course Catalog by Difficulty Level", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Course Level")
    axes[1].set_ylabel("Number of Courses")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{int(h)}", ha="center", fontsize=10)

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()

    # Chart 2: Precision@5 bar chart comparing recommendation models
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#94a3b8", "#0ea5e9", "#2563eb"]
    bars = ax.bar(
        metrics_df["Model"],
        metrics_df["Precision@5"],
        color=colors,
        edgecolor="black",
        width=0.5
    )
    ax.set_title("Offline Evaluation: Precision@5 Across Recommender Models", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision@5")
    ax.set_ylim(0, max(metrics_df["Precision@5"].max() * 1.25, 0.35))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "evaluation_comparison.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"Saved charts to:\n - {chart1_path}\n - {chart2_path}")


def run_all_test_cases_and_save(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    rating_errors: Dict[str, float],
    output_file: str = "results/sample_outputs.txt"
) -> str:
    """
    Execute all three required assignment test cases and save outputs to results/sample_outputs.txt.
    """
    engine = CourseRecommenderEngine(items_df, ratings_df)

    n_users = ratings_df["user_id"].nunique()
    n_items = items_df["item_id"].nunique()
    n_ratings = len(ratings_df)
    sparsity = 1.0 - (n_ratings / (n_users * n_items))

    lines = []
    lines.append("=" * 80)
    lines.append("CS467 MINI-PROJECT (MP-A14): ONLINE PROGRAMMING TUTORIAL & MOOC RECOMMENDER")
    lines.append("EVALUATION REPORT & REQUIRED TEST CASES OUTPUT")
    lines.append("=" * 80)
    lines.append(f"Dataset Summary: {n_items} courses | {n_users} learners | {n_ratings} ratings")
    lines.append(f"Interaction Matrix Sparsity (Missing Entries): {sparsity * 100:.2f}%\n")

    lines.append("-" * 80)
    lines.append("OFFLINE EVALUATION RESULTS (Hold-out 20% Test Split)")
    lines.append("-" * 80)
    lines.append(metrics_df.to_string(index=False))
    lines.append(
        f"\nUser-Based Collaborative Filtering Rating Prediction -> "
        f"RMSE: {rating_errors['RMSE']}, MAE: {rating_errors['MAE']}\n"
    )

    # TEST CASE 1: Goal query
    q1 = "beginner python data visualization and pandas project"
    lines.append("-" * 80)
    lines.append(f"TEST CASE 1 (Goal Query): '{q1}'")
    lines.append("-" * 80)
    tc1_res = engine.recommend_by_query(q1, top_k=5)
    lines.append(tc1_res.to_string(index=False) + "\n")

    # TEST CASE 2: Next-course test
    u_test = "U01"
    u_history = ratings_df[(ratings_df["user_id"] == u_test) & (ratings_df["rating"] >= 4)].merge(
        items_df[["item_id", "course_title"]], on="item_id"
    )
    lines.append("-" * 80)
    lines.append(f"TEST CASE 2 (Next-Course Test for Learner {u_test})")
    lines.append("Completed High-Rated Courses Sample: " + ", ".join(u_history["course_title"].head(5).tolist()))
    lines.append("-" * 80)
    lines.append("[A] NLP Content-Based Top-5 Next Courses:")
    lines.append(engine.recommend_content_based_user(u_test, top_k=5).to_string(index=False))
    lines.append("\n[B] Hybrid (NLP + User-CF) Top-5 Next Courses:")
    lines.append(engine.recommend_hybrid(u_test, top_k=5).to_string(index=False) + "\n")

    # TEST CASE 3: Cold-start test
    cold_course_id = "C040"
    cold_ratings_count = len(ratings_df[ratings_df["item_id"] == cold_course_id])
    cold_title = items_df[items_df["item_id"] == cold_course_id]["course_title"].values[0]
    q3 = "advanced generative ai agents llm and rag systems with langchain"
    lines.append("-" * 80)
    lines.append(
        f"TEST CASE 3 (Cold-Start Test): New Course '{cold_course_id} - {cold_title}' "
        f"(Ratings count = {cold_ratings_count})"
    )
    lines.append(f"Learner Query: '{q3}'")
    lines.append("-" * 80)
    tc3_res = engine.recommend_by_query(q3, top_k=5)
    lines.append(tc3_res.to_string(index=False) + "\n")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


if __name__ == "__main__":
    items = pd.read_csv("data/items.csv")
    ratings = pd.read_csv("data/ratings.csv")

    metrics_df, rating_errors = evaluate_models(items, ratings, k=5)
    generate_visualizations(items, ratings, metrics_df, output_dir="results")
    report = run_all_test_cases_and_save(items, ratings, metrics_df, rating_errors)
    print("\n" + report)