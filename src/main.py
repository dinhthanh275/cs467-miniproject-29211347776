import argparse
import os
import sys

import pandas as pd

# Ensure local imports work regardless of current working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from recommender import CourseRecommenderEngine


def print_banner(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def run_required_demos(engine: CourseRecommenderEngine) -> None:
    """
    Run all three required demonstration test cases from the assignment specification.
    """
    # Test Case 1: Study Goal Query
    q1 = "beginner python data visualization and pandas project"
    print_banner(f"DEMO TEST CASE 1 | Study Goal Query: '{q1}'")
    print(engine.recommend_by_query(q1, top_k=5).to_string(index=False))

    # Test Case 2: Next-Course Recommendation for Learner U01
    user_id = "U01"
    print_banner(f"DEMO TEST CASE 2 | Next-Course Recommendations for Learner '{user_id}'")
    print("--- [A] NLP Content-Based Recommendations ---")
    print(engine.recommend_content_based_user(user_id, top_k=5).to_string(index=False))
    print("\n--- [B] Hybrid (NLP + User-Based CF) Recommendations ---")
    print(engine.recommend_hybrid(user_id, top_k=5).to_string(index=False))

    # Test Case 3: Cold-Start Course Verification (C040 has 0 ratings)
    q3 = "advanced generative ai agents llm and rag systems with langchain"
    print_banner("DEMO TEST CASE 3 | Cold-Start Course Test (C040 - 0 Ratings)")
    print(f"Query: '{q3}'\n")
    print(engine.recommend_by_query(q3, top_k=5).to_string(index=False))


def interactive_cli(engine: CourseRecommenderEngine) -> None:
    """
    Interactive command-line interface allowing the instructor or user to test
    custom learner IDs or free-text study goal queries live.
    """
    valid_users = sorted(engine.ratings_df["user_id"].unique())
    while True:
        print("\n" + "-" * 60)
        print("CS467 Course Recommender Interactive Demo")
        print("  [1] Run all 3 required assignment test cases")
        print(f"  [2] Recommend next courses by Learner ID ({valid_users[0]}..{valid_users[-1]})")
        print("  [3] Recommend courses by custom study goal query")
        print("  [4] Add a new cold-start course & test query matching")
        print("  [0] Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            run_required_demos(engine)
        elif choice == "2":
            uid = input(f"Enter Learner ID (e.g., U01, U04, U08): ").strip().upper()
            if uid not in valid_users:
                print(f"Unknown user '{uid}'. Showing Popularity Baseline instead:")
                print(engine.recommend_popularity(top_k=5).to_string(index=False))
            else:
                print(f"\nTop-5 Hybrid Recommendations for {uid}:")
                print(engine.recommend_hybrid(uid, top_k=5).to_string(index=False))
        elif choice == "3":
            query = input("Enter your study goal (e.g., 'intermediate react typescript web api'): ").strip()
            if query:
                detected = engine.nlp.extract_level_and_skills(query)
                print(f"Detected Level & Skills: {detected}\n")
                print(engine.recommend_by_query(query, top_k=5).to_string(index=False))
        elif choice == "4":
            cid = input("New Course ID (e.g., C041): ").strip().upper() or "C041"
            title = input("Course Title: ").strip() or "Rust Systems Programming for Beginners"
            level = input("Level (Beginner/Intermediate/Advanced): ").strip().capitalize() or "Beginner"
            tags = input("Topic Tags (semicolon-separated, e.g., rust;systems;beginner): ").strip() or "rust;systems;beginner"
            summary = input("Curriculum Summary: ").strip() or "Learn Rust ownership borrowing memory safety and CLI tools from scratch."
            engine.add_new_course({
                "item_id": cid,
                "course_title": title,
                "level": level,
                "duration_hours": 15,
                "curriculum_summary": summary,
                "topic_tags": tags,
            })
            print(f"Added cold-start course '{cid}: {title}' (0 ratings). Querying it now...")
            print(engine.recommend_by_query(f"{level} {title} {tags.replace(';', ' ')}", top_k=5).to_string(index=False))
        elif choice == "0":
            print("Exiting demo. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CS467 Mini-Project (MP-A14): Online Programming Tutorial and MOOC Recommender"
    )
    parser.add_argument("--user", type=str, help="Get Top-5 next-course recommendations for a learner ID (e.g., U01)")
    parser.add_argument("--query", type=str, help="Get Top-5 course recommendations for a free-text study goal")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI menu")
    args = parser.parse_args()

    items_path = os.path.join(CURRENT_DIR, "..", "data", "items.csv")
    ratings_path = os.path.join(CURRENT_DIR, "..", "data", "ratings.csv")

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)
    engine = CourseRecommenderEngine(items_df, ratings_df)

    if args.query:
        print_banner(f"Custom Query Recommendations: '{args.query}'")
        print(engine.recommend_by_query(args.query, top_k=5).to_string(index=False))
    elif args.user:
        uid = args.user.strip().upper()
        print_banner(f"Next-Course Hybrid Recommendations for Learner '{uid}'")
        print(engine.recommend_hybrid(uid, top_k=5).to_string(index=False))
    elif args.interactive:
        interactive_cli(engine)
    else:
        run_required_demos(engine)


if __name__ == "__main__":
    main()