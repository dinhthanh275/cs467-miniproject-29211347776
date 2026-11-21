from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import CourseNLPPipeline


class CourseRecommenderEngine:
    """
    Course Recommendation Engine combining three approaches:
    1. Popularity Baseline (Bayesian Weighted Rating)
    2. NLP Content-Based Recommender (TF-IDF + Cosine Similarity)
    3. User-Based Collaborative Filtering & Hybrid Recommender Extension
    """

    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()

        # Initialize and fit the NLP pipeline
        self.nlp = CourseNLPPipeline()
        self.items_df, self.tfidf_matrix = self.nlp.fit_transform_courses(self.items_df)

        # Map item_id <-> matrix row index
        self.item_id_to_idx: Dict[str, int] = {
            row["item_id"]: idx for idx, row in self.items_df.iterrows()
        }
        self.idx_to_item_id: Dict[int, str] = {
            idx: row["item_id"] for idx, row in self.items_df.iterrows()
        }

        # Build user-item rating matrix and user-user similarity matrix
        self._build_interaction_matrix()

    def _build_interaction_matrix(self) -> None:
        all_users = sorted(self.ratings_df["user_id"].unique())
        all_items = list(self.items_df["item_id"])

        pivot = self.ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).reindex(index=all_users, columns=all_items)

        self.user_item_matrix = pivot
        self.user_means = pivot.mean(axis=1)

        # Mean-center ratings per user before computing cosine similarity for User-Based CF
        centered = pivot.sub(self.user_means, axis=0).fillna(0.0)
        sim_matrix = cosine_similarity(centered.values)
        self.user_similarity_df = pd.DataFrame(
            sim_matrix, index=all_users, columns=all_users
        )

    def add_new_course(self, new_course: Dict) -> None:
        """
        Add a newly uploaded course (0 ratings - cold-start item) to the catalog and refit NLP vectors.
        """
        if new_course["item_id"] in self.item_id_to_idx:
            return
        new_row_df = pd.DataFrame([new_course])
        self.items_df = pd.concat([self.items_df, new_row_df], ignore_index=True)
        self.items_df, self.tfidf_matrix = self.nlp.fit_transform_courses(self.items_df)
        self.item_id_to_idx = {
            row["item_id"]: idx for idx, row in self.items_df.iterrows()
        }
        self.idx_to_item_id = {
            idx: row["item_id"] for idx, row in self.items_df.iterrows()
        }
        self._build_interaction_matrix()

    def get_user_rated_items(self, user_id: str) -> set:
        if user_id not in self.ratings_df["user_id"].values:
            return set()
        return set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

    # ------------------------------------------------------------------
    # 1. POPULARITY BASELINE
    # ------------------------------------------------------------------
    def recommend_popularity(
        self,
        user_id: Optional[str] = None,
        top_k: int = 5,
        min_ratings: int = 3
    ) -> pd.DataFrame:
        """
        Rank courses using a Bayesian weighted average rating with a minimum rating count threshold,
        excluding courses already rated by the target user.
        """
        stats = (
            self.ratings_df.groupby("item_id")["rating"]
            .agg(rating_count="count", avg_rating="mean")
            .reset_index()
        )
        global_mean = self.ratings_df["rating"].mean()

        merged = self.items_df.merge(stats, on="item_id", how="left")
        merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
        merged["avg_rating"] = merged["avg_rating"].fillna(0.0)

        qualified = merged[merged["rating_count"] >= min_ratings].copy()

        # Weighted rating formula: (v / (v + m)) * R + (m / (v + m)) * C
        v = qualified["rating_count"]
        m = float(min_ratings)
        R = qualified["avg_rating"]
        C = global_mean
        qualified["score"] = (v / (v + m)) * R + (m / (v + m)) * C

        if user_id:
            seen = self.get_user_rated_items(user_id)
            qualified = qualified[~qualified["item_id"].isin(seen)]

        qualified = qualified.sort_values(
            by=["score", "rating_count"], ascending=False
        ).head(top_k)

        qualified["explanation"] = qualified.apply(
            lambda r: f"Avg {r['avg_rating']:.2f}/5.0 ({r['rating_count']} ratings)",
            axis=1,
        )
        return qualified[
            ["item_id", "course_title", "level", "score", "explanation"]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. NLP CONTENT-BASED RECOMMENDER (Query & User Profile)
    # ------------------------------------------------------------------
    def recommend_by_query(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """
        Recommend top-k courses matching a learner's free-text study goal query.
        Handles cold-start courses with zero ratings seamlessly via curriculum text.
        """
        q_vec = self.nlp.transform_query(query_text)
        query_info = self.nlp.extract_level_and_skills(query_text)
        target_levels = set(query_info["levels"])

        cos_scores = cosine_similarity(q_vec, self.tfidf_matrix).ravel()

        results = []
        for idx, row in self.items_df.iterrows():
            base_sim = float(cos_scores[idx])
            level_bonus = 0.05 if (row["level"] in target_levels and base_sim > 0) else 0.0
            final_score = base_sim + level_bonus

            matched_kws = self.nlp.explain_matching_keywords(
                q_vec, self.tfidf_matrix[idx], top_n=4
            )
            kw_str = ", ".join([kw for kw, _ in matched_kws]) if matched_kws else "general match"

            results.append({
                "item_id": row["item_id"],
                "course_title": row["course_title"],
                "level": row["level"],
                "score": round(final_score, 4),
                "explanation": f"Matched keywords: [{kw_str}]",
            })

        res_df = pd.DataFrame(results).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)

    def _build_user_profile_vector(self, user_id: str) -> Optional[csr_matrix]:
        """
        Construct a user preference vector as the rating-weighted average of rated course TF-IDF vectors.
        """
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return None

        weights = []
        vectors = []
        for _, row in user_ratings.iterrows():
            iid = row["item_id"]
            if iid in self.item_id_to_idx:
                idx = self.item_id_to_idx[iid]
                w = max(float(row["rating"]) - 2.5, 0.2)
                weights.append(w)
                vectors.append(self.tfidf_matrix[idx].toarray().ravel())

        if not vectors:
            return None

        profile_arr = np.average(np.vstack(vectors), axis=0, weights=weights)
        return csr_matrix(profile_arr.reshape(1, -1))

    def recommend_content_based_user(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_seen: bool = True
    ) -> pd.DataFrame:
        """
        Recommend next courses for a learner based on their NLP user profile vector.
        """
        profile_vec = self._build_user_profile_vector(user_id)
        if profile_vec is None:
            return self.recommend_popularity(user_id=user_id, top_k=top_k)

        cos_scores = cosine_similarity(profile_vec, self.tfidf_matrix).ravel()
        seen = self.get_user_rated_items(user_id) if exclude_seen else set()

        results = []
        for idx, row in self.items_df.iterrows():
            if row["item_id"] in seen:
                continue
            sim = float(cos_scores[idx])
            matched_kws = self.nlp.explain_matching_keywords(
                profile_vec, self.tfidf_matrix[idx], top_n=4
            )
            kw_str = ", ".join([kw for kw, _ in matched_kws]) if matched_kws else "profile similarity"

            results.append({
                "item_id": row["item_id"],
                "course_title": row["course_title"],
                "level": row["level"],
                "score": round(sim, 4),
                "explanation": f"Profile keywords: [{kw_str}]",
            })

        return (
            pd.DataFrame(results)
            .sort_values(by="score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 3. PROJECT-SPECIFIC EXTENSION: USER-BASED CF & HYBRID RECOMMENDER
    # ------------------------------------------------------------------
    def predict_rating_cf(self, user_id: str, item_id: str, k_neighbors: int = 5) -> float:
        """
        Predict numeric rating (1-5) for (user_id, item_id) using User-Based Collaborative Filtering.
        """
        global_mean = float(self.ratings_df["rating"].mean())
        if user_id not in self.user_item_matrix.index:
            return global_mean

        u_mean = float(self.user_means.loc[user_id])
        if item_id not in self.user_item_matrix.columns:
            return u_mean

        item_ratings = self.user_item_matrix[item_id].dropna()
        item_ratings = item_ratings[item_ratings.index != user_id]
        if item_ratings.empty:
            return u_mean

        sims = self.user_similarity_df.loc[user_id, item_ratings.index]
        positive_sims = sims[sims > 0].sort_values(ascending=False).head(k_neighbors)
        if positive_sims.empty:
            return u_mean

        neighbor_ids = positive_sims.index
        neighbor_ratings = item_ratings.loc[neighbor_ids]
        neighbor_means = self.user_means.loc[neighbor_ids]

        numerator = np.sum(positive_sims.values * (neighbor_ratings.values - neighbor_means.values))
        denominator = np.sum(np.abs(positive_sims.values))
        pred = u_mean + (numerator / denominator if denominator > 0 else 0.0)
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_hybrid(
        self,
        user_id: str,
        top_k: int = 5,
        alpha: float = 0.55,
        exclude_seen: bool = True
    ) -> pd.DataFrame:
        """
        Hybrid Recommender combining NLP Content-Based similarity (weight alpha)
        and normalized User-Based Collaborative Filtering predictions (weight 1 - alpha).
        """
        profile_vec = self._build_user_profile_vector(user_id)
        if profile_vec is None:
            return self.recommend_popularity(user_id=user_id, top_k=top_k)

        cos_scores = cosine_similarity(profile_vec, self.tfidf_matrix).ravel()
        seen = self.get_user_rated_items(user_id) if exclude_seen else set()

        results = []
        for idx, row in self.items_df.iterrows():
            iid = row["item_id"]
            if iid in seen:
                continue

            cb_score = float(cos_scores[idx])
            cf_pred_rating = self.predict_rating_cf(user_id, iid)
            cf_norm = (cf_pred_rating - 1.0) / 4.0

            hybrid_score = alpha * cb_score + (1.0 - alpha) * cf_norm
            matched_kws = self.nlp.explain_matching_keywords(
                profile_vec, self.tfidf_matrix[idx], top_n=3
            )
            kw_str = ", ".join([kw for kw, _ in matched_kws]) if matched_kws else "general"

            results.append({
                "item_id": iid,
                "course_title": row["course_title"],
                "level": row["level"],
                "score": round(hybrid_score, 4),
                "explanation": (
                    f"Hybrid (NLP={cb_score:.2f}, CF_pred={cf_pred_rating:.2f}/5) | "
                    f"Keywords: [{kw_str}]"
                ),
            })

        return (
            pd.DataFrame(results)
            .sort_values(by="score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    items = pd.read_csv("data/items.csv")
    ratings = pd.read_csv("data/ratings.csv")
    engine = CourseRecommenderEngine(items, ratings)

    print("=== Top 5 Popular Courses ===")
    print(engine.recommend_popularity(top_k=5).to_string(index=False))

    print("\nQuery: 'beginner python data visualization and pandas project'")
    print(engine.recommend_by_query("beginner python data visualization and pandas project", top_k=5).to_string(index=False))

    print("\nHybrid Next-Course Recommendations for U01")
    print(engine.recommend_hybrid("U01", top_k=5).to_string(index=False))