import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# Difficulty level and programming domain skill keywords
LEVEL_KEYWORDS: Dict[str, Set[str]] = {
    "beginner": {
        "beginner", "beginners", "basics", "basic", "fundamentals",
        "intro", "introduction", "essentials", "starter", "absolute"
    },
    "intermediate": {
        "intermediate", "practical", "bootcamp", "building",
        "patterns", "scalable", "engineering"
    },
    "advanced": {
        "advanced", "production", "distributed", "deep",
        "architecture", "orchestration", "enterprise", "scratch"
    },
}

DOMAIN_SKILLS: Set[str] = {
    "python", "pandas", "numpy", "matplotlib", "seaborn", "plotly", "dash",
    "sql", "postgresql", "database", "statistics", "probability",
    "machine-learning", "scikit-learn", "deep-learning", "pytorch", "neural-networks",
    "nlp", "transformers", "tf-idf", "computer-vision", "opencv", "yolo",
    "html", "css", "javascript", "es6", "react", "typescript", "nodejs", "express",
    "fastapi", "rest", "api", "fullstack", "frontend", "backend",
    "git", "github", "linux", "bash", "docker", "kubernetes", "cicd", "aws", "cloud", "devops",
    "java", "spring-boot", "csharp", "dotnet", "flutter", "dart", "android", "kotlin", "mobile",
    "algorithms", "data-structures", "interview", "system-design", "security", "pentesting",
    "time-series", "forecasting", "recommender-systems", "generative-ai", "llm", "rag", "langchain",
    "data-visualization", "data-analysis", "data-science", "automation", "oop", "testing", "project"
}


class CourseNLPPipeline:
    """
    NLP Preprocessing and TF-IDF Feature Extraction Pipeline for Online Programming Courses.
    """

    def __init__(self, max_features: int = 1500):
        # Retain short domain-relevant programming terms such as 'go' and 'c'
        custom_stop_words = set(ENGLISH_STOP_WORDS) - {"go", "c"}
        self.stop_words = custom_stop_words
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # Unigrams and bigrams
            max_features=max_features,
            sublinear_tf=True,
        )
        self.feature_names: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def preprocess_text(self, text: str) -> str:
        """
        Clean raw text: lowercase, normalize technical compound terms,
        strip punctuation/extra spaces, tokenize, and remove stop words.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()

        # Normalize common technical terms and multi-word phrases
        text = text.replace("c#", "csharp").replace(".net", "dotnet").replace("node.js", "nodejs")
        text = text.replace("scikit-learn", "scikitlearn").replace("data visualization", "data-visualization")
        text = text.replace("machine learning", "machine-learning").replace("deep learning", "deep-learning")

        # Remove special punctuation while preserving alphanumeric characters and hyphens
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()
        filtered_tokens = [
            tok for tok in tokens
            if tok not in self.stop_words and (len(tok) > 1 or tok in {"c", "r"})
        ]
        return " ".join(filtered_tokens)

    def extract_level_and_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract target difficulty levels and technical skill keywords from text or learner queries.
        """
        cleaned = self.preprocess_text(text)
        tokens = set(cleaned.split())
        normalized_skills = {s.replace("scikit-learn", "scikitlearn") for s in DOMAIN_SKILLS}

        detected_levels = []
        for lvl, kw_set in LEVEL_KEYWORDS.items():
            if tokens.intersection(kw_set):
                detected_levels.append(lvl.capitalize())

        detected_skills = sorted([
            tok.replace("scikitlearn", "scikit-learn")
            for tok in tokens.intersection(normalized_skills)
        ])

        return {
            "levels": detected_levels if detected_levels else ["Any"],
            "skills": detected_skills,
        }

    def build_course_document(self, row: pd.Series) -> str:
        """
        Combine title, level, topic tags (weighted 2x for keyword density), and curriculum summary.
        """
        title = str(row.get("course_title", ""))
        level = str(row.get("level", ""))
        tags = str(row.get("topic_tags", "")).replace(";", " ")
        summary = str(row.get("curriculum_summary", ""))

        raw_doc = f"{title} {level} {tags} {tags} {summary}"
        return self.preprocess_text(raw_doc)

    def fit_transform_courses(self, items_df: pd.DataFrame):
        """
        Fit the TF-IDF vectorizer on the course catalog and return processed DataFrame and TF-IDF matrix.
        """
        df = items_df.copy()
        df["cleaned_doc"] = df.apply(self.build_course_document, axis=1)
        df["extracted_skills"] = df["cleaned_doc"].apply(
            lambda txt: self.extract_level_and_skills(txt)["skills"]
        )

        tfidf_matrix = self.vectorizer.fit_transform(df["cleaned_doc"])
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return df, tfidf_matrix

    def transform_query(self, query_text: str):
        """
        Vectorize a new free-text query or new course description using the fitted vocabulary.
        """
        if not self.is_fitted:
            raise RuntimeError("NLP pipeline is not fitted yet. Call fit_transform_courses() first.")
        cleaned_query = self.preprocess_text(query_text)
        return self.vectorizer.transform([cleaned_query])

    def explain_matching_keywords(
        self,
        query_vector,
        item_vector,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Identify top n-gram features contributing most to the cosine similarity between query and course.
        """
        if not self.is_fitted:
            return []
        q_arr = query_vector.toarray().ravel()
        i_arr = item_vector.toarray().ravel()
        contributions = q_arr * i_arr

        top_indices = np.argsort(contributions)[::-1]
        matched = []
        for idx in top_indices:
            if contributions[idx] > 0 and len(matched) < top_n:
                matched.append((str(self.feature_names[idx]), round(float(contributions[idx]), 4)))
        return matched


if __name__ == "__main__":
    items = pd.read_csv("data/items.csv")
    nlp = CourseNLPPipeline()
    processed_df, matrix = nlp.fit_transform_courses(items)
    print(f"TF-IDF Matrix Shape: {matrix.shape}")

    sample_query = "beginner python data visualization and pandas project"
    q_vec = nlp.transform_query(sample_query)
    info = nlp.extract_level_and_skills(sample_query)
    print(f"Query: '{sample_query}'")
    print(f"Detected Level & Skills: {info}")
    print(f"Query Vector Non-zero Features: {q_vec.nnz}")