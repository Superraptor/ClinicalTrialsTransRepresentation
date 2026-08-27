"""
Unsupervised Clustering and Post-Clustering Characterization Pipeline
Clusters clinical trials based on semantic and lexical representations (TF-IDF & Embeddings)
and characterizes each emergent cluster using c-TF-IDF, representative exemplars, and metadata profiles.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

# Optional imports for advanced embeddings and clustering
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class TrialClusteringEngine:
    """
    Unsupervised clustering engine for ClinicalTrials.gov study records.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        min_cluster_size: int = 15,
        use_embeddings: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state
        self.use_embeddings = use_embeddings
        self.embedding_model: Optional[Any] = None

        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {embedding_model_name}...")
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.warning(f"SentenceTransformer not available ({e}). Falling back to TF-IDF.")
                self.use_embeddings = False

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=2,
        )
        self.umap_model_2d = None
        self.umap_model_cluster = None
        self.cluster_model = None

    def prepare_corpus(self, df: pd.DataFrame) -> List[str]:
        """
        Combines relevant textual fields into a rich corpus representation per study.
        """
        corpus = []
        for _, row in df.iterrows():
            parts = [
                str(row.get("brief_title", "") or ""),
                str(row.get("official_title", "") or ""),
                str(row.get("brief_summary", "") or ""),
                str(row.get("detailed_description", "") or ""),
                str(row.get("conditions", "") or ""),
                str(row.get("interventions", "") or ""),
                str(row.get("keywords", "") or ""),
                str(row.get("condition_mesh_terms", "") or ""),
            ]
            # Clean and join
            clean_text = " ".join([p.strip() for p in parts if p.strip()])
            corpus.append(clean_text)
        return corpus

    def fit_transform(
        self,
        df: pd.DataFrame,
        method: str = "hybrid",  # 'embeddings_kmeans', 'embeddings_hdbscan', 'tfidf_kmeans', 'hybrid'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs unsupervised clustering on study corpus and returns enriched dataframe with cluster labels and metadata.
        """
        corpus = self.prepare_corpus(df)
        n_samples = len(corpus)

        if n_samples < self.n_clusters:
            logger.warning(f"Number of samples ({n_samples}) is less than n_clusters ({self.n_clusters}). Adjusting...")
            effective_clusters = max(2, n_samples // 2)
        else:
            effective_clusters = self.n_clusters

        # 1. Feature Representation
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

        if self.use_embeddings and self.embedding_model is not None:
            logger.info("Computing dense sentence embeddings...")
            embeddings = self.embedding_model.encode(
                corpus,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            feature_matrix = embeddings
        else:
            feature_matrix = tfidf_matrix.toarray()

        # 2. Dimensionality Reduction
        if HAS_UMAP and n_samples >= 15:
            logger.info("Applying UMAP dimensionality reduction...")
            self.umap_model_2d = umap.UMAP(
                n_neighbors=min(15, n_samples - 1),
                min_dist=0.1,
                n_components=2,
                metric="cosine",
                random_state=self.random_state,
            )
            coords_2d = self.umap_model_2d.fit_transform(feature_matrix)

            self.umap_model_cluster = umap.UMAP(
                n_neighbors=min(15, n_samples - 1),
                min_dist=0.0,
                n_components=min(10, feature_matrix.shape[1]),
                metric="cosine",
                random_state=self.random_state,
            )
            reduced_features = self.umap_model_cluster.fit_transform(feature_matrix)
        else:
            logger.info("Applying TruncatedSVD dimensionality reduction...")
            svd_2d = TruncatedSVD(n_components=2, random_state=self.random_state)
            coords_2d = svd_2d.fit_transform(feature_matrix)
            svd_high = TruncatedSVD(n_components=min(10, feature_matrix.shape[1]), random_state=self.random_state)
            reduced_features = svd_high.fit_transform(feature_matrix)

        # 3. Clustering Execution
        if "hdbscan" in method and HAS_HDBSCAN and n_samples >= self.min_cluster_size * 2:
            logger.info("Fitting HDBSCAN clustering...")
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=5,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            cluster_labels = hdb.fit_predict(reduced_features)
            self.cluster_model = hdb
        else:
            logger.info(f"Fitting KMeans with k={effective_clusters}...")
            kmeans = KMeans(n_clusters=effective_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_features)
            self.cluster_model = kmeans

        # Attach results to dataframe
        res_df = df.copy()
        res_df["cluster_id"] = cluster_labels
        res_df["umap_x"] = coords_2d[:, 0]
        res_df["umap_y"] = coords_2d[:, 1]

        # 4. Post-Clustering Characterization (c-TF-IDF & Metadata profiles)
        cluster_summaries = self.characterize_clusters(res_df, corpus)
        
        # Add human-interpretable cluster labels to dataframe
        id_to_label = {c_id: c_info["cluster_label"] for c_id, c_info in cluster_summaries.items()}
        res_df["cluster_label"] = res_df["cluster_id"].map(id_to_label)

        return res_df, cluster_summaries

    def compute_c_tf_idf(self, df: pd.DataFrame, corpus: List[str]) -> Tuple[Dict[int, List[Tuple[str, float]]], CountVectorizer]:
        """
        Computes Class-based TF-IDF (c-TF-IDF) to find words that distinctly identify each cluster.
        """
        cluster_docs: Dict[int, List[str]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            c_id = row["cluster_id"]
            if c_id not in cluster_docs:
                cluster_docs[c_id] = []
            if i < len(corpus):
                cluster_docs[c_id].append(corpus[i])

        # Aggregate all text per cluster
        clusters = sorted(cluster_docs.keys())
        aggregated_docs = [" ".join(cluster_docs[c]) for c in clusters]

        c_vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10000,
            min_df=1,
        )
        X_sparse = c_vectorizer.fit_transform(aggregated_docs)
        X = np.asarray(X_sparse.toarray(), dtype=np.float64)
        words = np.array(c_vectorizer.get_feature_names_out())

        # c-TF-IDF formula: W = tf * log(1 + A / f)
        total_words_per_cluster = X.sum(axis=1, keepdims=True) + 1e-9
        tf = X / total_words_per_cluster
        
        f = X.sum(axis=0, keepdims=True) + 1e-9
        A = np.mean(total_words_per_cluster)
        idf = np.log(1.0 + (A / f))

        c_tf_idf = tf * idf

        top_terms_by_cluster: Dict[int, List[Tuple[str, float]]] = {}
        for i, c_id in enumerate(clusters):
            row_scores = c_tf_idf[i].flatten()
            top_indices = np.argsort(row_scores)[::-1][:15]
            top_terms_by_cluster[c_id] = [(words[idx], float(row_scores[idx])) for idx in top_indices]

        return top_terms_by_cluster, c_vectorizer

    def characterize_clusters(self, df: pd.DataFrame, corpus: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Builds a comprehensive profile for each cluster including top keywords,
        archetypal representative studies, historical trajectory, and demographic distribution.
        """
        top_terms_dict, _ = self.compute_c_tf_idf(df, corpus)
        cluster_summaries: Dict[int, Dict[str, Any]] = {}

        for c_id in sorted(df["cluster_id"].unique()):
            sub_df = df[df["cluster_id"] == c_id]
            size = len(sub_df)
            top_terms = top_terms_dict.get(c_id, [])
            top_keywords = [term for term, score in top_terms[:5]]
            
            # Descriptive cluster label from top terms
            if c_id == -1:
                cluster_label = "Unclustered / Outliers"
            else:
                cluster_label = " & ".join(top_keywords[:3]).title()

            # Representative studies closest to cluster centroid
            centroid_x = sub_df["umap_x"].mean()
            centroid_y = sub_df["umap_y"].mean()
            dists = np.sqrt((sub_df["umap_x"] - centroid_x) ** 2 + (sub_df["umap_y"] - centroid_y) ** 2)
            closest_indices = dists.nsmallest(min(3, size)).index
            
            representative_studies = []
            for idx in closest_indices:
                st = sub_df.loc[idx]
                representative_studies.append({
                    "nct_id": st.get("nct_id", ""),
                    "title": st.get("brief_title", ""),
                    "year": int(st["analysis_year"]) if pd.notna(st.get("analysis_year")) else None,
                    "lead_sponsor": st.get("lead_sponsor_name", ""),
                    "study_type": st.get("study_type", ""),
                })

            # Temporal metrics
            years = sub_df["analysis_year"].dropna().astype(int)
            year_mean = float(years.mean()) if len(years) > 0 else None
            year_min = int(years.min()) if len(years) > 0 else None
            year_max = int(years.max()) if len(years) > 0 else None

            # Funding metrics
            nih_pct = float((sub_df["has_nih_funding"] == True).mean() * 100) if "has_nih_funding" in sub_df else 0.0
            industry_pct = float((sub_df["has_industry_funding"] == True).mean() * 100) if "has_industry_funding" in sub_df else 0.0

            # Formal capture metrics
            formal_pct = float((sub_df["has_formal_capture"] == True).mean() * 100) if "has_formal_capture" in sub_df else 0.0

            cluster_summaries[int(c_id)] = {
                "cluster_id": int(c_id),
                "cluster_label": cluster_label,
                "study_count": size,
                "proportion_pct": round(size / len(df) * 100, 2),
                "top_c_tf_idf_terms": top_terms,
                "top_keywords": top_keywords,
                "representative_studies": representative_studies,
                "temporal_profile": {
                    "mean_year": round(year_mean, 1) if year_mean else None,
                    "min_year": year_min,
                    "max_year": year_max,
                },
                "funding_profile": {
                    "nih_funding_pct": round(nih_pct, 1),
                    "industry_funding_pct": round(industry_pct, 1),
                },
                "formal_capture_pct": round(formal_pct, 1),
            }

        return cluster_summaries
