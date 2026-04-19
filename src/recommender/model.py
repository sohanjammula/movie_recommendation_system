# src/recommender/model.py
"""
Robust loader for the recommender artefact and the public `recommend` function.

The artefact can be in one of two formats:

1️⃣ New format (produced by src/train.py):
    {
        "vectorizer": TfidfVectorizer,
        "svd"       : TruncatedSVD,
        "nn"        : NearestNeighbors,
        "df"        : pandas.DataFrame
    }

2️⃣ Old format (produced by the original notebook):
    A two‑element tuple/list:
        (movie_dict, similarity_matrix)

The loader detects the format, reconstructs the required objects,
and, if needed, builds a temporary NN index from the stored similarity
matrix so that the rest of the code works unchanged.
"""

import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional

# ----------------------------------------------------------------------
# Paths & globals
# ----------------------------------------------------------------------
_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "recommender.pkl"
if not _MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at {_MODEL_PATH}. "
        "Place a valid 'recommender.pkl' under ./artifacts."
    )

# ----------------------------------------------------------------------
# Helper: build a NearestNeighbors from a pre‑computed similarity matrix
# ----------------------------------------------------------------------
def _build_nn_from_similarity(similarity: np.ndarray) -> NearestNeighbors:
    """
    The old artefact stored a dense cosine similarity matrix.
    Scikit‑learn can perform NN queries on a *pre‑computed* distance
    matrix, which we obtain by ``distance = 1 - similarity``.
    """
    distance = 1.0 - similarity
    nn = NearestNeighbors(metric="precomputed", algorithm="brute")
    nn.fit(distance)          # the whole distance matrix becomes nn._fit_X
    return nn

# ----------------------------------------------------------------------
# Load the artefact – try the new dict first, fall back to the old tuple
# ----------------------------------------------------------------------
_raw = joblib.load(_MODEL_PATH)

# ----------------------------------------------------------------------
# 1️⃣ New format (dictionary)
# ----------------------------------------------------------------------
if isinstance(_raw, dict) and {"vectorizer", "svd", "nn", "df"} <= set(_raw.keys()):
    vectorizer = _raw["vectorizer"]   # kept for completeness – not used at inference
    svd = _raw["svd"]
    nn = _raw["nn"]
    df = _raw["df"]
    _bundle_type = "new_dict"

# ----------------------------------------------------------------------
# 2️⃣ Old format (tuple / list with two elements)
# ----------------------------------------------------------------------
elif isinstance(_raw, (list, tuple)) and len(_raw) == 2:
    movie_dict, similarity = _raw

    # Re‑create the DataFrame that was saved with .to_dict()
    df = pd.DataFrame.from_dict(movie_dict)

    # Basic sanity check – the columns we need for the UI must exist
    required = {"title", "year", "vote_average"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"The old artefact is missing columns {missing}. "
            "Re‑train the model with the new script."
        )

    # Build a NN object that works on the pre‑computed similarity matrix
    nn = _build_nn_from_similarity(similarity)

    # The old artefact did not keep the TF‑IDF vectorizer or the SVD transformer
    vectorizer = None
    svd = None
    _bundle_type = "old_tuple"

# ----------------------------------------------------------------------
# 3️⃣ Unrecognised format – raise a clear error
# ----------------------------------------------------------------------
else:
    raise RuntimeError(
        "The artefact at "
        f"{_MODEL_PATH} has an unrecognised format. "
        "It should be either a dict with keys "
        "'vectorizer', 'svd', 'nn', 'df' (new format) "
        "or a two‑element tuple/list (old format). "
        "Re‑create the artefact with the provided training script."
    )

# ----------------------------------------------------------------------
# Store the training matrix (the data on which the NN was fitted)
# Using a slice later (`[i:i+1]`) guarantees a 2‑D shape for both
# dense vectors and pre‑computed distance rows.
# ----------------------------------------------------------------------
_train_X = nn._fit_X  # could be the reduced vectors or the distance matrix

# ----------------------------------------------------------------------
# Public function that the UI imports
# ----------------------------------------------------------------------
def recommend(
    movie_title: str,
    top_n: int = 10,
    min_rating: Optional[float] = None,
    min_year: Optional[int] = None,
) -> List[dict]:
    """
    Return *top_n* most similar movies for the given title.

    Parameters
    ----------
    movie_title : str
        Exact title as stored in the catalogue (case‑sensitive).
    top_n : int, default 10
        Number of recommendations to return.
    min_rating : float | None, optional
        Keep only movies with ``vote_average >= min_rating``.
    min_year : int | None, optional
        Keep only movies released in or after ``min_year``.

    Returns
    -------
    List[dict] – each dict contains:
        - title
        - year
        - vote_average
        - hybrid_score  (0.7 * content similarity + 0.3 * normalised rating)
    """
    # ------------------------------------------------------------------
    # 1️⃣ Locate the query movie inside the stored DataFrame
    # ------------------------------------------------------------------
    if movie_title not in df["title"].values:
        raise ValueError(f"Movie '{movie_title}' not found in the catalogue.")
    query_idx = df.index[df["title"] == movie_title][0]

    # ------------------------------------------------------------------
    # 2️⃣ Retrieve the query vector (always 2‑D)
    #    * For a *regular* NN (metric='cosine') _train_X holds the
    #      dense reduced vectors → slice gives shape (1, n_features).
    #    * For a *pre‑computed* NN (metric='precomputed') _train_X
    #      holds the full distance matrix → slice gives shape (1, n_samples).
    # ------------------------------------------------------------------
    query_vec = _train_X[query_idx : query_idx + 1]   # safe 2‑D slice

    # ------------------------------------------------------------------
    # 3️⃣ Perform the nearest‑neighbour search – ask for a few extra neighbours
    # ------------------------------------------------------------------
    distances, indices = nn.kneighbors(query_vec, n_neighbors=top_n + 15)
    distances = distances.ravel()
    indices = indices.ravel()

    # ------------------------------------------------------------------
    # 4️⃣ Assemble a temporary results DataFrame and apply optional filters
    # ------------------------------------------------------------------
    results = df.iloc[indices].copy()
    results["cosine_dist"] = distances
    results["content_sim"] = 1 - results["cosine_dist"]   # higher = more similar

    if min_rating is not None:
        results = results[results["vote_average"] >= min_rating]
    if min_year is not None:
        results = results[results["year"] >= min_year]

    # ------------------------------------------------------------------
    # 5️⃣ Hybrid re‑ranking (0.7 * content + 0.3 * normalised rating)
    # ------------------------------------------------------------------
    max_rating = df["vote_average"].max()
    results["rating_norm"] = results["vote_average"] / max_rating
    results["hybrid_score"] = (
        0.7 * results["content_sim"] + 0.3 * results["rating_norm"]
    )

    # ------------------------------------------------------------------
    # 6️⃣ Drop the query movie itself and keep the best `top_n` rows
    # ------------------------------------------------------------------
    results = results[results["title"] != movie_title]
    results = results.sort_values("hybrid_score", ascending=False).head(top_n)

    # ------------------------------------------------------------------
    # 7️⃣ Return a JSON‑serialisable list of dictionaries
    # ------------------------------------------------------------------
    return results[
        ["title", "year", "vote_average", "hybrid_score"]
    ].to_dict(orient="records")


