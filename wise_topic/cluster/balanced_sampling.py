from typing import Callable, Sequence, Optional, Dict

import numpy as np

from bertopic.backend._utils import select_backend
from flair.embeddings import TransformerDocumentEmbeddings

from phate import PHATE
from umap import UMAP

from llm_patterns.utils.caching_query import caching_query
from .clustering import kmeans_clustering, hdbscan_clustering, bayesian_gaussian_mixture, agglomerative_clustering
from .embedders import tfidf


def embedder_template(emb_model, caching_fn: str) -> Callable:
    return lambda x: caching_query(caching_fn, lambda: select_backend(emb_model).embed_documents(x))


def transformer_template(model) -> Callable:
    return lambda x, fn: caching_query(fn, lambda: model.fit_transform(x))


embedders = {
    "instructor_large": lambda: embedder_template(
        TransformerDocumentEmbeddings("hkunlp/instructor-large"),
        "instructor-large.pklz",
    ),
    "sbert": lambda: embedder_template("all-mpnet-base-v2", "sbert.pklz"),
    "sbert_mini": lambda: embedder_template("all-MiniLM-L6-v2", "sbert_mini.pklz"),
    "tfidf": embedder_template(tfidf(), "tfidf.pklz"),
}

dim_reducers = {
    "umap": lambda dim: transformer_template(UMAP(n_neighbors=15, n_components=dim, min_dist=0.0, metric="cosine")),
    "phate": lambda dim: transformer_template(PHATE(n_components=dim)),
}

clusterers = {
    "kmeans": lambda x, kwargs: kmeans_clustering(x, **{"n_clusters": 10, **kwargs}),
    "bgm": lambda x, kwargs: bayesian_gaussian_mixture(x, **{"prior": 0.01, **kwargs}),
    "hdbscan": lambda x, kwargs: hdbscan_clustering(x, **kwargs),
    "agglomerative": lambda x, kwargs: agglomerative_clustering(x, **{"n_clusters": 10, **kwargs}),
}


def cluster_docs(
    docs: Sequence[str],
    embedder: str = "sbert",
    dim_reducer: str = "umap",
    cluster_algo: str = "kmeans",
    cluster_dim: int = 5,
    cluster_kwargs: Optional[Dict] = None,
):
    cluster_kwargs = cluster_kwargs or {}
    embeddings = embedders[embedder]()(np.array(docs))
    reduced = dim_reducers[dim_reducer.lower()](dim=cluster_dim)(embeddings, f"{embedder}_{dim_reducer}.pklz")
    reduced2 = dim_reducers[dim_reducer](dim=2)(embeddings, f"{embedder}_{dim_reducer}2.pklz")

    cluster_fit = clusterers[cluster_algo](reduced, cluster_kwargs)
    labels = cluster_fit.labels_

    out = {"labels": labels, "embeddings": embeddings, "reduced": reduced, "reduced2d": reduced2}
    return out
