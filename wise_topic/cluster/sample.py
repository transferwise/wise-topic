from typing import Dict, List

import numpy as np


def sample_docs(documents: np.ndarray, labels: np.ndarray, docs_per_label: int = 10) -> Dict[int, List[str]]:
    out = {}
    for i in np.unique(labels):
        d = documents[labels == i]
        out[i] = np.random.choice(d, min(docs_per_label, len(d)), replace=False)
    return out


def sample_docs_from_proba(documents: np.ndarray, p: np.ndarray, n: int = 10):
    out = {}
    # square the probabilities to accentuate high-probability topics
    re_p = (p * p) / ((p * p).sum(axis=1, keepdims=True))
    for i in range(p.shape[1]):
        out[i] = np.random.choice(documents, size=n, replace=False, p=re_p[:, i] / re_p[:, i].sum())
    return out
