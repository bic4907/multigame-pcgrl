from __future__ import annotations

import inspect

import numpy as np
from sklearn.manifold import TSNE


def run_tsne(
    embeddings: np.ndarray,
    perplexity: float,
    max_iter: int,
    seed: int,
    n_components: int = 2,
    n_jobs: int = -1,
) -> np.ndarray:
    perplexity = min(float(perplexity), max(5.0, (len(embeddings) - 1) / 3.0))
    kwargs = {
        "n_components": int(n_components),
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "metric": "cosine",
        "random_state": int(seed),
        "verbose": 1,
    }
    tsne_params = inspect.signature(TSNE).parameters
    iter_arg = "max_iter" if "max_iter" in tsne_params else "n_iter"
    kwargs[iter_arg] = int(max_iter)
    if "n_jobs" in tsne_params:
        kwargs["n_jobs"] = int(n_jobs)
    reducer = TSNE(**kwargs)
    return reducer.fit_transform(embeddings)
