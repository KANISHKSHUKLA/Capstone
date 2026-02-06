from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine, cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import warnings


def mean_pool(embeddings: List[NDArray]) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    return np.mean(stacked, axis=0)


def max_pool(embeddings: List[NDArray]) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    return np.max(stacked, axis=0)


def min_pool(embeddings: List[NDArray]) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    return np.min(stacked, axis=0)


def weighted_mean_pool(
    embeddings: List[NDArray], weights: Optional[List[float]] = None
) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    if weights is None:
        return np.mean(stacked, axis=0)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    return np.average(stacked, axis=0, weights=w)


def last_token_pool(embedding_matrix: NDArray) -> NDArray:
    return embedding_matrix[-1, :]


def first_token_pool(embedding_matrix: NDArray) -> NDArray:
    return embedding_matrix[0, :]


def attention_weighted_pool(
    embedding_matrix: NDArray, attention_weights: NDArray
) -> NDArray:
    attention_weights = attention_weights / (attention_weights.sum() + 1e-9)
    return np.average(embedding_matrix, axis=0, weights=attention_weights)


def sqrt_mean_pool(embeddings: List[NDArray]) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    signed = np.sign(stacked) * np.sqrt(np.abs(stacked))
    return np.mean(signed, axis=0)


def power_mean_pool(embeddings: List[NDArray], p: float = 2.0) -> NDArray:
    stacked = np.stack(embeddings, axis=0)
    if abs(p) < 1e-9:
        return np.exp(np.mean(np.log(np.abs(stacked) + 1e-9), axis=0)) * np.sign(
            np.mean(stacked, axis=0)
        )
    return np.power(
        np.mean(np.power(np.abs(stacked) + 1e-9, p), axis=0), 1.0 / p
    ) * np.sign(np.mean(stacked, axis=0))


class EmbeddingNormalizer:
    def __init__(self, norm: str = "l2"):
        self.norm = norm
        self.normalizer = Normalizer(norm=norm)

    def fit(self, X: NDArray) -> "EmbeddingNormalizer":
        self.normalizer.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.normalizer.transform(X).squeeze()

    def fit_transform(self, X: NDArray) -> NDArray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.normalizer.fit_transform(X).squeeze()


class EmbeddingReducer:
    def __init__(
        self,
        method: str = "pca",
        n_components: int = 128,
        random_state: Optional[int] = 42,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self.scaler = StandardScaler()
        self.fitted = False

    def _fit_pca(self, X: NDArray) -> None:
        self.reducer = PCA(
            n_components=min(self.n_components, X.shape[1], X.shape[0]),
            random_state=self.random_state,
        )
        self.reducer.fit(X)

    def _fit_rp(self, X: NDArray) -> None:
        self.reducer = GaussianRandomProjection(
            n_components=min(self.n_components, X.shape[1]),
            random_state=self.random_state,
        )
        self.reducer.fit(X)

    def fit(self, X: NDArray) -> "EmbeddingReducer":
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.fit_transform(X)
        if self.method == "pca":
            self._fit_pca(X_scaled)
        elif self.method == "random_projection":
            self._fit_rp(X_scaled)
        else:
            self._fit_pca(X_scaled)
        self.fitted = True
        return self

    def transform(self, X: NDArray) -> NDArray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled).squeeze()

    def fit_transform(self, X: NDArray) -> NDArray:
        return self.fit(X).transform(X)


class MultiEmbeddingAggregator:
    def __init__(
        self,
        strategies: List[str],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        reduce_dim: Optional[int] = None,
    ):
        self.strategies = strategies
        self.weights = weights or {s: 1.0 for s in strategies}
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.normalizer = EmbeddingNormalizer("l2") if normalize else None
        self.reducer: Optional[EmbeddingReducer] = None
        self.fitted = False

    def _aggregate_single(
        self, embedding_dict: Dict[str, NDArray]
    ) -> NDArray:
        parts = []
        total_weight = 0.0
        for s in self.strategies:
            if s in embedding_dict:
                w = self.weights.get(s, 1.0)
                parts.append(embedding_dict[s] * w)
                total_weight += w
        if not parts:
            raise ValueError("No embeddings found for any strategy")
        combined = np.sum(parts, axis=0) / (total_weight + 1e-9)
        return combined

    def fit(self, embedding_dicts: List[Dict[str, NDArray]]) -> "MultiEmbeddingAggregator":
        aggregated = np.vstack(
            [self._aggregate_single(d) for d in embedding_dicts]
        )
        if self.normalize and self.normalizer is not None:
            aggregated = self.normalizer.fit_transform(aggregated)
        if self.reduce_dim is not None and aggregated.shape[1] > self.reduce_dim:
            self.reducer = EmbeddingReducer(
                method="pca", n_components=self.reduce_dim
            )
            self.reducer.fit(aggregated)
        self.fitted = True
        return self

    def transform_single(self, embedding_dict: Dict[str, NDArray]) -> NDArray:
        out = self._aggregate_single(embedding_dict)
        if self.normalize and self.normalizer is not None:
            out = self.normalizer.transform(out)
        if self.reducer is not None:
            out = self.reducer.transform(out)
        return out

    def transform(
        self, embedding_dicts: List[Dict[str, NDArray]]
    ) -> NDArray:
        return np.vstack([self.transform_single(d) for d in embedding_dicts])


def cosine_similarity(a: NDArray, b: NDArray) -> float:
    return 1.0 - cosine(a, b)


def pairwise_cosine_similarity_matrix(embeddings: NDArray) -> NDArray:
    normalizer = Normalizer(norm="l2")
    en = normalizer.fit_transform(embeddings)
    return np.dot(en, en.T)


def centroid_distance(embeddings: NDArray, centroid: NDArray) -> NDArray:
    if centroid.ndim == 1:
        centroid = centroid.reshape(1, -1)
    return cdist(embeddings, centroid, metric="cosine").squeeze()


def cluster_embeddings(
    embeddings: NDArray,
    n_clusters: int = 8,
    random_state: Optional[int] = 42,
) -> Tuple[NDArray, NDArray]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    return labels, centers


def tsne_project(
    embeddings: NDArray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: Optional[int] = 42,
) -> NDArray:
    if embeddings.shape[0] < 3:
        return embeddings[:, :n_components]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, embeddings.shape[0] - 1),
            random_state=random_state,
        )
        return tsne.fit_transform(embeddings)


def concatenate_embeddings(
    embedding_list: List[NDArray], axis: int = -1
) -> NDArray:
    return np.concatenate(embedding_list, axis=axis)


def gated_combination(
    emb_a: NDArray, emb_b: NDArray, gate: float
) -> NDArray:
    return gate * emb_a + (1.0 - gate) * emb_b


def whitening_transform(embeddings: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    whitened = centered @ W
    return whitened, mean, W


def apply_whitening(
    embeddings: NDArray, mean: NDArray, W: NDArray
) -> NDArray:
    return (embeddings - mean) @ W


def sentence_embedding_from_tokens(
    token_embeddings: NDArray,
    attention_mask: Optional[NDArray] = None,
    strategy: str = "mean",
) -> NDArray:
    if attention_mask is not None:
        mask = attention_mask.reshape(-1, 1)
        summed = np.sum(token_embeddings * mask, axis=0)
        count = np.sum(mask) + 1e-9
        return summed / count
    if strategy == "mean":
        return np.mean(token_embeddings, axis=0)
    if strategy == "max":
        return np.max(token_embeddings, axis=0)
    return np.mean(token_embeddings, axis=0)


def layer_wise_combination(
    layer_embeddings: List[NDArray], weights: Optional[NDArray] = None
) -> NDArray:
    stacked = np.stack(layer_embeddings, axis=0)
    if weights is None:
        return np.mean(stacked, axis=0)
    w = weights.reshape(-1, 1)
    return np.sum(stacked * w, axis=0) / (np.sum(weights) + 1e-9)


class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, NDArray] = {}
        self._order: List[str] = []

    def get(self, key: str) -> Optional[NDArray]:
        return self._cache.get(key)

    def set(self, key: str, value: NDArray) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


def embedding_statistics(embeddings: NDArray) -> Dict[str, float]:
    return {
        "mean_norm": float(np.linalg.norm(np.mean(embeddings, axis=0))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "max_norm": float(np.max(np.linalg.norm(embeddings, axis=1))),
        "min_norm": float(np.min(np.linalg.norm(embeddings, axis=1))),
    }
