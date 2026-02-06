import re
import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


def tokenize_whitespace(text: str) -> List[str]:
    return text.split()


def tokenize_regex(text: str, pattern: str = r"\b\w+\b") -> List[str]:
    return re.findall(pattern, text.lower())


def ngram_sequences(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class LexicalFeatureExtractor:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize_regex(text)
        self.sentences = re.split(r"[.!?]+", text)
        self.sentences = [s.strip() for s in self.sentences if s.strip()]

    def type_token_ratio(self) -> float:
        if not self.tokens:
            return 0.0
        return len(set(self.tokens)) / len(self.tokens)

    def hapax_legomena_ratio(self) -> float:
        if not self.tokens:
            return 0.0
        counts = Counter(self.tokens)
        hapax = sum(1 for c in counts.values() if c == 1)
        return hapax / len(self.tokens)

    def yule_k(self) -> float:
        if not self.tokens:
            return 0.0
        counts = Counter(self.tokens)
        m1 = sum(counts.values())
        m2 = sum(c * c for c in counts.values())
        if m2 == 0:
            return 0.0
        return (m1 * m1) / (m2 - m1) if m2 != m1 else 0.0

    def simpson_diversity(self) -> float:
        if not self.tokens:
            return 0.0
        n = len(self.tokens)
        counts = Counter(self.tokens)
        return 1.0 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0

    def avg_word_length(self) -> float:
        if not self.tokens:
            return 0.0
        return statistics.mean(len(t) for t in self.tokens)

    def std_word_length(self) -> float:
        if len(self.tokens) < 2:
            return 0.0
        return statistics.pstdev(len(t) for t in self.tokens)

    def sentence_length_mean(self) -> float:
        if not self.sentences:
            return 0.0
        lengths = [len(tokenize_regex(s)) for s in self.sentences]
        return statistics.mean(lengths)

    def sentence_length_std(self) -> float:
        if len(self.sentences) < 2:
            return 0.0
        lengths = [len(tokenize_regex(s)) for s in self.sentences]
        return statistics.pstdev(lengths)

    def extract_all(self) -> Dict[str, float]:
        return {
            "type_token_ratio": self.type_token_ratio(),
            "hapax_legomena_ratio": self.hapax_legomena_ratio(),
            "yule_k": self.yule_k(),
            "simpson_diversity": self.simpson_diversity(),
            "avg_word_length": self.avg_word_length(),
            "std_word_length": self.std_word_length(),
            "sentence_length_mean": self.sentence_length_mean(),
            "sentence_length_std": self.sentence_length_std(),
        }


class StructuralFeatureExtractor:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize_regex(text)

    def digit_ratio(self) -> float:
        if not self.text:
            return 0.0
        digits = sum(c.isdigit() for c in self.text)
        return digits / len(self.text)

    def uppercase_ratio(self) -> float:
        if not self.text:
            return 0.0
        letters = sum(c.isalpha() for c in self.text)
        if letters == 0:
            return 0.0
        upper = sum(c.isupper() for c in self.text)
        return upper / letters

    def punctuation_ratio(self) -> float:
        if not self.text:
            return 0.0
        punct = sum(1 for c in self.text if c in ".,;:!?\"'()-")
        return punct / len(self.text)

    def whitespace_ratio(self) -> float:
        if not self.text:
            return 0.0
        return self.text.count(" ") / len(self.text)

    def paragraph_count_approx(self) -> int:
        return max(1, self.text.count("\n\n") + 1)

    def avg_paragraph_length(self) -> float:
        paras = [p.strip() for p in self.text.split("\n\n") if p.strip()]
        if not paras:
            return 0.0
        return statistics.mean(len(tokenize_regex(p)) for p in paras)

    def extract_all(self) -> Dict[str, float]:
        return {
            "digit_ratio": self.digit_ratio(),
            "uppercase_ratio": self.uppercase_ratio(),
            "punctuation_ratio": self.punctuation_ratio(),
            "whitespace_ratio": self.whitespace_ratio(),
            "paragraph_count": float(self.paragraph_count_approx()),
            "avg_paragraph_length": self.avg_paragraph_length(),
        }


class TfIdfFeaturePipeline:
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        use_idf: bool = True,
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, texts: List[str]) -> "TfIdfFeaturePipeline":
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        dense = X.toarray()
        self.scaler.fit(dense)
        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        dense = X.toarray()
        return self.scaler.transform(dense)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class BagOfWordsPipeline:
    def __init__(
        self,
        max_features: int = 10000,
        binary: bool = False,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        self.max_features = max_features
        self.binary = binary
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            binary=binary,
            ngram_range=ngram_range,
        )
        self.normalizer = Normalizer(norm="l2")
        self.fitted = False

    def fit(self, texts: List[str]) -> "BagOfWordsPipeline":
        self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.normalizer.fit_transform(X)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        return self.normalizer.fit_transform(X)


class AcademicFeaturePipeline:
    def __init__(
        self,
        include_lexical: bool = True,
        include_structural: bool = True,
        include_tfidf: bool = True,
        tfidf_max_features: int = 2000,
    ):
        self.include_lexical = include_lexical
        self.include_structural = include_structural
        self.include_tfidf = include_tfidf
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_pipeline: Optional[TfIdfFeaturePipeline] = None
        self.lexical_names: List[str] = []
        self.structural_names: List[str] = []
        self.feature_order: List[str] = []

    def _lexical_vector(self, text: str) -> np.ndarray:
        ext = LexicalFeatureExtractor(text)
        d = ext.extract_all()
        self.lexical_names = list(d.keys())
        return np.array([d[k] for k in self.lexical_names], dtype=np.float64)

    def _structural_vector(self, text: str) -> np.ndarray:
        ext = StructuralFeatureExtractor(text)
        d = ext.extract_all()
        self.structural_names = list(d.keys())
        return np.array([d[k] for k in self.structural_names], dtype=np.float64)

    def fit(self, texts: List[str]) -> "AcademicFeaturePipeline":
        if self.include_lexical:
            self._lexical_vector(texts[0] if texts else "")
        if self.include_structural:
            self._structural_vector(texts[0] if texts else "")
        if self.include_tfidf and texts:
            self.tfidf_pipeline = TfIdfFeaturePipeline(
                max_features=self.tfidf_max_features
            )
            self.tfidf_pipeline.fit(texts)
        self.feature_order = (
            self.lexical_names + self.structural_names + ["tfidf_dim_" + str(i) for i in range(self.tfidf_max_features if self.tfidf_pipeline else 0)]
        )[: len(self.lexical_names) + len(self.structural_names) + (self.tfidf_max_features if self.tfidf_pipeline else 0)]
        return self

    def transform_single(self, text: str) -> np.ndarray:
        parts = []
        if self.include_lexical:
            parts.append(self._lexical_vector(text))
        if self.include_structural:
            parts.append(self._structural_vector(text))
        if self.include_tfidf and self.tfidf_pipeline is not None:
            parts.append(self.tfidf_pipeline.transform([text]).squeeze(0))
        return np.concatenate(parts, axis=0) if parts else np.array([])

    def transform(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.transform_single(t) for t in texts])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class SVDReductionPipeline:
    def __init__(self, n_components: int = 100, random_state: Optional[int] = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.fitted = False

    def fit(self, X: np.ndarray) -> "SVDReductionPipeline":
        self.svd.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.svd.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.svd.fit_transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        return self.svd.explained_variance_ratio_


class NMFFeaturePipeline:
    def __init__(self, n_components: int = 50, random_state: Optional[int] = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.nmf = NMF(n_components=n_components, random_state=random_state)
        self.fitted = False

    def fit(self, X: np.ndarray) -> "NMFFeaturePipeline":
        X_nonneg = np.maximum(X, 0)
        self.nmf.fit(X_nonneg)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_nonneg = np.maximum(X, 0)
        return self.nmf.transform(X_nonneg)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_nonneg = np.maximum(X, 0)
        return self.nmf.fit_transform(X_nonneg)


def compute_correlation_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(feature_matrix.T)


def remove_high_correlation_features(
    X: np.ndarray, threshold: float = 0.95
) -> Tuple[np.ndarray, List[int]]:
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)
    to_drop = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            if corr[i, j] >= threshold:
                to_drop.add(j)
    keep = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep], keep


def variance_threshold_mask(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    variances = np.var(X, axis=0)
    return variances > threshold


def select_by_variance(X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    mask = variance_threshold_mask(X, threshold)
    return X[:, mask]
