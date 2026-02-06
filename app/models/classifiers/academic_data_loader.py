import os
import json
import gzip
import pickle
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union
from dataclasses import dataclass, field
from collections import defaultdict
import re
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import resample


@dataclass
class DocumentRecord:
    doc_id: str
    raw_text: str
    metadata: Dict
    label: Optional[int] = None
    split: Optional[str] = None
    feature_vector: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


class AcademicCorpusIndex:
    def __init__(self, corpus_root: str, extension: str = ".txt"):
        self.corpus_root = Path(corpus_root)
        self.extension = extension
        self.index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.corpus_root.exists():
            return
        for path in self.corpus_root.rglob(f"*{self.extension}"):
            rel = path.relative_to(self.corpus_root)
            key = str(rel).replace(os.sep, "_").replace(self.extension, "")
            self.index[key] = path

    def get_path(self, doc_id: str) -> Optional[Path]:
        return self.index.get(doc_id)

    def list_ids(self) -> List[str]:
        return sorted(self.index.keys())

    def __len__(self) -> int:
        return len(self.index)


class AcademicDataLoader:
    def __init__(
        self,
        corpus_root: str,
        max_doc_length: int = 10000,
        min_doc_length: int = 50,
        encoding: str = "utf-8",
        normalize_unicode: bool = True,
        strip_whitespace: bool = True,
    ):
        self.corpus_root = Path(corpus_root)
        self.max_doc_length = max_doc_length
        self.min_doc_length = min_doc_length
        self.encoding = encoding
        self.normalize_unicode = normalize_unicode
        self.strip_whitespace = strip_whitespace
        self.index = AcademicCorpusIndex(corpus_root)
        self._cache: Dict[str, str] = {}
        self._label_map: Optional[Dict[str, int]] = None

    def _normalize_text(self, text: str) -> str:
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _read_file(self, path: Path) -> str:
        path_str = str(path)
        if path_str in self._cache:
            return self._cache[path_str]
        try:
            with open(path, "r", encoding=self.encoding, errors="replace") as f:
                content = f.read()
        except Exception:
            content = ""
        content = self._normalize_text(content)
        if len(content) > self.max_doc_length:
            content = content[: self.max_doc_length]
        self._cache[path_str] = content
        return content

    def load_single(self, doc_id: str) -> Optional[DocumentRecord]:
        path = self.index.get_path(doc_id)
        if path is None:
            return None
        raw_text = self._read_file(path)
        if len(raw_text) < self.min_doc_length:
            return None
        return DocumentRecord(
            doc_id=doc_id,
            raw_text=raw_text,
            metadata={"path": str(path), "length": len(raw_text)},
        )

    def load_batch(self, doc_ids: List[str]) -> List[DocumentRecord]:
        records = []
        for doc_id in doc_ids:
            rec = self.load_single(doc_id)
            if rec is not None:
                records.append(rec)
        return records

    def load_all(self) -> List[DocumentRecord]:
        ids = self.index.list_ids()
        return self.load_batch(ids)

    def iter_batches(
        self, batch_size: int = 32, shuffle: bool = True, seed: Optional[int] = None
    ) -> Iterator[List[DocumentRecord]]:
        ids = self.index.list_ids()
        if shuffle and seed is not None:
            rng = np.random.default_rng(seed)
            ids = list(ids)
            rng.shuffle(ids)
        elif shuffle:
            ids = list(ids)
            np.random.shuffle(ids)
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            yield self.load_batch(batch_ids)


class LabeledAcademicLoader(AcademicDataLoader):
    def __init__(
        self,
        corpus_root: str,
        labels_path: Optional[str] = None,
        label_column: str = "label",
        id_column: str = "doc_id",
        **kwargs,
    ):
        super().__init__(corpus_root, **kwargs)
        self.labels_path = labels_path
        self.label_column = label_column
        self.id_column = id_column
        self.labels: Dict[str, int] = {}
        if labels_path:
            self._load_labels()

    def _load_labels(self) -> None:
        path = Path(self.labels_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            try:
                id_idx = header.index(self.id_column)
            except ValueError:
                id_idx = 0
            try:
                label_idx = header.index(self.label_column)
            except ValueError:
                label_idx = 1
            for line in f:
                parts = line.strip().split(",")
                if len(parts) > max(id_idx, label_idx):
                    self.labels[parts[id_idx]] = int(parts[label_idx])

    def set_labels(self, labels: Dict[str, int]) -> None:
        self.labels = labels

    def load_single(self, doc_id: str) -> Optional[DocumentRecord]:
        rec = super().load_single(doc_id)
        if rec is None:
            return None
        rec.label = self.labels.get(doc_id)
        return rec

    def get_label_vector(self, doc_ids: List[str]) -> np.ndarray:
        return np.array([self.labels.get(d, -1) for d in doc_ids], dtype=np.int64)


class StratifiedAcademicSplitter:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self, records: List[DocumentRecord]
    ) -> Iterator[Tuple[List[DocumentRecord], List[DocumentRecord]]]:
        ids = [r.doc_id for r in records]
        labels = np.array([r.label if r.label is not None else 0 for r in records])
        for train_idx, test_idx in self.skf.split(ids, labels):
            train_recs = [records[i] for i in train_idx]
            test_recs = [records[i] for i in test_idx]
            yield train_recs, test_recs


class AcademicBatchGenerator:
    def __init__(
        self,
        records: List[DocumentRecord],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self) -> Iterator[List[DocumentRecord]]:
        indices = np.arange(len(self.records))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        n_batches = len(indices) // self.batch_size
        if self.drop_last and len(indices) % self.batch_size != 0:
            indices = indices[: n_batches * self.batch_size]
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            yield [self.records[j] for j in batch_idx]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.records) // self.batch_size
        return (len(self.records) + self.batch_size - 1) // self.batch_size


def compute_doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def filter_by_label(
    records: List[DocumentRecord], allowed_labels: List[int]
) -> List[DocumentRecord]:
    allowed = set(allowed_labels)
    return [r for r in records if r.label is not None and r.label in allowed]


def undersample_by_label(
    records: List[DocumentRecord], random_state: Optional[int] = None
) -> List[DocumentRecord]:
    by_label: Dict[int, List[DocumentRecord]] = defaultdict(list)
    for r in records:
        if r.label is not None:
            by_label[r.label].append(r)
    min_count = min(len(v) for v in by_label.values()) if by_label else 0
    if min_count == 0:
        return records
    result = []
    for label, recs in by_label.items():
        sampled = resample(
            recs, n_samples=min_count, replace=False, random_state=random_state
        )
        result.extend(sampled)
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        rng.shuffle(result)
    return result


def oversample_by_label(
    records: List[DocumentRecord], random_state: Optional[int] = None
) -> List[DocumentRecord]:
    by_label: Dict[int, List[DocumentRecord]] = defaultdict(list)
    for r in records:
        if r.label is not None:
            by_label[r.label].append(r)
    max_count = max(len(v) for v in by_label.values()) if by_label else 0
    if max_count == 0:
        return records
    result = []
    rng = np.random.default_rng(random_state)
    for label, recs in by_label.items():
        n_extra = max_count - len(recs)
        if n_extra > 0:
            indices = rng.integers(0, len(recs), size=n_extra)
            recs = list(recs) + [recs[i] for i in indices]
        result.extend(recs)
    rng.shuffle(result)
    return result


def save_records_cache(records: List[DocumentRecord], path: str) -> None:
    data = [
        {
            "doc_id": r.doc_id,
            "raw_text": r.raw_text,
            "metadata": r.metadata,
            "label": r.label,
            "split": r.split,
        }
        for r in records
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=0)


def load_records_cache(path: str) -> List[DocumentRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        DocumentRecord(
            doc_id=item["doc_id"],
            raw_text=item["raw_text"],
            metadata=item.get("metadata", {}),
            label=item.get("label"),
            split=item.get("split"),
        )
        for item in data
    ]
