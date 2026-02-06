

import re
import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import spacy
import textstat


class AcademicFeatureEngineer:
    """
    Feature extractor for academic documents.

    This class intentionally exposes every intermediate computation
    to allow debugging, explanation, and auditing.
    """

    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp(raw_text)

        # Internal caches (explicit on purpose)
        self._sentences = list(self.doc.sents)
        self._tokens = [t for t in self.doc if t.is_alpha]
        self._pos_tags = [t.pos_ for t in self._tokens]

    # --------------------------------------------------
    # Sentence-Level Statistics
    # --------------------------------------------------

    def _sentence_lengths(self) -> List[int]:
        lengths = []
        for sent in self._sentences:
            token_count = len([t for t in sent if t.is_alpha])
            lengths.append(token_count)
        return lengths

    def compute_sentence_statistics(self) -> Dict[str, float]:
        lengths = self._sentence_lengths()

        if not lengths:
            return {
                "avg_sentence_length": 0.0,
                "std_sentence_length": 0.0,
                "max_sentence_length": 0.0,
                "min_sentence_length": 0.0,
            }

        return {
            "avg_sentence_length": statistics.mean(lengths),
            "std_sentence_length": statistics.pstdev(lengths),
            "max_sentence_length": max(lengths),
            "min_sentence_length": min(lengths),
        }

    # --------------------------------------------------
    # Lexical Complexity
    # --------------------------------------------------

    def compute_lexical_metrics(self) -> Dict[str, float]:
        unique_words = set([t.text.lower() for t in self._tokens])
        total_words = len(self._tokens)

        lexical_diversity = (
            len(unique_words) / total_words if total_words > 0 else 0.0
        )

        return {
            "lexical_diversity": lexical_diversity,
            "unique_word_count": len(unique_words),
            "total_word_count": total_words,
        }

    # --------------------------------------------------
    # Readability & Cognitive Load
    # --------------------------------------------------

    def compute_readability(self) -> Dict[str, float]:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(self.raw_text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(self.raw_text),
            "gunning_fog_index": textstat.gunning_fog(self.raw_text),
            "smog_index": textstat.smog_index(self.raw_text),
            "coleman_liau_index": textstat.coleman_liau_index(self.raw_text),
        }

    # --------------------------------------------------
    # POS Distribution (Syntactic Structure)
    # --------------------------------------------------

    def compute_pos_distribution(self) -> Dict[str, float]:
        pos_counts = Counter(self._pos_tags)
        total = sum(pos_counts.values())

        pos_ratios = {}
        for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]:
            pos_ratios[f"{pos.lower()}_ratio"] = (
                pos_counts.get(pos, 0) / total if total > 0 else 0.0
            )

        return pos_ratios

    # --------------------------------------------------
    # Citation Pattern Analysis
    # --------------------------------------------------

    def compute_citation_features(self) -> Dict[str, float]:
        patterns = [
            r"\[\d+\]",                 # [1], [2]
            r"\([A-Za-z]+ et al\., \d{4}\)",  # (Smith et al., 2020)
            r"\([A-Za-z]+, \d{4}\)"     # (Smith, 2020)
        ]

        citation_hits = 0
        for pattern in patterns:
            matches = re.findall(pattern, self.raw_text)
            citation_hits += len(matches)

        total_tokens = len(self.raw_text.split())

        return {
            "citation_count": citation_hits,
            "citation_density": citation_hits / max(total_tokens, 1),
        }

    # --------------------------------------------------
    # Feature Aggregation
    # --------------------------------------------------

    def extract_all_features(self) -> Dict[str, float]:
        """
        Master method aggregating all features.
        """
        features = {}
        features.update(self.compute_sentence_statistics())
        features.update(self.compute_lexical_metrics())
        features.update(self.compute_readability())
        features.update(self.compute_pos_distribution())
        features.update(self.compute_citation_features())

        return features
