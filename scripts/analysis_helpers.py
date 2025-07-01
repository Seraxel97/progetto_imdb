"""Helper utilities for text and CSV analysis."""
from __future__ import annotations

import base64
import io
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

from scripts.enhanced_utils_unified import (
    get_top_phrases,
    extract_topics,
    extract_keywords_by_label,
)


def normalize_predictions(predictions: Dict[str, Any] | None) -> Dict[str, List[int]]:
    """Ensure predictions are lists of ints."""
    clean: Dict[str, List[int]] = {}
    if not predictions:
        return clean
    for name, pred in predictions.items():
        if pred is None:
            continue
        if isinstance(pred, (np.ndarray, pd.Series)):
            values = np.atleast_1d(pred).astype(int).tolist()
        elif isinstance(pred, Iterable) and not isinstance(pred, (str, bytes)):
            values = [int(x) for x in pred]
        else:
            values = [int(pred)]
        clean[name] = values
    return clean


def safe_generate_report(
    df: pd.DataFrame,
    predictions: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    deep_analysis: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Generate scientific report with robust type handling."""
    preds = normalize_predictions(predictions)
    report = {
        "dataset_statistics": {},
        "sentiment_distribution": {},
        "model_performance": {},
        "linguistic_analysis": {},
        "term_frequency": {},
        "quality_metrics": {},
    }

    try:
        if df is not None:
            report["dataset_statistics"] = {
                "total_samples": len(df),
                "data_types": dict(df.dtypes.astype(str)),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
            }

        if deep_analysis:
            basic = deep_analysis.get("basic_stats", {})
            semantic = deep_analysis.get("semantic_patterns", {})
            quality = deep_analysis.get("quality_metrics", {})
            word_stats = deep_analysis.get("word_analysis", {})

            report["linguistic_analysis"] = {
                "total_words": basic.get("total_words", 0),
                "unique_words": basic.get("unique_words", 0),
                "vocabulary_richness": basic.get("vocabulary_richness", 0),
                "avg_words_per_text": basic.get("avg_words_per_text", 0),
                "avg_chars_per_text": basic.get("avg_chars_per_text", 0),
                "positive_indicators": semantic.get("positive_indicators", 0),
                "negative_indicators": semantic.get("negative_indicators", 0),
                "sentiment_ratio": semantic.get("sentiment_ratio", 1),
                "emotion_distribution": semantic.get("emotion_distribution", {}),
            }

            report["quality_metrics"] = {
                "overall_quality_score": quality.get("quality_score", 0),
                "data_completeness": quality.get("data_completeness", 0),
                "readability_score": quality.get("readability_score", 0),
                "empty_texts": quality.get("empty_texts", 0),
                "potential_spam": quality.get("potential_spam", 0),
            }

            report["term_frequency"] = {
                "most_common_terms": word_stats.get("top_words", [])[:10],
                "rare_words_count": word_stats.get("rare_words_count", 0),
                "avg_word_length": word_stats.get("word_length_avg", 0),
            }

        if preds and metrics:
            for name, pred in preds.items():
                if name in metrics:
                    m = metrics[name]
                    unique_vals, counts = np.unique(pred, return_counts=True)
                    pred_dist = {int(k): int(v) for k, v in zip(unique_vals, counts)}
                    report["model_performance"][name] = {
                        "model_type": m.get("model_type", "Unknown"),
                        "avg_confidence": m.get("confidence_avg", 0),
                        "confidence_std": m.get("confidence_std", 0),
                        "prediction_distribution": pred_dist,
                        "total_predictions": len(pred),
                    }

        if preds:
            all_preds: List[int] = []
            for p in preds.values():
                all_preds.extend([int(x) for x in p])
            if all_preds:
                unique_vals, counts = np.unique(all_preds, return_counts=True)
                total = len(all_preds)
                label_map = {0: "negative", 1: "positive", 2: "neutral"}
                report["sentiment_distribution"] = {
                    "counts": {label_map.get(int(v), f"class_{int(v)}"): int(c) for v, c in zip(unique_vals, counts)},
                    "percentages": {label_map.get(int(v), f"class_{int(v)}"): (int(c) / total) * 100 for v, c in zip(unique_vals, counts)},
                    "total_classified": int(total),
                }

        return report
    except Exception as e:  # pragma: no cover - runtime safety
        return {"error": str(e)}


def _join_text(texts: Iterable[str]) -> str:
    return " ".join([str(t) for t in texts if isinstance(t, str)])


def extract_top_phrases(texts: Iterable[str], labels: Iterable[str] | None = None, top_n: int = 10) -> Dict[str, Any]:
    """Return top phrases optionally grouped by label."""
    result: Dict[str, Any] = {}
    if labels is not None and len(list(texts)) == len(list(labels)):
        grouped: Dict[str, List[str]] = defaultdict(list)
        for t, l in zip(texts, labels):
            grouped[str(l)].append(str(t))
        for lab, group in grouped.items():
            result[lab] = get_top_phrases(_join_text(group), n_phrases=top_n)
    else:
        result["all"] = get_top_phrases(_join_text(texts), n_phrases=top_n)
    return result


def generate_topic_summary(texts: Iterable[str], labels: Iterable[str] | None = None, top_n: int = 5) -> Dict[str, Any]:
    """Return topic clusters optionally grouped by label."""
    result: Dict[str, Any] = {}
    if labels is not None and len(list(texts)) == len(list(labels)):
        grouped: Dict[str, List[str]] = defaultdict(list)
        for t, l in zip(texts, labels):
            grouped[str(l)].append(str(t))
        for lab, group in grouped.items():
            result[lab] = extract_topics(_join_text(group))[:top_n]
    else:
        result["all"] = extract_topics(_join_text(texts))[:top_n]
    return result


def contextual_keyword_extraction(texts: Iterable[str], labels: Iterable[str] | None = None, top_n: int = 10) -> Dict[str, Any]:
    """Extract keywords grouped by label."""
    result: Dict[str, Any] = {}
    if labels is not None and len(list(texts)) == len(list(labels)):
        grouped: Dict[str, List[str]] = defaultdict(list)
        for t, l in zip(texts, labels):
            grouped[str(l)].append(str(t))
        for lab, group in grouped.items():
            joined = _join_text(group)
            result[lab] = extract_keywords_by_label(joined, lab)[:top_n]
    else:
        joined = _join_text(texts)
        result["all"] = extract_keywords_by_label(joined, "all")[:top_n]
    return result


def generate_wordclouds(texts: Iterable[str], labels: Iterable[str] | None = None) -> Dict[str, Any]:
    """Generate wordcloud images grouped by label and return base64 strings."""
    clouds: Dict[str, Any] = {}
    if not WORDCLOUD_AVAILABLE:
        return clouds

    def _wc(text: str) -> str:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    if labels is not None and len(list(texts)) == len(list(labels)):
        grouped: Dict[str, List[str]] = defaultdict(list)
        for t, l in zip(texts, labels):
            grouped[str(l)].append(str(t))
        for lab, group in grouped.items():
            clouds[lab] = _wc(_join_text(group))
    else:
        clouds["all"] = _wc(_join_text(texts))
    return clouds

__all__ = [
    "normalize_predictions",
    "safe_generate_report",
    "extract_top_phrases",
    "generate_topic_summary",
    "contextual_keyword_extraction",
    "generate_wordclouds",
]
