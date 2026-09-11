"""Ensemble classifier — pure weighted soft voting (FR-14).

Combines 4-model scores using weighted average only. No override rules.
"""
from __future__ import annotations

from .config import Config


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of P(Fake) over the models actually present in `scores`."""
    total_w = 0.0
    acc = 0.0
    for key, p in scores.items():
        w = weights.get(key, 0.0)
        acc += w * p
        total_w += w
    if total_w <= 0:
        raise ValueError("ensemble: no weighted model scores provided")
    return acc / total_w


class Ensemble:
    """Configured weighted-soft-voting helper."""

    def __init__(self, cfg: Config):
        default_thresh = float(cfg.ensemble.get("threshold", 0.50))
        self.image_threshold = float(cfg.ensemble.get("image_threshold", 0.40))
        self.video_threshold = float(cfg.ensemble.get("video_threshold", default_thresh))
        self.image_weights = {k: float(v) for k, v in cfg.ensemble.image_weights.items()}
        self.video_weights = {k: float(v) for k, v in cfg.ensemble.video_weights.items()}

    def combine(self, scores: dict[str, float], kind: str,
                forensic_score: float = 0.0) -> dict:
        weights = self.video_weights if kind == "video" else self.image_weights
        threshold = self.video_threshold if kind == "video" else self.image_threshold
        p_fake = weighted_average(scores, weights)
        result = self.to_verdict(p_fake, threshold=threshold)
        result["scores"] = {k: round(v, 4) for k, v in scores.items()}
        result["disagreement"] = False
        return result

    def to_verdict(self, p_fake: float, threshold: float | None = None) -> dict:
        thresh = self.image_threshold if threshold is None else threshold
        return {
            "verdict": "FAKE" if p_fake >= thresh else "REAL",
            "p_fake": round(float(p_fake), 4),
            "confidence": round(
                p_fake * 100.0 if p_fake >= thresh else (1.0 - p_fake) * 100.0,
                2,
            ),
            "threshold": thresh,
        }