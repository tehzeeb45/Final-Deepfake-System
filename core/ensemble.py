"""Ensemble classifier — pure weighted soft voting (FR-14).

Combines 4-model scores using weighted average only. No override rules.
"""
from __future__ import annotations

from .config import Config


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Plain weighted mean of P(Fake) over the models actually present in `scores`."""
    total_w = 0.0
    acc = 0.0
    for key, p in scores.items():
        w = weights.get(key, 0.0)
        if w <= 0:
            continue
        acc += w * p
        total_w += w
    if total_w <= 0:
        raise ValueError("ensemble: no weighted model scores provided")
    return acc / total_w


class Ensemble:
    """Configured weighted-soft-voting helper with forensic tiebreaker."""

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
        
        # Calculate active models and their individual votes
        active_scores = {k: v for k, v in scores.items() if weights.get(k, 0.0) > 0}
        fake_votes = sum(1 for v in active_scores.values() if v >= threshold)
        real_votes = len(active_scores) - fake_votes
        majority_is_fake = fake_votes > real_votes

        p_fake = weighted_average(scores, weights)

        # Majority Consensus Gate: REMOVED
        # Was clamping 2-2 ties to REAL, causing 26 Fake images to be misclassified.
        # Plain weighted average with calibrated weights now handles outliers correctly.

        # Forensic tiebreaker: use physics-based camera noise analysis
        # to correct false positives when neural networks disagree
        p_fake = self._apply_forensic_correction(
            scores, p_fake, forensic_score, kind, threshold
        )
        result = self.to_verdict(p_fake, threshold=threshold)
        result["scores"] = {k: round(v, 4) for k, v in scores.items()}
        result["forensic_score"] = round(forensic_score, 4)
        result["disagreement"] = self._check_disagreement(scores)
        result["majority_vote"] = "FAKE" if majority_is_fake else "REAL"
        result["vote_counts"] = {"fake": fake_votes, "real": real_votes}
        return result

    def _apply_forensic_correction(
        self, scores: dict, p_fake: float, forensic_score: float,
        kind: str, threshold: float,
    ) -> float:
        """Forensic score logging only — no override applied.

        Override was disabled because it caused some fakes to be
        misclassified as Real. Neural networks decide the verdict.
        """
        if kind != "image":
            return p_fake

        xc = scores.get("new_xception", 0.5)
        eff = scores.get("new_efficientnet", 0.5)
        vit = scores.get("new_vit_large_clip", 0.5)

        # Debug logging only — no score modification
        print(f"  [Forensic] Xc={xc:.1%} Eff={eff:.1%} ViT={vit:.1%} "
              f"forensic={forensic_score:.3f} p_fake={p_fake:.3f}")

        # Override DISABLED — let trained neural networks decide
        return p_fake

    @staticmethod
    def _check_disagreement(scores: dict) -> bool:
        """Flag when models strongly disagree (one > 80%, another < 20%)."""
        active = [v for k, v in scores.items()
                  if k != "new_vit_small" and v is not None]
        if len(active) < 2:
            return False
        return max(active) > 0.80 and min(active) < 0.20

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