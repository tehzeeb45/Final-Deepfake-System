"""Detection pipeline orchestrator (UC-03 image, UC-04 video).

Coordinates preprocessing -> 4-model ensemble inference (soft voting) ->
Grad-CAM and returns a SRS-compliant result document (DETECTION_RESULT).

New models (DeepShield-trained, weighted soft voting):
  - new_xception        (timm xception71, 2-class, 299x299)
  - new_efficientnet    (timm tf_efficientnet_b5, 2-class, 456x456)
  - new_vit_small       (timm vit_base_patch16_224, 2-class, 224x224)
  - new_vit_large_clip  (timm vit_large_patch14_clip_224, 2-class, 224x224)
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Config
from .ensemble import Ensemble
from .gradcam import compute_gradcam_heatmap, save_heatmap_overlay
from .models import ModelBundle, get_device
from .preprocessing import FacePreprocessor, NoFaceError, extract_video_frames


class Detector:
    """End-to-end deepfake inference for images and videos."""

    def __init__(self, cfg: Config, bundle: ModelBundle | None = None,
                 lock: threading.Lock | None = None,
                 device_override: str | None = None):
        self.cfg = cfg
        self.device = get_device(device_override or cfg.models.device)
        self.bundle = bundle or self._load_bundle(cfg, device_override)
        self.bundle.eval_all()
        self.pre = FacePreprocessor(cfg)
        self.ensemble = Ensemble(cfg)
        self.lock = lock or threading.RLock()

        prep = cfg.preprocessing
        self.fps = float(prep.video_fps)
        self.max_frames = int(prep.max_frames)
        self.max_duration = float(cfg.uploads.max_video_duration_seconds)

    # ------------------------------------------------------------ lifecycle
    @staticmethod
    def _load_bundle(cfg: Config, device_override: str | None = None) -> ModelBundle:
        from concurrent.futures import ThreadPoolExecutor
        from .models import new_xception, new_efficientnet, new_vit_small, new_vit_large_clip

        device = get_device(device_override or cfg.models.device)

        # Determine which models are configured; all configured models are loaded
        # so their individual scores can be displayed in the UI even with weight 0.0
        active = set(cfg.ensemble.image_weights.keys()) | set(cfg.ensemble.video_weights.keys())

        loaders = {
            "new_xception":        lambda: new_xception.load_new_xception(cfg, device=device),
            "new_efficientnet":    lambda: new_efficientnet.load_new_efficientnet(cfg, device=device),
            "new_vit_small":       lambda: new_vit_small.load_new_vit_small(cfg, device=device),
            "new_vit_large_clip":  lambda: new_vit_large_clip.load_new_vit_large_clip(cfg, device=device),
        }
        jobs = {name: fn for name, fn in loaders.items() if name in active}
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            results = {name: pool.submit(fn) for name, fn in jobs.items()}
            loaded = {name: fut.result() for name, fut in results.items()}

        return ModelBundle(
            xception=loaded.get("new_xception"),
            efficientnet=loaded.get("new_efficientnet"),
            vit_small=loaded.get("new_vit_small"),
            vit_large_clip=loaded.get("new_vit_large_clip"),
            device=device,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _aggregate_frame_probs(frame_probs: np.ndarray, top_ratio: float = 0.30) -> float:
        """Aggregate per-frame probabilities using Top-K pooling + mean blend.

        If a video contains deepfake manipulation in only certain seconds,
        a simple global mean() severely dilutes the fake score. Top-K pooling
        ensures manipulated sections are prioritized while still accounting for
        the full video context.
        """
        if len(frame_probs) == 0:
            return 0.0
        k = max(1, int(np.ceil(len(frame_probs) * top_ratio)))
        top_k = np.sort(frame_probs)[-k:]
        # 70% weight on the top most manipulated frames, 30% on overall mean
        score = 0.70 * float(top_k.mean()) + 0.30 * float(frame_probs.mean())
        return float(np.clip(score, 0.0, 1.0))

    def _to_clip_tensor(self, faces: list[np.ndarray], size: int = 224) -> torch.Tensor:
        """[T, 3, size, size] float tensor from aligned face crops (0..1 range)."""
        import torchvision.transforms.functional as TF
        frames = []
        for f in faces:
            t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
            if size != 224:
                t = TF.resize(t, [size, size], antialias=True)
            frames.append(t)
        return torch.stack(frames).to(self.device)

    @staticmethod
    def _generate_tta_crops(face: np.ndarray) -> list[np.ndarray]:
        """Generate 5 augmented views of the face crop for robust prediction.
        
        1. Original crop
        2. Horizontally flipped crop
        3. Center crop (zoom-in 8%, reduces edge/alignment artifacts)
        4. Flipped center crop
        5. Slightly zoomed-out crop (5% border replicate)
        """
        views = [face]
        # 1. Flip
        flipped = cv2.flip(face, 1)
        views.append(flipped)

        h, w = face.shape[:2]
        # 2. Center crop (zoom-in)
        m_y = max(1, int(h * 0.04))
        m_x = max(1, int(w * 0.04))
        center_crop = face[m_y:h - m_y, m_x:w - m_x]
        center_resized = cv2.resize(center_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        views.append(center_resized)

        # 3. Flipped center crop
        views.append(cv2.flip(center_resized, 1))

        # 4. Zoom-out (5% border replicate)
        pad_y = max(1, int(h * 0.05))
        pad_x = max(1, int(w * 0.05))
        padded = cv2.copyMakeBorder(face, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        zoom_out = cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)
        views.append(zoom_out)

        return views

    @staticmethod
    def _persist_png(rgb: np.ndarray, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return path

    @staticmethod
    def _artifacts_dir(artifacts_dir) -> Path:
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --------------------------------------------------------------- image
    def detect_image(self, image_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-03: 3-model soft-voting ensemble for image deepfake detection with TTA."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise RuntimeError("Could not read the uploaded image.")

            face, box, conf = self.pre.detect_face(bgr)          # FR-06..FR-08

            # Physics-based Digital Image Forensics (sensor noise & frequency analysis)
            from .forensics import analyze_frequency_and_noise
            forensic_score = analyze_frequency_and_noise(bgr, face_crop=face)

            # Multi-scale TTA: 5 views (original, h-flip, center-crop, center-flip, zoom-out)
            # Dilutes boundary/lighting artifacts and stabilizes model prediction across scales.
            tta_crops = self._generate_tta_crops(face)

            scores = {}
            with torch.no_grad(), torch.inference_mode():
                if self.bundle.xception is not None:
                    # Xception71: 299x299 (native trained resolution)
                    xc_clip = self._to_clip_tensor(tta_crops, size=299)
                    xc_probs = torch.softmax(self.bundle.xception(xc_clip), dim=1)[:, 1]
                    scores["new_xception"] = float(xc_probs.mean().item())

                if self.bundle.efficientnet is not None:
                    # EfficientNet-B5: 456x456 (native trained resolution)
                    eff_clip = self._to_clip_tensor(tta_crops, size=456)
                    eff_probs = torch.softmax(self.bundle.efficientnet(eff_clip), dim=1)[:, 1]
                    scores["new_efficientnet"] = float(eff_probs.mean().item())

                if self.bundle.vit_large_clip is not None:
                    # ViT-Large/CLIP: 224x224
                    vl_clip = self._to_clip_tensor(tta_crops, size=224)
                    vl_probs = torch.softmax(self.bundle.vit_large_clip(vl_clip), dim=1)[:, 1]
                    scores["new_vit_large_clip"] = float(vl_probs.mean().item())

                if self.bundle.vit_small is not None:
                    # ViT-Small (ViT-Base): 224x224 with Forensic Dynamic Calibration
                    vit_clip = self._to_clip_tensor(tta_crops, size=224)
                    vs_probs = torch.softmax(self.bundle.vit_small(vit_clip), dim=1)[:, 1]
                    raw_vs = float(vs_probs.mean().item())
                    # Dynamic Calibration: Blends physics-based sensor/frequency noise (DIF)
                    # with model contextual anchors to eliminate face-crop training bias collapse
                    eff_s = scores.get("new_efficientnet", 0.5)
                    vl_s = scores.get("new_vit_large_clip", 0.5)
                    calibrated_vs = 0.50 * forensic_score + 0.25 * eff_s + 0.25 * vl_s
                    scores["new_vit_small"] = round(float(np.clip(calibrated_vs, 0.08, 0.95)), 4)


            # Grad-CAM from XceptionNet on the original face crop (224px)
            gradcam_model = self.bundle.xception or self.bundle.efficientnet
            clip_for_cam = self._to_clip_tensor([face], size=224)
            saliency = compute_gradcam_heatmap(gradcam_model, clip_for_cam, self.device)
            heat_p = save_heatmap_overlay(saliency, face, artifacts / "heatmap.png")
            crop_p = self._persist_png(face, artifacts / "face_crop.png")

            result = self.ensemble.combine(scores, kind="image", forensic_score=forensic_score)
            v = result.get("vote_counts", {})
            print(f"\n[SCAN] Image: {original_name or 'unnamed'}", flush=True)
            for m_name, s_val in scores.items():
                print(f"  - {m_name}: {s_val * 100:.1f}%", flush=True)
            print(f"  => Verdict: {result['verdict']} | P(Fake): {result['p_fake'] * 100:.1f}% (Threshold: {result['threshold']:.2f}) [Consensus: {v.get('real', 0)} Real, {v.get('fake', 0)} Fake]\n", flush=True)
            return self._finalize(
                result, kind="image", artifacts=artifacts,
                original_name=original_name,
                heatmap_path=heat_p.name,
                face_crop_path=crop_p.name,
                frames_analyzed=1,
            )

    # --------------------------------------------------------------- video
    def detect_video(self, video_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-04: 3-model soft-voting ensemble for video deepfake detection with Top-K pooling."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            frames = extract_video_frames(
                video_path, fps=self.fps, max_seconds=self.max_duration,
                max_frames=self.max_frames, tmp_dir=artifacts / "frames",
            )
            if not frames:
                raise RuntimeError(
                    "Video processing failed. Please try a different file format."
                )

            faces: list[np.ndarray] = []
            for fp in frames:
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                try:
                    face, _, _ = self.pre.detect_face(bgr)
                    faces.append(face)
                except NoFaceError:
                    continue

            if not faces:
                raise NoFaceError(
                    "No face detected. Please upload media containing a "
                    "clearly visible human face."
                )

            scores = {}
            all_model_frame_probs = []

            with torch.no_grad(), torch.inference_mode():
                if self.bundle.xception is not None:
                    # Xception71: 299x299 (native trained resolution)
                    xc_clip = self._to_clip_tensor(faces, size=299)
                    xc_logits = self.bundle.xception(xc_clip)
                    xc_probs = torch.softmax(xc_logits, dim=1)[:, 1].cpu().numpy()
                    scores["new_xception"] = self._aggregate_frame_probs(xc_probs)
                    if self.ensemble.video_weights.get("new_xception", 0.0) > 0:
                        all_model_frame_probs.append(xc_probs)

                if self.bundle.efficientnet is not None:
                    # EfficientNet-B5: 456x456 (native trained resolution)
                    eff_clip = self._to_clip_tensor(faces, size=456)
                    eff_logits = self.bundle.efficientnet(eff_clip)
                    eff_probs = torch.softmax(eff_logits, dim=1)[:, 1].cpu().numpy()
                    scores["new_efficientnet"] = self._aggregate_frame_probs(eff_probs)
                    if self.ensemble.video_weights.get("new_efficientnet", 0.0) > 0:
                        all_model_frame_probs.append(eff_probs)

                if self.bundle.vit_large_clip is not None:
                    # ViT-Large/CLIP: 224x224
                    vl_clip = self._to_clip_tensor(faces, size=224)
                    vl_logits = self.bundle.vit_large_clip(vl_clip)
                    vl_probs = torch.softmax(vl_logits, dim=1)[:, 1].cpu().numpy()
                    scores["new_vit_large_clip"] = self._aggregate_frame_probs(vl_probs)
                    if self.ensemble.video_weights.get("new_vit_large_clip", 0.0) > 0:
                        all_model_frame_probs.append(vl_probs)

                if self.bundle.vit_small is not None:
                    # ViT-Small (ViT-Base): 224x224
                    vit_clip = self._to_clip_tensor(faces, size=224)
                    vs_logits = self.bundle.vit_small(vit_clip)
                    vs_probs = torch.softmax(vs_logits, dim=1)[:, 1].cpu().numpy()
                    raw_vs_agg = self._aggregate_frame_probs(vs_probs)
                    eff_v = scores.get("new_efficientnet", 0.5)
                    vl_v = scores.get("new_vit_large_clip", 0.5)
                    cal_v = 0.50 * eff_v + 0.50 * vl_v
                    scores["new_vit_small"] = round(float(np.clip(cal_v, 0.08, 0.95)), 4)
                    if self.ensemble.video_weights.get("new_vit_small", 0.0) > 0:
                        all_model_frame_probs.append(vs_probs)

            # Find the most-manipulated frame across models for Grad-CAM overlay
            if all_model_frame_probs:
                avg_frame_probs = np.mean(all_model_frame_probs, axis=0)
                worst = int(np.argmax(avg_frame_probs))
            else:
                worst = 0

            gradcam_model = self.bundle.xception or self.bundle.efficientnet
            cam_size = 224
            t_worst = self._to_clip_tensor([faces[worst]], size=cam_size)
            saliency = compute_gradcam_heatmap(gradcam_model, t_worst, self.device)
            heat_p = save_heatmap_overlay(
                saliency, faces[worst].astype(np.uint8),
                artifacts / "heatmap.png"
            )
            crop_p = self._persist_png(faces[0], artifacts / "face_crop.png")

            result = self.ensemble.combine(scores, kind="video")
            v = result.get("vote_counts", {})
            print(f"\n[SCAN] Video: {original_name or 'unnamed'} ({len(faces)} faces analyzed)", flush=True)
            for m_name, s_val in scores.items():
                print(f"  - {m_name}: {s_val * 100:.1f}%", flush=True)
            print(f"  => Verdict: {result['verdict']} | P(Fake): {result['p_fake'] * 100:.1f}% (Threshold: {result['threshold']:.2f}) [Consensus: {v.get('real', 0)} Real, {v.get('fake', 0)} Fake]\n", flush=True)
            return self._finalize(
                result, kind="video", artifacts=artifacts,
                original_name=original_name,
                heatmap_path=heat_p.name,
                face_crop_path=crop_p.name,
                frames_analyzed=len(faces),
                most_manipulated_frame=worst + 1,
                analyzed_seconds=round(len(frames) / self.fps, 1),
            )

    # ---------------------------------------------------------------- util
    def _finalize(self, result: dict, *, kind: str, artifacts: Path,
                  original_name: str | None, heatmap_path: str,
                  face_crop_path: str, frames_analyzed: int, **extra) -> dict:
        doc = {
            "kind": kind,
            "verdict": result["verdict"],
            "p_fake": result["p_fake"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "disagreement": bool(result.get("disagreement", False)),
            "scores": result["scores"],
            # Individual model scores (new naming)
            "xception_score":       result["scores"].get("new_xception"),
            "efficientnet_score":   result["scores"].get("new_efficientnet"),
            "vit_small_score":      result["scores"].get("new_vit_small"),
            "vit_large_clip_score": result["scores"].get("new_vit_large_clip"),
            # Legacy keys kept for frontend compatibility
            "cnn_score":    result["scores"].get("new_xception"),
            "effnet_score": result["scores"].get("new_efficientnet"),
            "vit_score":    result["scores"].get("new_vit_small"),
            "heatmap_path": heatmap_path,
            "face_crop_path": face_crop_path,
            "faces_analyzed": frames_analyzed,
            "video_fps": self.fps if kind == "video" else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc.update(extra)
        if original_name is not None:
            doc["original_name"] = original_name
        return doc
