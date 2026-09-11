export const MODEL_LABELS = {
  new_xception: "Xception71",
  new_efficientnet: "EfficientNet-B5",
  new_vit_small: "ViT-Small",
  new_vit_large_clip: "ViT-Large/CLIP",
  // Legacy keys
  cnn: "Xception71",
  efficientnet: "EfficientNet-B5",
  vit: "ViT-Small",
};

export function formatModelScores(scores = {}) {
  return Object.entries(scores)
    .map(([k, v]) => (MODEL_LABELS[k] || k) + ": " + (+v * 100).toFixed(1) + "%")
    .join(" · ");
}