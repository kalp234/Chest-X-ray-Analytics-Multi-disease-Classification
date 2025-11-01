# backend/utils/postprocess.py
import numpy as np

def filter_predictions(probs, labels, thresholds=None):
    """
    Postprocess ensemble output probabilities into a readable format.

    Keeps only classes above threshold. Ensures at least one label appears.
    Args:
      probs: np.ndarray of shape (num_classes,)
      labels: list of class names
      thresholds: list or scalar threshold(s)
    Returns:
      list of dicts: [{ "label": str, "confidence": float }]
    """
    if thresholds is None:
        thresholds = [0.5] * len(labels)
    elif isinstance(thresholds, (float, int)):
        thresholds = [thresholds] * len(labels)

    preds = []
    for label, p, t in zip(labels, probs, thresholds):
        if float(p) >= float(t):
            preds.append({
                "label": label,
                "confidence": round(float(p) * 100, 2)
            })

    # --- fallback: always return top-1 ---
    if not preds:
        top_idx = int(np.argmax(probs))
        preds.append({
            "label": labels[top_idx],
            "confidence": round(float(probs[top_idx]) * 100, 2)
        })

    # --- sort descending by confidence for cleaner UI ---
    preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    return preds
