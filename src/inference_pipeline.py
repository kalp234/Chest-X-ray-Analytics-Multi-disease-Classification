# src/inference_pipeline.py
"""
Inference Pipeline
------------------
Handles end-to-end prediction using the ensemble of 4 models (Swin, EfficientNet-B3, EfficientNet-B4, DenseNet-121).
Optionally generates Grad-CAM explanations for a target class.
"""

import os
import torch
import numpy as np
from src.model_loader import load_ensemble
from backend.utils.preprocess import preprocess_image
from backend.utils.postprocess import filter_predictions
from src.gradcam_utils import ensemble_gradcam

# ==========================================================
#  GLOBAL INITIALIZATION
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preload ensemble once on server start
ENSEMBLE_MODEL = load_ensemble(device=DEVICE)

# ChestX-ray14 labels
LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

# Optional per-class thresholds (tuned during validation)
CLASS_THRESHOLDS = [0.5] * len(LABELS)


# ==========================================================
#  MAIN INFERENCE FUNCTION
# ==========================================================
def inference(
    image_path: str,
    with_gradcam: bool = False,
    gradcam_target: int | None = None,
    gradcam_savepath: str | None = None,
    threshold_mode: bool = True
):
    """
    Run inference for a single image.
    Args:
        image_path (str): Path to X-ray image.
        with_gradcam (bool): Whether to generate Grad-CAM heatmap.
        gradcam_target (int): Class index (0–13) to explain with Grad-CAM.
        gradcam_savepath (str): Where to save Grad-CAM overlay image.
        threshold_mode (bool): If True → filter predictions by threshold, else return all.
    Returns:
        dict with:
            - predictions: list of {"label": str, "confidence": float}
            - gradcam_path: path (if generated)
    """
    # -----------------------------
    # Step 1: Preprocess image
    # -----------------------------

    img_tensor = preprocess_image(image_path).to(DEVICE)
    img_tensor.requires_grad_(True)  # ✅ allow Grad-CAM gradients


    # -----------------------------
    # Step 2: Forward pass
    # -----------------------------
    with torch.no_grad():
        probs = ENSEMBLE_MODEL(img_tensor).cpu().numpy()[0]  # shape (14,)

    # -----------------------------
    # Step 3: Postprocess predictions
    # -----------------------------
    if threshold_mode:
        predictions = filter_predictions(probs, LABELS, thresholds=CLASS_THRESHOLDS)
    else:
        # show all classes (useful for debugging)
        predictions = [{"label": l, "confidence": round(float(p)*100,2)} for l,p in zip(LABELS, probs)]

    # -----------------------------
    # Step 4: Grad-CAM (optional)
    # -----------------------------
    gradcam_path = None
    if with_gradcam and gradcam_target is not None:
        gradcam_path = gradcam_savepath or os.path.join("data", "gradcam_overlay.png")
    try:
        gradcam_path, boxes = ensemble_gradcam(
            models=list(ENSEMBLE_MODEL.models),
            input_tensor=img_tensor,
            target_class=gradcam_target,
            weights=ENSEMBLE_MODEL.weights.cpu().numpy(),
            labels=LABELS,  # ✅ Add this line
            save_path=gradcam_path,
        )
    except Exception as e:
        print(f"[GradCAM Error] {e}")
        gradcam_path = None
        boxes = []


    # -----------------------------
    # Step 5: Return results
    # -----------------------------
    result = {
        "predictions": predictions,
        "gradcam_path": gradcam_path,
    }
    return result


# ==========================================================
#  DEBUG TEST (run standalone)
# ==========================================================
if __name__ == "__main__":
    test_img = "data/sample_xrays/test1.jpg"  # put a test image here
    result = inference(test_img, with_gradcam=True, gradcam_target=2)
    print("Predictions:")
    for r in result["predictions"]:
        print(f" - {r['label']}: {r['confidence']}%")
    print("Grad-CAM:", result["gradcam_path"])
