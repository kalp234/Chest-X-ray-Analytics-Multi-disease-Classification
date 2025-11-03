"""
Inference Pipeline - returns predictions and (optional) Grad-CAM outputs.
"""
import os
import numpy as np
import torch
from src.model_loader import load_ensemble
from backend.utils.preprocess import preprocess_image
from backend.utils.postprocess import filter_predictions
from src.gradcam_utils import ensemble_gradcam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# preload
ENSEMBLE_MODEL = load_ensemble(device=DEVICE)

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

CLASS_THRESHOLDS = [0.5] * len(LABELS)

def inference(
    image_path: str,
    with_gradcam: bool = False,
    gradcam_target: int | None = None,
    gradcam_save_dir: str | None = None,
    threshold_mode: bool = True
):
    # preprocess -> tensor on device
    img_tensor = preprocess_image(image_path).to(DEVICE)  # shape [1,3,H,W]

    # forward (no grad for predictions)
    with torch.no_grad():
        out = ENSEMBLE_MODEL(img_tensor)
        logits = None
        # safely extract logits tensor
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict):
            # prefer "logits" key else first value
            logits = out.get("logits", next(iter(out.values())))
        else:
            logits = out

        probs = logits.softmax(dim=1).cpu().numpy()[0] if torch.is_tensor(logits) else np.array(logits).ravel()

    # postprocess predictions (percent)
    if threshold_mode:
        predictions = filter_predictions(probs, LABELS, thresholds=CLASS_THRESHOLDS)
    else:
        predictions = [{"label": l, "confidence": round(float(p) * 100, 2)} for l,p in zip(LABELS, probs)]

    predictions = sorted(predictions, key=lambda x: float(x["confidence"]), reverse=True)

    gradcam_info = None
    if with_gradcam:
        if gradcam_target is None:
            top_idx = int(np.argmax(probs))
        else:
            top_idx = int(gradcam_target)

        save_dir = gradcam_save_dir or "data"
        base_name = os.path.splitext(os.path.basename(image_path))[0] + "_gradcam"

        # ensemble_gradcam expects models list and weights array; ensure correct types
        models = list(ENSEMBLE_MODEL.models)
        weights = ENSEMBLE_MODEL.weights.cpu().numpy() if hasattr(ENSEMBLE_MODEL, "weights") else np.ones(len(models))

        gradcam_info = ensemble_gradcam(
            models=models,
            input_tensor=img_tensor,
            target_class=top_idx,
            weights=weights,
            save_dir=save_dir,
            base_name=base_name
        )

    result = {
        "predictions": predictions,
        "top_label": predictions[0]["label"] if predictions else None,
        "gradcam_info": gradcam_info
    }
    return result


# debug run
if __name__ == "__main__":
    example = "data/sample_xrays/test1.jpg"
    r = inference(example, with_gradcam=True)
    print(r)
