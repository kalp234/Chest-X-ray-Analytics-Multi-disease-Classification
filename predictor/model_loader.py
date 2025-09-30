import base64
import os
from .gradcam_explainer import predict_image


def encode_image_to_base64(path):
    """Convert saved image into base64 string for JSON response."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_inference(image_path):
    results = predict_image(image_path)

    predictions = []
    gradcam_b64, bbox_b64 = None, None

    if isinstance(results, dict) and "predictions" in results:
        saved_path = results.get("saved_path")
        if saved_path and os.path.exists(saved_path):
            gradcam_b64 = f"data:image/png;base64,{encode_image_to_base64(saved_path)}"

        bbox_path = results.get("bbox_path")
        if bbox_path and os.path.exists(bbox_path):
            bbox_b64 = f"data:image/png;base64,{encode_image_to_base64(bbox_path)}"

        top1_label = results.get("top1_label")  # ✅ actual top1 used for GradCAM

        # keep original order (unsorted)
        for label, conf in results["predictions"]:
            predictions.append({
                "label": label,
                "confidence": round(float(conf) * 100, 2),
                "bbox": bbox_b64 if label == top1_label else None  # ✅ attach bbox only to real top1
            })
    else:
        print("⚠️ Unexpected results format from predict_image:", results)

    return {
        "predictions": predictions,
        "gradcam": gradcam_b64
    }

