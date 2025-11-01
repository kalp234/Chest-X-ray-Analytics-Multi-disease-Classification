from fastapi import APIRouter, UploadFile, File, Form
import os, base64
from src.inference_pipeline import inference

router = APIRouter(tags=["Explain"])

def encode_image_to_base64(path):
    """Utility: convert saved Grad-CAM image into base64 string for frontend"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@router.post("/")
async def explain(file: UploadFile = File(...), gradcam_class: int = Form(...)):
    """Generate Grad-CAM and optional explainability metrics."""
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)

    # Save uploaded file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        # Run Grad-CAM inference
        result = inference(filepath, with_gradcam=True, gradcam_target=gradcam_class)

        gradcam_path = result.get("gradcam_path")
        gradcam_b64 = None

        if gradcam_path and os.path.exists(gradcam_path):
            print(f"[DEBUG] Grad-CAM saved at: {gradcam_path}")

            gradcam_b64 = f"data:image/png;base64,{encode_image_to_base64(gradcam_path)}"

        response = {
            "status": "success",
            "gradcam_b64": gradcam_b64,
            "predictions": result.get("predictions"),
            "metrics": result.get("metrics", None),  # optional explainability metrics
            "bboxes": result.get("bboxes", []),
        }

        return response

    except Exception as e:
        print(f"[ERROR] /explain -> {e}")
        return {"status": "error", "message": str(e)}
