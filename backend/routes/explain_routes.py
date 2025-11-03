from fastapi import APIRouter, UploadFile, File, Form
import os, base64
from src.inference_pipeline import inference

router = APIRouter(tags=["Explain"])

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@router.post("/")
async def explain(file: UploadFile = File(...), gradcam_class: int | None = Form(None)):
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)

    # save uploaded
    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        result = inference(filepath, with_gradcam=True, gradcam_target=gradcam_class, gradcam_save_dir="data")
        gradcam_info = result.get("gradcam_info") or {}

        # prepare base64 outputs
        heatmap_b64 = None
        overlay_b64 = None
        bbox_b64 = None
        triple_b64 = None

        def _enc(p):
            return f"data:image/png;base64,{encode_image_to_base64(p)}" if p and os.path.exists(p) else None

        heatmap_b64 = _enc(gradcam_info.get("heatmap_path"))
        overlay_b64 = _enc(gradcam_info.get("overlay_path"))
        bbox_b64 = _enc(gradcam_info.get("bbox_path"))
        triple_b64 = _enc(gradcam_info.get("triple_path"))
        original_b64 = _enc(filepath)

        response = {
            "status": "success",
            "original_b64": original_b64,
            "heatmap_b64": heatmap_b64,
            "overlay_b64": overlay_b64,
            "bbox_b64": bbox_b64,
            "triple_b64": triple_b64,
            "bbox": gradcam_info.get("bbox"),
            "predictions": result.get("predictions"),
            "top_label": result.get("top_label")
        }
        return response

    except Exception as e:
        print("[ERROR] /explain ->", e)
        return {"status": "error", "message": str(e)}
