# backend/routes/inference_routes.py
from fastapi import APIRouter, UploadFile, File
import tempfile
from src.inference_pipeline import inference

router = APIRouter(tags=["Predict"])

@router.post("/")  # ✅ this defines POST /predict/
async def predict(file: UploadFile = File(...)):
    """Run ensemble inference for uploaded image"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = inference(tmp_path, with_gradcam=False)
        return {"status": "success", "predictions": result["predictions"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}
