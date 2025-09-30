# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import default_storage
# import os

# from .model_loader import run_inference


# @csrf_exempt
# def predict_view(request):
#     if request.method == "POST":
#         file = request.FILES.get("file")
#         if not file:
#             return JsonResponse({"error": "No file uploaded"}, status=400)

#         # Save uploaded file temporarily
#         file_path = default_storage.save(file.name, file)
#         abs_path = os.path.join(default_storage.location, file_path)

#         try:
#             result = run_inference(abs_path)

#             # format predictions for JSON
#             predictions = [
#                 {"label": label, "confidence": round(conf * 100, 2)}
#                 for label, conf in result["predictions"]
#             ]

#             top3 = [
#                 {"label": label, "confidence": round(conf * 100, 2)}
#                 for label, conf in result["top3"]
#             ]

#             response = {
#                 "predictions": predictions,
#                 "top3": top3,
#                 "probabilities": result["probabilities"],
#                 "gradcam": result.get("gradcam_base64"),
#             }

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#         finally:
#             if os.path.exists(abs_path):
#                 os.remove(abs_path)

#         return JsonResponse(response)

#     return JsonResponse({"error": "POST request required"}, status=400)


# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import default_storage
# import os

# from .model_loader import run_inference


# @csrf_exempt
# def predict_view(request):
#     if request.method == "POST":
#         file = request.FILES.get("file")
#         if not file:
#             return JsonResponse({"error": "No file uploaded"}, status=400)

#         # Save file temporarily
#         file_path = default_storage.save(file.name, file)
#         abs_path = os.path.join(default_storage.location, file_path)

#         try:
#             result = run_inference(abs_path)

#             predictions = [
#                 {"label": label, "confidence": round(conf, 2)}
#                 for label, conf in result["predictions"]
#             ]

#             response = {
#                 "predictions": predictions,
#                 "probabilities": result["probabilities"],
#                 "gradcam": result.get("gradcam_base64"),
#             }

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#         finally:
#             if os.path.exists(abs_path):
#                 os.remove(abs_path)

#         return JsonResponse(response)

#     return JsonResponse({"error": "POST request required"}, status=400)


# predictor/views.py

# predictor/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os

from .model_loader import run_inference
from .gradcam_explainer import predict_image
from django.conf import settings


@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        file_path = default_storage.save(file.name, file)
        abs_path = os.path.join(default_storage.location, file_path)

        try:
            result = run_inference(abs_path)
            return JsonResponse(result, safe=False)
        except Exception as e:
            import traceback
            print("❌ ERROR in predict_view:", traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)
        finally:
            if os.path.exists(abs_path):
                os.remove(abs_path)

    return JsonResponse({"error": "POST request required"}, status=400)




