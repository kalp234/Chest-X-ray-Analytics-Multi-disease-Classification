# src/gradcam_utils.py
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from collections.abc import Iterable
from typing import Optional, Tuple

# ----------------------------
# utilities to safely extract tensors from nested outputs
# ----------------------------
def _find_tensor(obj):
    """Recursively search for the first torch.Tensor in obj (tuple/list/dict or tensor)."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _find_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for item in obj:
            t = _find_tensor(item)
            if t is not None:
                return t
    return None


def _find_first_grad_tensor(grad_out):
    """grad_out may be a tensor, or tuple/list of tensors (possibly nested). Return first tensor."""
    return _find_tensor(grad_out)


# ===============================================================
# GRADCAM CORE CLASS
# ===============================================================
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # register hooks
        target_layer.register_forward_hook(self.forward_hook)
        # use full backward hook for PyTorch 2.x compatibility
        try:
            target_layer.register_full_backward_hook(self.backward_hook)
        except Exception:
            # fallback for older versions
            target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # output might be a tensor or tuple/list — find tensor
        try:
            t = _find_tensor(output)
            if t is None:
                # fallback: store raw output (helps debugging)
                self.activations = None
                # print("[GradCAM debug] forward_hook: no tensor found in output, stored raw output type:", type(output))
            else:
                # detach activation to avoid keeping computation graph
                self.activations = t.detach()
        except Exception as e:
            self.activations = None
            # print("[GradCAM forward_hook error]", e)

    def backward_hook(self, module, grad_in, grad_out):
        # grad_out may be a tensor or nested structure — extract first tensor
        try:
            g = _find_first_grad_tensor(grad_out)
            if g is None:
                g = _find_first_grad_tensor(grad_in)
            if g is None:
                self.gradients = None
            else:
                self.gradients = g.detach()
        except Exception as e:
            self.gradients = None
            # print("[GradCAM backward_hook error]", e)

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        """
        Compute Grad-CAM heatmap for the given class index.
        Returns a 2D numpy array normalized to [0,1] with shape (H, W).
        """
        # forward (do NOT wrap with no_grad — we need gradients)
        # zero grads first
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Extract logits-like tensor from possibly nested outputs
        logits = _find_tensor(output)
        if logits is None:
            raise ValueError("Unable to find a tensor (logits) in model output.")

        # Ensure logits is at least 2D (B, C)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        # compute class score and backward
        score = logits[:, class_idx].sum()

        # zero again before backward to be safe
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # check activations and gradients
        if self.activations is None:
            raise RuntimeError("GradCAM: activations were not captured by forward hook.")
        if self.gradients is None:
            raise RuntimeError("GradCAM: gradients were not captured by backward hook.")

        # compute weights and cam
        # gradients: (B, C, Hg, Wg), activations: (B, C, Ha, Wa)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, Ha, Wa)
        cam = F.relu(cam)

        # interpolate to input spatial dims
        try:
            h, w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        except Exception:
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # detach and return numpy 2D
        cam_np = cam.detach().cpu().numpy()[0, 0]
        return cam_np


# ===============================================================
# AUTO-LAYER DETECTOR
# ===============================================================
def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Finds the last Conv2d module inside a model. If not found, raises ValueError.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


# ===============================================================
# GENERATE GRADCAM FOR A SINGLE MODEL
# ===============================================================
def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int, architecture: str):
    """
    Safe wrapper that attempts to compute Grad-CAM for `model`. On failure returns zeros heatmap.
    """
    try:
        target_layer = find_last_conv_layer(model)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor, target_class)
        # ensure float32 numpy
        cam = np.array(cam, dtype=np.float32)
        return cam
    except Exception as e:
        print(f"[GradCAM Error in {architecture}] {e}")
        h, w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        return np.zeros((h, w), dtype=np.float32)


# ===============================================================
# ENSEMBLE GRADCAM ACROSS MODELS
# ===============================================================
def ensemble_gradcam(
    models,
    input_tensor,
    target_class,
    weights,
    labels=None,
    save_path="gradcam_result.png",
):
    import cv2
    import numpy as np

    all_heatmaps = []

    if not input_tensor.requires_grad:
        input_tensor.requires_grad_(True)

    # Generate individual model GradCAMs
    for i, model in enumerate(models):
        arch = type(model).__name__.lower()
        cam = generate_gradcam(model, input_tensor, target_class, arch)
        all_heatmaps.append(weights[i] * cam)

    # Combine and normalize ensemble CAM
    combined = np.sum(all_heatmaps, axis=0)
    combined = np.maximum(combined, 0)
    combined = (combined - combined.min()) / (combined.max() + 1e-8)
    heatmap = np.uint8(255 * combined)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # Reconstruct grayscale X-ray
    img_np = input_tensor.detach().cpu().numpy()[0]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_uint8 = np.uint8(255 * img_np)

    if img_uint8.shape[0] == 3:
        xray_img = np.transpose(img_uint8, (1, 2, 0))
    elif img_uint8.shape[0] == 1:
        xray_img = np.repeat(img_uint8[0, :, :][:, :, None], 3, axis=2)
    else:
        xray_img = np.transpose(img_uint8, (1, 2, 0))

    # Blend heatmap with original
    overlay = cv2.addWeighted(xray_img, 0.6, heatmap_color, 0.4, 0)
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)

    # -------------------------------------------------------------
    #  BOUNDING BOX EXTRACTION + CLASS LABELS
    # -------------------------------------------------------------
    boxes = []
    try:
        thresh = np.uint8(heatmap > np.percentile(heatmap, 90))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_name = labels[target_class] if labels and target_class < len(labels) else "Region"

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 80:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "label": label_name})
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    overlay,
                    label_name,
                    (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    except Exception as e:
        print(f"[GradCAM BBox Error] {e}")

    cv2.imwrite(save_path, overlay)
    return save_path, boxes

