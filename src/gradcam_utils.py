import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- GradCAM core (same as before, safe detach + eval)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        elif isinstance(output, dict):
            output = list(output.values())[0]

        loss = output[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
        # normalize to [0,1]
        cam_np -= cam_np.min()
        cam_np /= (cam_np.max() + 1e-8)
        return cam_np

def find_last_conv_layer(model):
    last_conv = None
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv

def ensemble_gradcam(models,
                     input_tensor,
                     target_class,
                     weights=None,
                     save_dir="data",
                     base_name="gradcam",
                     colormap=cv2.COLORMAP_TURBO,
                     pre_blur=(13,13),
                     post_blur=(7,7),
                     max_alpha=0.65,
                     percent_for_bbox=92):
    """
    Produces a soft medical-looking Grad-CAM overlay while keeping x-ray anatomy visible.
    Key change: per-pixel alpha map = activation * max_alpha (so strong hotspots more opaque).
    """

    os.makedirs(save_dir, exist_ok=True)
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # ensure grad enabled for input if needed
    if not input_tensor.requires_grad:
        input_tensor.requires_grad_(True)

    all_maps = []
    for i, model in enumerate(models):
        try:
            model.eval()
            target_layer = find_last_conv_layer(model)
            gc = GradCAM(model, target_layer)
            hm = gc.generate(input_tensor, target_class)
            all_maps.append(weights[i] * hm)
        except Exception as e:
            print(f"[GradCAM Error] model {i}: {e}")

    if not all_maps:
        raise RuntimeError("No heatmaps produced")

    combined = np.sum(all_maps, axis=0).astype(np.float32)
    combined = np.maximum(combined, 0.0)

    # percentile-based contrast stretch to emphasize mid/high activations, reduce noise
    low = np.percentile(combined, 2.0)
    high = np.percentile(combined, 99.5)
    combined = np.clip((combined - low) / (high - low + 1e-8), 0.0, 1.0)

    # pre-colorization blur => soft transitions
    raw_uint8 = np.uint8(255 * combined)
    if pre_blur is not None:
        raw_uint8 = cv2.GaussianBlur(raw_uint8, pre_blur, 0)

    raw_float = raw_uint8.astype(np.float32) / 255.0
    raw_float = (raw_float - raw_float.min()) / (raw_float.max() - raw_float.min() + 1e-8)

    # colorize using TURBO (smooth perceptual)
    heatmap_u8 = np.uint8(255 * raw_float)
    colored = cv2.applyColorMap(heatmap_u8, colormap)

    # final color smoothing + small contrast
    if post_blur is not None:
        colored = cv2.GaussianBlur(colored, post_blur, 0)
    colored = cv2.convertScaleAbs(colored, alpha=1.05, beta=8)

    # --- prepare original xray (preserve details)
    img = input_tensor.detach().cpu().numpy()[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(255 * img)
    if img.shape[0] == 3:
        xray = np.transpose(img, (1,2,0))  # H W C
        xray = cv2.cvtColor(xray, cv2.COLOR_RGB2BGR)
    else:
        xray = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)

    # apply mild CLAHE on Y channel to improve local contrast without blowing out bones
    lab = cv2.cvtColor(xray, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    xray_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    xray_clahe = cv2.convertScaleAbs(xray_clahe, alpha=1.0, beta=0)

    # --- PER-PIXEL BLEND: alpha_map = raw_float * max_alpha (so hotspots more opaque)
    alpha = (raw_float[..., None] * float(max_alpha)).astype(np.float32)  # H W 1
    # enforce a small floor so heatmap doesn't disappear entirely on weak activations
    alpha = np.clip(alpha, 0.05, max_alpha)

    # convert to float and blend
    xray_f = xray_clahe.astype(np.float32)
    colored_f = colored.astype(np.float32)
    overlay_f = (xray_f * (1.0 - alpha) + colored_f * alpha)
    overlay = np.uint8(np.clip(overlay_f, 0, 255))

    # --- bounding box detection using top percentile of raw_float
    thresh_val = np.percentile(raw_float, percent_for_bbox)
    mask = (raw_float >= thresh_val).astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_overlay = overlay.copy()
    bbox_coords = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x,y,w_box,h_box = cv2.boundingRect(largest)
        # sanity: filter full-image or tiny boxes
        if h_box > 12 and w_box > 12 and not (h_box > overlay.shape[0]*0.9 and w_box > overlay.shape[1]*0.9):
            cv2.rectangle(bbox_overlay, (x,y), (x+w_box, y+h_box), (0,0,255), 2)
            bbox_coords = {"x":int(x), "y":int(y), "w":int(w_box), "h":int(h_box)}

    # --- save outputs
    heatmap_path = os.path.join(save_dir, f"{base_name}_heatmap.png")
    overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    bbox_path = os.path.join(save_dir, f"{base_name}_bbox.png")
    cv2.imwrite(heatmap_path, colored)
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(bbox_path, bbox_overlay)

    # composite for visualization
    spacer = 255 * np.ones((xray.shape[0], 10, 3), dtype=np.uint8)
    triple = np.concatenate([xray, spacer, colored, spacer, bbox_overlay], axis=1)
    triple_path = os.path.join(save_dir, f"{base_name}_triple.png")
    cv2.imwrite(triple_path, triple)

    print(f"[GradCAM Saved] -> {save_dir}")
    return {
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "bbox_path": bbox_path,
        "triple_path": triple_path,
        "bbox": bbox_coords,
    }

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

