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
