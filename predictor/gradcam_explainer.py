import os, cv2, torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
from django.conf import settings

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, "predictor", "best_model.pth")
SAVE_DIR = os.path.join(settings.MEDIA_ROOT, "gradcam_outputs")

# Classes
CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

# Model
class CheXNet(nn.Module):
    def __init__(self, n_classes=14, pretrained=False):
        super().__init__()
        if pretrained:
            self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.backbone = densenet121(weights=None)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

def load_model():
    model = CheXNet(n_classes=len(CLASS_NAMES), pretrained=False)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    model.load_state_dict(clean, strict=False)
    model.to(DEVICE).eval()
    return model

# GradCAM
class GradCAM:
    def __init__(self, target_layer: nn.Module):
        self.activations, self.gradients = None, None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)
    def _fwd_hook(self, m, i, o): self.activations = o
    def _bwd_hook(self, m, gi, go): self.gradients = go[0]
    def generate(self):
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)[0,0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_bbox_from_cam(cam_mask, orig_img, label=None, confidence=None):
    """Return image with bounding box + disease label drawn."""
    thresh = (cam_mask > cam_mask.max() * 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img_with_box = orig_img.copy()
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0,255,0), 2)

        if label:
            text = f"{label} {confidence:.1f}%" if confidence is not None else label
            cv2.putText(
                img_with_box, text,
                (x, y-10 if y > 20 else y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA
            )

        return img_with_box, (x, y, w, h)

    return orig_img, None

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Main Prediction with Grad-CAM
def predict_image(img_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = load_model()
    target_layer = model.backbone.features.denseblock4
    cam = GradCAM(target_layer)

    pil = Image.open(img_path).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    # ✅ Return ALL 14 disease predictions
    predictions = [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))]

    # ✅ Pick the single top-1 disease for GradCAM + BBox
    top1 = int(np.argmax(probs))
    cls = CLASS_NAMES[top1]
    conf = float(probs[top1])

    orig = np.array(pil.resize((224,224)))

    # Grad-CAM for top1
    model.zero_grad(set_to_none=True)
    logits[0, top1].backward(retain_graph=True)
    cam_mask = cam.generate()
    cam_resz = cv2.resize(cam_mask, (orig.shape[1], orig.shape[0]))
    heatmap = cv2.applyColorMap((cam_resz*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

    # Save heatmap
    saved_path = os.path.abspath(os.path.join(SAVE_DIR, f"{cls}_heatmap.png"))
    cv2.imwrite(saved_path, overlay)

    # Save bounding box (for top1 only)
    bbox_img, bbox_coords = get_bbox_from_cam(cam_resz, orig, cls, conf*100)
    bbox_path = os.path.abspath(os.path.join(SAVE_DIR, f"{cls}_bbox.png"))
    cv2.imwrite(bbox_path, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

    return {
    "predictions": [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))],
    "saved_path": saved_path,
    "bbox_path": bbox_path,
    "bbox_coords": bbox_coords,
    "top1_label": cls   # ✅ explicitly return top-1 disease
}



