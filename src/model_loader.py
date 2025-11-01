# src/model_loader.py
import torch
import torch.nn as nn
from torchvision import models
from timm import create_model
import os

class EnsembleModel(nn.Module):
    def __init__(self, model_paths, weights, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        assert len(model_paths) == len(weights)
        self.device = device
        self.models = nn.ModuleList([self._load_model(p).to(device) for p in model_paths])
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.weights /= self.weights.sum()

    def _load_model(self, path: str):
  
    
    
        fname = os.path.basename(path).lower()
    
        # -------------------------------------------------
        # Detect DenseNet
        # -------------------------------------------------
        if "dense" in fname or "densenet" in fname:
            print(f"[INFO] Loading DenseNet from {path}")
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, 14)
    
        # -------------------------------------------------
        # EfficientNet-B3 / B4
        # -------------------------------------------------
        elif "b3" in fname:
            from timm import create_model
            model = create_model("efficientnet_b3", pretrained=False, num_classes=14)
        elif "b4" in fname:
            from timm import create_model
            model = create_model("efficientnet_b4", pretrained=False, num_classes=14)
    
        # -------------------------------------------------
        # Swin Transformer
        # -------------------------------------------------
        elif "swin" in fname:
            from timm import create_model
            model = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=14)
    
        else:
            raise ValueError(f"Unknown model type for file: {path}")
    
        # -------------------------------------------------
        # Load state dict safely
        # -------------------------------------------------
        checkpoint = torch.load(path, map_location="cpu")
    
        # Handle if checkpoint contains "model_state_dict"
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    
        # Strip 'module.' prefix if present
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
        # Load with strict=False to tolerate small mismatches
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys in {fname}: {len(missing)} keys ignored.")
        if unexpected:
            print(f"[WARN] Unexpected keys in {fname}: {len(unexpected)} keys ignored.")
    
        print(f"[OK] {fname} model loaded successfully ✅")
        return model


    def forward(self, x):
        with torch.no_grad():
            outputs = [w * m(x) for m, w in zip(self.models, self.weights)]
            combined = torch.stack(outputs).sum(dim=0)
            return torch.sigmoid(combined)

def load_ensemble(device="cuda" if torch.cuda.is_available() else "cpu"):
    model_paths = [
        "models/efficientnet_b3.pth",
        "models/efficientnet_b4.pth",
        "models/densenet121.pth",
        "models/swin_transformer.pth",
    ]
    weights = [0.25, 0.25, 0.2, 0.3]
    model = EnsembleModel(model_paths, weights, device=device)
    model.eval()
    return model
