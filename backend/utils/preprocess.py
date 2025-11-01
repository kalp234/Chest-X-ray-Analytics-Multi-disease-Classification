# backend/utils/preprocess.py
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Robust preprocessing for ensemble chest X-ray inference.

    Handles:
      • RGB / grayscale images
      • Internet or dataset X-rays
      • Varying brightness or inverted polarity
    Returns:
      torch.Tensor of shape (1, 3, 224, 224)
    """
    img = Image.open(image_path)

    # --- Handle grayscale vs RGB ---
    if img.mode != "RGB":
        img = img.convert("L")                # grayscale
        img = ImageOps.equalize(img)          # contrast normalization
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = img.convert("RGB")              # expand to 3 channels
    else:
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(1.1)

    # --- Resize and normalize (ImageNet stats) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor
