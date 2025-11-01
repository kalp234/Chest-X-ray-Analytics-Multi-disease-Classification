# src/explainability_metrics.py
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

def insertion_deletion_auc(prob_curve):
    x = np.linspace(0, 1, len(prob_curve))
    auc = np.trapz(prob_curve, x)
    return auc

def compute_ssim(map1, map2):
    map1, map2 = map1.astype(np.float32), map2.astype(np.float32)
    return ssim(map1, map2, data_range=1.0)

def compute_emd(map1, map2):
    map1, map2 = map1.flatten(), map2.flatten()
    return wasserstein_distance(map1, map2)
