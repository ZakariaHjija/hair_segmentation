



import os
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_image(t):
    """Convertit un tensor PyTorch CxHxW en image numpy HxWx3."""
    t = t.detach().cpu().numpy()
    if t.ndim == 3 and t.shape[0] == 3:
        t = np.transpose(t, (1, 2, 0))   # C,H,W -> H,W,C
    elif t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def mask_to_rgb(mask):
    """Convertit un masque binaire en une image RGB (cheveux = rouge)."""
    mask = mask.squeeze()
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    mask_rgb[..., 0] = mask  # canal rouge
    return mask_rgb


def overlay_mask(image, mask_rgb, alpha=0.5):
    """Superpose le masque RGB sur l'image originale."""
    image = image.copy()
    if image.max() > 1:     # normalisation
        image = image / 255.0
    return (1 - alpha) * image + alpha * mask_rgb

def save_prediction(image, gt_mask, pred_mask, save_path):
    """
    Sauvegarde une image, son masque GT, le masque prédiction, et l'overlay.
    save_path est le chemin SANS extension.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convertir tensors → numpy
    img = tensor_to_image(image)
    gt = tensor_to_image(gt_mask)
    pred = tensor_to_image(pred_mask)

    # Seuiler la prédiction si elle est en probabilités
    if pred.max() > 1:
        pred = pred / pred.max()
    pred_bin = (pred > 0.5).astype(np.float32)

    # Masques RGB
    gt_rgb = mask_to_rgb(gt)
    pred_rgb = mask_to_rgb(pred_bin)

    # Overlay
    overlay_pred = overlay_mask(img, pred_rgb, alpha=0.5)

    # Sauvegardes
    plt.imsave(save_path + "_image.png", img)
    plt.imsave(save_path + "_mask.png", gt_rgb)
    plt.imsave(save_path + "_pred.png", pred_rgb)
    plt.imsave(save_path + "_overlay.png", overlay_pred)