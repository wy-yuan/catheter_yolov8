from pathlib import Path
import cv2
import numpy as np
from skimage.draw import polygon   # used inside poly_to_mask
import matplotlib.pyplot as plt

# --- 1. poly_to_mask (same as before) ---------------------------
def poly_to_mask(poly_flat, img_h, img_w):
    if len(poly_flat) % 2:                         # must be pairs
        raise ValueError("Odd number of polygon values.")

    pts = np.asarray(poly_flat, dtype=np.float32).reshape(-1, 2)
    xs = np.clip(pts[:, 0] * img_w, 0, img_w - 1)
    ys = np.clip(pts[:, 1] * img_h, 0, img_h - 1)

    rr, cc = polygon(ys, xs, (img_h, img_w))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[rr, cc] = 255
    return mask

# --- 2. read polygon from YOLO txt ------------------------------
def mask_from_yolo_label(txt_path, img_h, img_w):
    """
    Parameters
    ----------
    txt_path : str or Path
        Path to YOLOv8-seg label file (one line: cls cx cy w h x1 y1 x2 y2 …).
    img_h, img_w : int
        Target mask dimensions.

    Returns
    -------
    mask : np.ndarray, uint8
    """
    with open(txt_path, "r") as f:
        parts = f.readline().strip().split()

    if len(parts) <= 5:             # no polygon points
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # tokens: 0=class, 1-4=bbox, 5..=polygon
    poly_flat = list(map(float, parts[5:]))
    return poly_to_mask(poly_flat, img_h, img_w)

# ----------------------------------------------------------------
# Example: reconstruct mask for one validation image
# ----------------------------------------------------------------
if __name__ == "__main__":
    # paths
    # img_file = Path("./dataset_Phantom/images/val/val_00011.png")
    # label_file = Path("./dataset_Phantom/labels/val/val_00011.txt")
    img_file = Path("./dataset_Phantom/images/train/train_00001.png")
    label_file = Path("./dataset_Phantom/labels/train/train_00001.txt")

    # load image just to get H×W
    img = cv2.imread(str(img_file))
    h, w = img.shape[:2]

    mask = mask_from_yolo_label(label_file, h, w)
    # cv2.imwrite("val_00002_recon.png", mask)

    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="gray")
    plt.title("Reconstructed mask from polygon")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

