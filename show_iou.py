import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

img_id = "00011"
# IMG = "./dataset_T3-T6/images/train/train_{}.png".format(img_id)
# MASK = "./dataset_T3-T6/masks/train/train_{}.png".format(img_id)

# IMG = "./dataset_T3-T6/images/val/val_{}.png".format(img_id)
# MASK = "./dataset_T3-T6/masks/val/val_{}.png".format(img_id)

# IMG = "./dataset_Phantom/images/train/train_{}.png".format(img_id)
# MASK = "./dataset_Phantom/masks/train/train_{}.png".format(img_id)
IMG = "./dataset_Phantom/images/val/val_{}.png".format(img_id)
MASK = "./dataset_Phantom/masks/val/val_{}.png".format(img_id)
MODEL = "./cathPhantom/cath_y8ns/weights/best.pt"
IMG_SIZE = 256                     # use the same size you trained with
OUT = os.path.join("./runs/segment/", "iou_{}.jpg".format(img_id))
# -------------------------------------------------------------------

# 1) load model & predict
model = YOLO(MODEL)
result = model(str(IMG), imgsz=IMG_SIZE, conf=0.5, verbose=False)[0]

# assume single catheter → take the first predicted mask
pred_mask = result.masks.data[0].cpu().numpy() > 0.5        # bool H×W

# 2) load ground-truth mask
gt_mask = (cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE) > 0)[:, :, 0] # bool H×W

# 3) compute IoU
tp = np.logical_and(pred_mask, gt_mask).sum()
fp = np.logical_and(pred_mask, ~gt_mask).sum()
fn = np.logical_and(~pred_mask, gt_mask).sum()
print(pred_mask.sum(), gt_mask.shape)
print("tp, fp, fn:", tp, fp, fn)
iou = tp / (tp + fp + fn + 1e-6)

# 4) overlay mask & IoU text
img = cv2.imread(str(IMG))
overlay = img.copy()
overlay[pred_mask] = (0, 255, 0)        # green mask
img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

cv2.putText(
    img, f"IoU: {iou:.2f}", (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA
)

cv2.imwrite(str(OUT), img)
print(f"!!!saved {OUT}\nIoU = {iou:.4f}")

plt.figure(figsize=(3, 3))
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
