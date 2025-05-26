"""
Evaluate YOLOv8-seg model on a test set and print mAP50.

Author: you – 2025-05-24
"""

from ultralytics import YOLO

# ------------------------------------------------------------------
# 1.  Load the trained checkpoint
# ------------------------------------------------------------------
model_path = "best.pt"          # or "runs/segment/train/weights/best.pt"
model = YOLO(model_path)        # automatically chooses seg head

# ------------------------------------------------------------------
# 2.  Run built-in validation *on the test split*
# ------------------------------------------------------------------
metrics = model.val(
    data="catheter.yaml",       # must have test: key
    splits="test",              # tell Ultralytics to use test set
    iou=0.5,                    # single IoU threshold = 0.50
    plots=False, save_txt=False # no extra files
)

# ------------------------------------------------------------------
# 3.  Extract and print the mAP50 numbers
# ------------------------------------------------------------------
# For segmentation tasks Ultralytics returns both box & mask metrics
print("="*40)
print(f"Mask mAP50  : {metrics.seg.map50:.4f}")
print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
print(f"Box  mAP50   : {metrics.box.map50:.4f}")   # localisation only
print("="*40)
