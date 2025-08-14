from ultralytics.nn.tasks import SegmentationModel
import torch.nn.functional as F
from ultralytics import YOLO
import torch

def dice_loss(pred, tgt, eps=1e-6):
    """
    Soft Dice loss for a batch of masks.
    pred, tgt: float tensors of shape (B, 1, H, W) in [0,1]
    """
    inter = (pred * tgt).sum((2, 3))
    union = pred.sum((2, 3)) + tgt.sum((2, 3))
    dice = 1 - (2 * inter + eps) / (union + eps)
    return dice.mean()


def focal_bce(pred, tgt, gamma=2.0, alpha=0.25):
    """
    Focal Binary-Cross-Entropy loss.
    pred: logits (unnormalised)  tgt: {0,1} float
    """
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
    p_t = p * tgt + (1 - p) * (1 - tgt)
    focal = alpha * (1 - p_t) ** gamma * ce
    return focal.mean()


# ---------- custom network ----------------------------------------
class SegWithFocal(SegmentationModel):
    """
    YOLO segmentation model with extra focal-BCE + Dice mask loss.
    """

    focal_gamma = 2.0
    focal_alpha = 0.25
    w_focal    = 10.0
    w_dice     = 10.0

    def compute_loss(self, preds, batch):
        # call parent to get the standard losses and logging dict
        loss, log = super().compute_loss(preds, batch)

        # 'up_masks' is a tuple produced in the parent implementation:
        # (proto, pred_masks, gt_masks)
        if "up_masks" not in log:          # safety: non-segment task
            return loss, log

        _, pred_masks, gt_masks = log["up_masks"]   # (nT, 1, H, W) float 0-1

        # BCE_mask already in 'loss' via parent; we ADD focal + dice
        focal = focal_bce(pred_masks, gt_masks,
                          gamma=self.focal_gamma,
                          alpha=self.focal_alpha)
        dice  = dice_loss(pred_masks.sigmoid(), gt_masks)

        loss += self.w_focal * focal + self.w_dice * dice

        # for TensorBoard / CSV logging
        log["loss/focal_mask"] = focal.detach()
        log["loss/dice_mask"]  = dice.detach()

        return loss, log


if __name__ == "__main__":
    # train_with_focal.py ------------------------------------------------
    base = YOLO("yolo11n-seg.pt")             # loads weights + default seg head
    net = SegWithFocal()
    net.load_state_dict(base.model.state_dict(), strict=False)
    base.model = net

    # ❸  Train
    base.train(
        task="segment",
        data="catheter.yaml",
        epochs=100,
        imgsz=256,
        batch=16,
        device=0,
        project="cathPhantom",
        name="cath_y11n_focal10dice10"
    )
