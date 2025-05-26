"""
Convert the catheter corpus (images & masks in separate HDF5 files)
into YOLOv8-segmentation folder structure.

"""
import h5py, cv2, argparse, os, pathlib, random, numpy as np
from skimage import measure

def ensure_dirs(root: pathlib.Path):
    for split in ("train", "val"):
        for sub in ("images", "masks", "labels"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)

def mask_to_poly(mask, max_points=30):
    """Return ≤max_points polygon vertices normalised to [0,1]."""
    contours = measure.find_contours(mask, 0.5)
    if not contours:          # empty mask – shouldn’t happen
        return None
    # pick largest contour
    pts = max(contours, key=lambda c: len(c))
    # sample evenly along contour
    step = max(1, len(pts) // max_points)
    pts = pts[::step][:max_points]            # (N, 2)   y,x
    if len(pts) < 3:                          # degenerate
        return None
    h, w = mask.shape
    pts[:, 0] /= h                            # y-norm
    pts[:, 1] /= w                            # x-norm
    return pts[:, ::-1].flatten()             # x1,y1,x2,y2,…


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    ensure_dirs(out_dir)

    with h5py.File(args.img_h5, "r") as f_img, h5py.File(args.lab_h5, "r") as f_lab:
        imgs = f_img[args.img_key][...].astype(np.float32)      # N,H,W,3
        labs = f_lab[args.lab_key][...].astype(np.uint8)      # N,H,W,2

    assert imgs.shape[0] == labs.shape[0], "image / label count mismatch"
    nb = imgs.shape[0]
    idx = list(range(nb))
    random.seed(args.seed)
    random.shuffle(idx)

    val_cut = int(nb * args.val_split)
    split_map = {i: "val" if i < val_cut else "train" for i in range(nb)}

    # -------------------------- iterate ------------------------------
    for k, i in enumerate(idx):
        split = split_map[k]
        img = imgs[i]
        mask = labs[i, ..., 0]               # first channel

        # save raw image & mask for debugging (optional)
        fname = f"{split}_{i:05d}"
        cv2.imwrite(str(out_dir / "images" / split / f"{fname}.png"), img)
        cv2.imwrite(str(out_dir / "masks"  / split / f"{fname}.png"), mask*255)

        # -------------- label -----------------------------------
        poly = mask_to_poly(mask)
        if poly is None:
            continue                         # skip empty / bad mask

        ys, xs = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        h, w = mask.shape
        # bbox centre-x, centre-y, width, height  (normalised)
        cx = (x_min + x_max) / (2 * w)
        cy = (y_min + y_max) / (2 * h)
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h

        with open(out_dir / "labels" / split / f"{fname}.txt", "w") as txt:
            txt.write(
                "0 " + " ".join(f"{p:.6f}" for p in (cx, cy, bw, bh, *poly))
            )

    # ------------------- write dataset YAML --------------------------
    yaml_str = f"""\
path: {out_dir}
train: images/train
val: images/val
nc: 1
names: ["catheter"]
"""
    (out_dir / "catheter.yaml").write_text(yaml_str)
    print("!!! Conversion done. Dataset root:", out_dir.resolve())

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_h5",   default="T3-T6.hdf5")
    parser.add_argument("--lab_h5",   default="T3-T6_label.hdf5")
    parser.add_argument("--img_key",  default="test",
                        help="dataset name inside image HDF5")
    parser.add_argument("--lab_key",  default="label",
                        help="dataset name inside label HDF5")
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
