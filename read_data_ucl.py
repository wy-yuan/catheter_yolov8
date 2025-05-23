import h5py, cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
IMG_H5   = "../dataset_from_UCL/T3-T6.hdf5"
LAB_H5   = "../dataset_from_UCL/T3-T6_label.hdf5"
IMG_PATH = "test"            # dataset name inside IMG_H5
LAB_PATH = "label"           # dataset name inside LAB_H5
# -------------------------------------------------------------------

with (
    h5py.File(IMG_H5, "r")  as f_img,
    h5py.File(LAB_H5, "r")  as f_lab
):
    data = f_img[IMG_PATH][...].astype(np.float32)   # shape: (N, H, W, 3)
    lab  = f_lab[LAB_PATH][...].astype(np.uint8)     # shape: (N, H, W, 2)

# quick sanity-check printout
print(f"images: {data.shape}  labels: {lab.shape}")

# -------------------------------------------------------------------
# VISUALISE THE FIRST SLICE
# -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(data[0, ..., 0], cmap="gray")
ax[0].set_title("Fluoro image")
ax[1].imshow(lab[0, ..., 0], cmap="gray")
cv2.imwrite(f"test_mask.png", lab[0, ..., 0])
print(np.max(data[0, ..., 0]), np.min(lab[0, ..., 0]))
ax[1].set_title("Segmentation mask")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()
