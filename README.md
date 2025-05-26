# Catheter-YOLOv8  
Tiny, end-to-end pipeline for **catheter / catheter-tip detection and segmentation** in fluoroscopy.

---

---

## 1  Convert the UCL HDF5 → YOLO folders
```bash
python prepare_data2yolo.py \
    --img_h5  ../dataset_from_UCL/T3-T6.hdf5          \
    --lab_h5  ../dataset_from_UCL/T3-T6_label.hdf5    \
    --img_key test --lab_key label                    \
    --out_dir dataset --val_split 0.20

python prepare_data2yolo.py --img_h5 ../dataset_from_UCL/T3-T6.hdf5 --lab_h5 ../dataset_from_UCL/T3-T6_label.hdf5 --img_key test --lab_key label --out_dir dataset --val_split 0.2

```
Creates:
```
dataset/
 ├─ images/{train,val}/xxx.png
 ├─ masks/{train,val}/xxx.png      # optional, for debugging
 ├─ labels/{train,val}/xxx.txt     # YOLO polygons
 └─ catheter.yaml                  # data config for Ultralytics
```

---

## 3  Train YOLOv8-nano-seg
```bash
yolo task=segment mode=train \
     model=yolov8n-seg.pt \
     data=dataset/catheter.yaml \
     epochs=100 imgsz=256 batch=16 \
     optimizer=AdamW lr0=1e-3 \
     project=cathT3T6 name=cath_y8n \
     device=0
yolo task=segment mode=train model=yolov8n-seg.pt data=catheter.yaml epochs=100 imgsz=256 batch=16 optimizer=AdamW lr0=1e-3 project=cathT3T6 name=cath_y8n device=0

```
*Weights & logs →* `cathT3T6/segment/cath_y8n/`.

---

## 4  Validate
```bash
yolo task=segment mode=val \
     model=cathT3T6/segment/cath_y8n/weights/best.pt \
     data=dataset/catheter.yaml
yolo task=segment mode=val model=runs/segment/cath_y8n/weights/best.pt data=catheter.yaml
```
Reports **seg mAP50**, **seg mAP50-95**, precision, recall, etc.

---

## 5  Single-image inference
```bash
yolo task=segment mode=predict \
     model=cathT3T6/segment/cath_y8n/weights/best.pt \
     source="./dataset/images/val/val_00002.png" \
     save=True
yolo task=segment mode=predict model=./cathT3T6/cath_y8n_1000/weights/best.pt source="./dataset_T3-T6/images/val/val_00002.png" save=True
```
Annotated image saved to `runs/segment/predict/`.

---

### 🔑 Helper scripts
| File | Purpose |
|------|---------|
| `read_data_ucl.py`         | Quick HDF5 viewer for raw UCL data. |
| `prepare_data2yolo.py`     | Converts HDF5 → YOLO folders + polygon labels. |
| `utils_label_to_mask.py`   | Rebuilds binary masks from YOLO polygon `.txt`. |
