# catheter_yolov8
Yolo model for catheter (and catheter-tip) detection and segmentation based on public fluoroscopy datasets

#### Read data from HDF5 files
---  read_data_ucl.py

#### Prepare dataset
python prepare_data2yolo.py --img_h5 ../dataset_from_UCL/T3-T6.hdf5 --lab_h5 ../dataset_from_UCL/T3-T6_label.hdf5 --img_key test --lab_key label --out_dir dataset --val_split 0.2

#### Model training
yolo task=segment mode=train model=yolov8n-seg.pt data=catheter.yaml epochs=100 imgsz=256 batch=16 optimizer=AdamW lr0=1e-3 project=cathT3T6 name=cath_y8n device=0

#### Validate
yolo task=segment mode=val model=runs/segment/cath_y8n/weights/best.pt data=catheter.yaml

#### Test
yolo task=segment mode=predict model=best.pt source="some/test/*.png" save=True