

bash tools/dist_train.sh   configs/transfusion_nusc_voxel_L.py  4
bash tools/dist_test.sh  configs/transfusion_nusc_voxel_L.py  work_dirs/transfusion_nusc_voxel_L/epoch_19.pth  4 --eval bbox


python tools/train.py configs/transfusion_nusc_voxel_L.py
python tools/test.py  configs/transfusion_nusc_voxel_L.py  work_dirs/transfusion_nusc_voxel_L/epoch_19.pth --eval bbox







