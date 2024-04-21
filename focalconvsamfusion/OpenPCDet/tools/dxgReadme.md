# prepare

## install
python3.8
torch 1.10
torchvision
> OpenPCDet 0.6
pip install spconv-cu111 # 3090
pip install spconv-cu118 # 4090

pip install -r requirements.txt
pip istall -e .
pip install av2
pip install kornia

> 3D_Corruptions_AD
pip install imgaug
pip install open3d
pip install h5py
pip install distortion
>> snow
pip install imagecorruptions
>> rain
pip install PyMieScatt
>> fog
git clone https://gitclone.com/github.com/MartinHahner/LiDAR_fog_sim.git
cp -r integral_lookup_tables/ /sda/dxg/OpenPCDet/pcdet/utils/corruptions_utils/

## log

lidar snow 和 rain 比较耗时
tools/corruptions.sh 批处理多个噪声的脚本, 执行前要cfg和ckpt路径，liaronly-batchsize=32,fusion-batchsize=16;; 也可以设置部分噪声进行批处理
tools/corruptions_pp.py 生成csv方便copy结果

### 改动的地方

```
modified:
OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py
OpenPCDet/tools/test.py
OpenpCDet/pcdet/datasets/__init__.py
OpenPCDet/tools/eval_utils/eval_utils.py

add:
OpenPCDet/pcdet/utils/corruptions_utils
OpenPCDet/pcdet/utils/Camera_corruptions.py
OpenPCDet/pcdet/utils/LiDAR_corruptions.py
tools/corruptions.sh
tools/corruptions_pp.py
```

相应的文件夹下有该文件的备份，需要时请替换 (张国欣)
xx_backup.py 为原版pcdet
xx_cor.py 为噪声版
 

## visualize

```python
import open3d
from visual_utils import open3d_vis_utils as V
V.draw_scenes(points=data_dict['points'][:, 1:],
              ref_boxes=pred_dicts[0]['pred_boxes'],
              ref_scores=pred_dicts[0]['pred_scores'],
              ref_labels=pred_dicts[0]['pred_labels']
            )

import matplotlib.pyplot as plt
plt.imshow(image) # HWC
plt.show()
```


# PVR-CNN

## demo
python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --data_path ../data/kitti/training/velodyne/000008.bin
## test
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth

RTX3090 PV-RCN推理 Batch=16, GPU Memory=21G, 耗时5min45s 1.5it/s, total iter=236

### test-rain

python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth --corruptions rain_sim None --severity 5

RTX3090 PV-RCN推理 Batch=16, GPU Memory=21G, 耗时5min45s 1.5it/s, total iter=236

### test-Gaussian(L)

python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth --corruptions gaussian_noise None --severity 5

RTX3090 PV-RCN推理 Batch=16, GPU Memory=21G, 耗时5min45s 1.5it/s, total iter=236

### test-Uniform(L)

python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth --corruptions uniform_noise None --severity 5

### test-Impulse(L)

python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth --corruptions impulse_noise None --severity 5



# pointpillar

> 对于Lidar-only的方法，Camera增强的方式即使默认不读取，不执行，传None,或者其他都可以。

## test
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 16 --ckpt ../checkpoints/pointpillar_7728.pth

### test-snow

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 16 --ckpt ../checkpoints/pointpillar_7728.pth --corruptions snow None --severity 5

### test-rain

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 16 --ckpt ../checkpoints/pointpillar_7728.pth --corruptions rain None --severity 5

### test-Impulse(L)

python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 16 --ckpt ../checkpoints/pointpillar_7728.pth --corruptions impulse_noise None --severity 5


# Voxel-RCNN（FocalsConv）

## test
python test.py --cfg_file cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml --batch_size 16 --ckpt ../checkpoints/voxelrcnn_focal_multimodal_85.66.pth --corruptions None None --severity 0
## test-glare
python test.py --cfg_file cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml --batch_size 16 --ckpt ../checkpoints/voxelrcnn_focal_multimodal_85.66.pth --corruptions glare glare --severity 5

python test.py --cfg_file cfgs/kitti_models/sam_onlinev3.yaml --batch_size 16 --ckpt ../checkpoints/sam_onlinev3/v3_avg_fpn/ckpt/checkpoint_epoch_73.pth --corruptions glare glare --severity 1


# 批处理脚本
```
rm -rf output/*
cd tools

bash corruptions.sh
python corruptions_pp.py
```