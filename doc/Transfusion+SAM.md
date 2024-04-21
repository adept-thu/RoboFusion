# install
:::tips
python == 3.8.16
mmdet3d == 0.11.0
:::
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

# 由于代码里面已经自带mmdet3d代码所以不需要额外克隆
# 直接编译即可
pip install -v -e .
# 注意需要额外克隆三个分割模型的代码到Transfusion代码仓库下
git clone https://github.com/ChaoningZhang/MobileSAM.git
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
git clone https://github.com/facebookresearch/segment-anything.git
```
# 代码修改的位置
## 添加分割相关backbone文件到mmdet3d的models/backbone中，分别为fast_sam.py mobilesam_encoder.py sam_encoder.py(以sam_encoder为例)
```
import torch
import torch.nn as nn
from ..builder import BACKBONES
from segment_anything import image_encoder_registry
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger



@BACKBONES.register_module()
class SAMEncoder(nn.Module):
    def __init__(self, encoder_checkpoint, model_type='vit_b', im_size=1024, is_optimize=False):
        super().__init__()
        self.im_size = im_size
        self.is_optimize = is_optimize
        self.image_encoder = image_encoder_registry[model_type](checkpoint=encoder_checkpoint)
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            pass

    def preprocess(self, x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size = 1024) -> torch.Tensor:
        """
        For sam_encoder
        Normalize pixel values and resize to a square input.
        """
        
        assert x.shape[1] == 3, f"The batch images should be 3 for RGB, but get {x.shape[1]}."
        x = x * 255
        x = resize(x, (img_size, img_size))
        
        
        # Normalize colors
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        pixel_mean = torch.as_tensor(pixel_mean, device="cuda")
        pixel_std = torch.as_tensor(pixel_std, device="cuda")
        x = (x - pixel_mean) / pixel_std
        
        # # Pad
        # h, w = x.shape[-2:]
        # padh = img_size - h
        # padw = img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def forward(self, x_rgb):
        # x_rgb = self.preprocess(x_rgb)
        x_rgb = self.image_encoder(x_rgb)
        if not self.is_optimize:
            x_rgb.detach_()
        return x_rgb
```
修改init.py文件，添加新类如：SAMEncoder
```python
__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'PointNet2SASSG', 'PointNet2SAMSG', 'MultiBackbone', 'DLASeg',
    'SAMEncoder','WaveBlock','BasicBlock2D','Depthfilter','MobileSamImageEncoder','fastsam','filterfastsam'
]
```
## 添加小波模块以及Sparse_map模块
添加小波模块：在backbone文件夹下，新增两个py文件：分别为wavevit.py 以及wavelet.py，其所使用的类与Focals-conv-mm代码文件中使用的相同
添加Sparse_map模块：Sparse_map相关代码添加在mmdet3d/detectors/transfusion_wave.py下，具体修改位置为64~87：其中使用的新类Depthfilter定义在wavevit.py中：
```python
if self.with_img_wave_fusion:
            BN, C, H, W = img.size()
            depth_img = depth_img.view(BN, 2, H, W)
            depth_img = depth_img.cuda()
            depth_feats = self.depthfilter(depth_img)
            #print(img_feats)
            if self.with_filter_fastsam:
               #print(img_feats[0])
               img_feats = img_feats[0].cuda()
               img_feats = self.filterfastsam(img_feats)
            img_feats = torch.cat([img_feats,depth_feats],dim=1)
            #print(img_feats.shape)
            #pdb.set_trace()
            img_feats = self.reduce_block(img_feats)
            bn,c,h,w = img_feats.shape
            #print(img_feats.shape)
            #pdb.set_trace()
            #print(img_feats.shape)
            img_feats = img_feats.view(bn,c,h*w).permute(0,2,1).contiguous() 
            img_feats = self.img_wave_fusionlayer(img_feats,h,w)
            img_feats = img_feats.view(bn,h,w,c).permute(0,3,1,2).contiguous()
```
其中涉及点云逆变换相关代码以及depthmap生成相关代码均放置在transfusion_wave.py文件下：具体函数为apply_reverse_3d_transformation以及GenerateDepthfeatures函数
## 修改模型config文件，以和mobilesam融合的transfusion为例：
```python
model = dict(
    type='TransFusionwaveDetector',
    freeze_img=True,
    img_backbone=dict(
        type='MobileSamImageEncoder',
        encoder_checkpoint=None,
        is_optimize=True,
        model_type='vit_t',
        im_size=1024),
    depthfilter=dict(type='Depthfilter', indim=2, outdim=16),
    reduce_block=dict(
        type='BasicBlock2D',
        in_channels=272,
        out_channels=16,
        kwargs=dict(kernel_size=1, stride=1, bias=False)),
    img_wave_fusionlayer=dict(
        type='WaveBlock', dim=16, num_heads=2, mlp_ratio=4),
    img_neck=dict(
        type='SimpleFeaturePyramid',
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        fuse_type='avg'),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=(120000, 160000),
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True,
        num_views=6,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=8,
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            nms_type=None)))
```

需要注意添加的模块：TransFusionwaveDetector（重写的transfusion类），MobileSamImageEncoder（mobilesam的骨干网络），Depthfilter（对深度图进行卷积的网络），BasicBlock2D（调整（图像特征concat深度图特征)通道的模块），WaveBlock（小波变换模块）。

需要注意相关config文件在/Transfusion_SAM/work_dirs文件夹下。
