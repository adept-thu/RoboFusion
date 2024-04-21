# install
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install spconv-cu111

cd ./focalsconv-mm/OpenPCDet
pip install -r requirements.txt
pip install SharedArray==3.2.2
python setup.py develop

cd ./pcdet/models/backbones_3d/SegmentAnything
python setup.py develop

cd ./pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM
python setup.py develop

cd ./pcdet/models/backbones_3d/focal_sparse_conv/ultralytics
python setup.py develop
```

# log
## 改动的地方

### 添加了SAM官方库，MobileSAM，FastSAM官方库：

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/SegmentAnything

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/ultralytics


### 添加小波模块

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/wavevit.py

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/torch_wavelet.py

Class Block：小波transformer模块
### 添加SparseDepth模块
需要注意：该模块与小波模块一起使用，该模块生成点云深度图，经过卷积后和图像特征拼接在一起，经过reduce_block后输入到小波模块中进行处理。

1. **添加该模块数据加载需要修改的位置：**

/focalsconv-mm/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py 542-545行
```python
if "sparsedepth" in get_item_list:
    usedepth = True
else:
    usedepth = False
data_dict = self.prepare_data(data_dict=input_dict, usedepth=usedepth) #注意prepare_data函数
相比之前添加了一个usedepth参数
```
/focalsconv-mm/OpenPCDet/pcdet/datasets/dataset.py 164-168行
```python
if self.training and usedepth:
    data_dict['depth_points'] =  points_after_aug
elif not(self.training) and usedepth:
    data_dict['depth_points'] =  points_before_aug # val,test过程没有数据增强，所以后面也
不会进行点云的逆变换。
```

2. **添加该模块网络结构中需要修改的地方**

新添加两个函数
focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/focal_sparse_conv.py
:::tips
def GenerateDepths(depth_features, depth_mean=14.41, depth_var=156.89, img_size=1024):
      ... 22-27行
def GenerateDepthfeatures(depth_points, batch_dict):
      ... 29-95行
:::
def GenerateDepths(depth_features, depth_mean=14.41, depth_var=156.89, img_size=1024):
         生成sparsedepth特征的代码（437-441行）
```python
if self.use_wave:
    depth_points = batch_dict['depth_points']
    depth_features = GenerateDepthfeatures(depth_points,batch_dict)
    depth_features = depth_features.to(device='cuda')
    depth_features = self.depthfilter(depth_features)
```

### 添加ViTDet上采样以及FPN模块

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/SimpleFeaturePyramid


**./SimpleFeaturePyramid/vit_feature_pyramid.py**

classSimpleFeaturePyramid：根据scale_factors的数量来匹配下采样倍率和fpn层数
### 添加fusion模块
/focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/utils.py
292-309行
```python
class fusion(nn.Module):

    def __init__(self, dim, cluster_feature=2304):
        super().__init__()
        self.FusionModel = Frequency_attention(dim=dim, num_heads=2, mlp_ratio=4)
        self.cluster1 = nn.AdaptiveAvgPool1d(cluster_feature)
 
    def forward(self, x):
        x_copy = x
        #size = x.shape[1]
        #print(x.shape)
        x = self.cluster1(x.transpose(1,2).contiguous())
        #print("x1",x.shape)
        x = self.FusionModel(x.transpose(1,2))
        #print("x2:",x.shape)
        x = nn.functional.interpolate(x.transpose(1,2), size=x_copy.shape[1])
        #print("x3:",x.shape)
        return x.transpose(1,2) + x_copy
```
### 改动3D Backbone

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/spconv_backbone_focal.py


136:195 为新的模型初始化逻辑

use_img:True 使用多模态
	use_seg: 初始化DeepLabV3 (baseline的做法)
	use_sam: Sam离线版本，需要使用fmap的离线文件
	use_raw_img: 仅使用原始图像 3x375x1242-->16x372x1240
	use_sam_online: Sam在线v1 只使用256x64x64的图像融合
	use_sam_onlinev2: Sam在线v2  添加了上采样层(使用的是vitdet的代码)
        use_mobile_samv1: 只使用64x64x256的尺度
        use_mobile_samv2: 使用上采样操作
        use_sam_fpn: 使用AD-FPN
        use_fastsam：使用fastsam在线
        zgx_code：False为使用fusion模块
        use_wave：使用小波模块
227-363 __init__里初始化相应的模块
435:557 forward里对应使用相应的模块

### 添加了部分注释，以及可视化保存文件

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/focal_sparse_conv.py


添加的代码均以注释掉，其余与baseline一致。

## Key Module

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/focal_sparse_conv.py


Focalsconv算子

207和222行 调用多模态融合

```
    # 两个模块融合的接口
    def construct_multimodal_features(self, x, x_rgb, batch_dict, fuse_sum=False):
        """ 
            Construct the multimodal features with both lidar sparse features and image features.
            Args:
                x: [N, C] lidar sparse features
                x_rgb: [b, c, h, w] image features
                batch_dict: input and output information during forward
                fuse_sum: bool, manner for fusion, True - sum, False - concat

            Return:
                image_with_voxelfeatures: [N, C] fused multimodal features
        """
```

```
66: voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3] # 体素索引转为点云坐标
```

```
96: voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy()) # 投影到图像
```

```
103: voxels_2d_int = voxels_2d_int[filter_idx] # 过滤掉图像外的点
```

```
109:if fuse_sum: # 融合方式
110:	imge_with_voxelfeature = image_features_batch + voxel_features_sparse
111:else:
112:	image_with_voxelfeature = torch.cat([image_features_batch, voxel_features_sparse], dim=1)
```

# Training

运行samv3:

```
cd focalsconv-mm/OpenPCDet/tools
python train.py --cfg_file cfgs/kitti_models/sam_onlinev3.yaml [--extra_tag debug] # extra_tag为实验标签(可选参数)
```

# 配置文件yaml

对应的参数会传入对应的model_cfg

```
    BACKBONE_3D: 
        NAME: VoxelBackBone8xFocal
        USE_IMG: True
        USE_SAM_ONLINEV2: True
        USE_SAM_FPN: True
        IMG_PRETRAIN: "../checkpoints/sam_image_encoder_b.pth"
```

在对应的NAME类中使用model_cfg.get()接收

```
        use_raw_img = model_cfg.get('USE_RAW_IMG', False)
        use_sam_online = model_cfg.get('USE_SAM_ONLINE', False)
        use_sam_onlinev2 = model_cfg.get('USE_SAM_ONLINEV2', False)
        use_sam_fpn = model_cfg.get('USE_SAM_FPN', False)
```

get()中第二个参数是默认值
