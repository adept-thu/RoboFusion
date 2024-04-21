# log

## 改动的地方

### 添加了SAM官方库：

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/SegmentAnything

### 添加ViTDet上采样以及FPN模块

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/SimpleFeaturePyramid

**./SimpleFeaturePyramid/vit_feature_pyramid.py**

classSimpleFeaturePyramid：根据scale_factors的数量来匹配下采样倍率和fpn层数

### 改动3D Backbone

> focalsconv-mm/OpenPCDet/pcdet/models/backbones_3d/spconv_backbone_focal.py

136:195 为新的模型初始化逻辑

```
use_img:True 使用多模态
	use_seg: 初始化DeepLabV3 (baseline的做法)
	use_sam: Sam离线版本，需要使用fmap的离线文件
	use_raw_img: 仅使用原始图像 3x375x1242-->16x372x1240
	use_sam_online: Sam在线v1 只使用256x64x64的图像融合
	use_sam_onlinev2: Sam在线v2和v3  添加了上采样层(使用的是vitdet的代码)
		v2: USE_SAM_FPN:False 不使用fpn 只使用256x256x256的尺度
		v3: USE_SAM_FPN:True 使用fpn 从低分辨率sum至高分辨率
```

283:330 forward里对应使用相应的模块

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
