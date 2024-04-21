import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F
from functools import partial
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage_wave import MVXTwowaveStageDetector
import pdb
import numpy as np
from torchvision.transforms.functional import resize




@DETECTORS.register_module()
class TransFusionwaveDetector(MVXTwowaveStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(TransFusionwaveDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(TransFusionwaveDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = True # vitfpn require grad

    def extract_img_feat(self, img, img_metas,depth_img):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.with_filter_fastsam:
                _,img_feats = self.img_backbone(img.float())
            else:
                img_feats = self.img_backbone(img.float())
        else:
            return None
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
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        #pdb.set_trace()
        img_size = img.size()
        #pdb.set_trace()
        depth_img = self.GenerateDepthfeatures(points,img_size,img_metas)
        img_feats = self.extract_img_feat(img, img_metas, depth_img)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)
    
    def translate(self,points,trans_vector): # liulin code
        points += trans_vector
        return points
    
    def scale(self, points,scale_factor):
        points = points * scale_factor
        return points
    
    def rotation(self,rotation, axis=None):
        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor([[rot_cos, 0, -rot_sin],
                                                 [0, 1, 0],
                                                 [rot_sin, 0, rot_cos]])
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor([[rot_cos, -rot_sin, 0],
                                                 [rot_sin, rot_cos, 0],
                                                 [0, 0, 1]])
            elif axis == 0:
                rot_mat_T = rotation.new_tensor([[0, rot_cos, -rot_sin],
                                                 [0, rot_sin, rot_cos],
                                                 [1, 0, 0]])
            else:
                raise ValueError('axis should in range')
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        points = points @ rot_mat_T
        
        return points
    
    def flip(self,points,detection):
        if direction == 'horizontal':
            points[:, 0] = -points[:, 0]
        elif direction == 'vertical':
            points[:, 2] = -points[:, 2]
    
    def apply_reverse_3d_transformation(self,pcd, img_meta):
        '''
        points [N,3]
        '''
        # flip reverse: flip
        # trans rot -> scale -> trans reverse: trans -> scale -> rot
        # img pad -> points_2d - pad
        device = pcd.device
        dtype = pcd.dtype
        #pdb.set_trace()
        #pdb.set_trace()
        pcd_rotate_mat = (torch.tensor(img_meta['pcd_rotation'], dtype=dtype, device=device)
            if 'pcd_rotation' in img_meta else torch.eye(
               3, dtype=dtype, device=device))

        pcd_scale_factor = (
            img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

        pcd_trans_factor = (
            torch.tensor(img_meta['pcd_trans'], dtype=dtype, device=device)
            if 'pcd_trans' in img_meta else torch.zeros(
                (3), dtype=dtype, device=device))

        pcd_horizontal_flip = img_meta[
            'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
            img_meta else False

        pcd_vertical_flip = img_meta[
            'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
            img_meta else False
        
        # flip操作在trans操作前面
        if pcd_horizontal_flip:
            pcd = self.flip(pcd,detection='pcd_horizontal_flip')
            
        if pcd_vertical_flip:
            pcd = self.flip(pcd,detection='pcd_vertical_flip')
        
        if 'pcd_trans' in img_meta:
            pcd = self.translate(pcd,-pcd_trans_factor)
        if 'pcd_scale_factor' in img_meta:
            pcd = self.scale(pcd, 1.0/pcd_scale_factor)
        if 'pcd_rotation' in img_meta:
            pcd = self.rotation(pcd, pcd_rotate_mat.inverse())
        
        return pcd
            

        
    
    
    @force_fp32()
    def GenerateDepths(self, depth_features, depth_mean=14.41, depth_var=156.89, img_size=(1024,1024)):
        depth_features[0] = (depth_features[0] - depth_mean)/np.sqrt(depth_var)
        depth_features[0] =  depth_features[0]*depth_features[1]
        depth_features = resize(depth_features, img_size)
        return depth_features
        
    @force_fp32()    
    def GenerateDepthfeatures(self, depth_points:list, img_size, img_metas):
        B_size, N, C, H, W = img_size
        # img_metas[sample_idx]['lidar2img'], depth_points本身就是列表
        #calibs = batch_dict['calib']
        #inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
            
        #depth_feature_list = []
        #pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
        depth_feature_total_list = []
        for b in range(B_size):
            '''
            img_crop_offset =  img_metas[b]['img_crop_offset'][:2] if 'img_crop_offset' in img_metas[b].keys() else 0
            img_flip = img_metas[b]['flip'] if 'flip' in img_metas[b].keys() else False
            img_shape = img_metas[b]['img_shape'][:2]
            img_pad_shape = img_metas[b]['input_shape'][:2]
            '''
            img_scale_factor = img_metas[b]['scale_factor'][:2] if 'scale_factor' in img_metas[b].keys() else [1.0, 1.0]
            points_batch = depth_points[b]
            calibs = img_metas[b]['lidar2img']
            #calibs = torch.from_numpy(calibs)
            points_batch = points_batch[:,0:3]
            points_batch = self.apply_reverse_3d_transformation(points_batch,img_meta=img_metas) # 点云逆变换 
            depth_feature_list = []
            #pdb.set_trace()
            for n in range(N):
                points_4d = torch.cat([points_batch, points_batch.new_ones(size=(points_batch.shape[0], 1))], dim=-1)
                calib_n = torch.Tensor(calibs[n]).cuda().float()
                pts_2d = points_4d @ calib_n.t()#.float()
                pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]
                points_2d = pts_2d[:,0:2]
                points_2d = points_2d.cpu()*img_scale_factor # 图片上有scale
                depth = pts_2d[:,2]
                filter_idx = (0<=points_2d[:, 1]) * (points_2d[:, 1] < H) * (0<=points_2d[:, 0]) * (points_2d[:, 0] < W)

                points_2d = points_2d[filter_idx]
                points_2d = torch.tensor(points_2d).long()
                depth = depth[filter_idx].cpu()

                sort_id = np.argsort(-depth)
                points_2d = points_2d[sort_id]
                depth = depth[sort_id]

                depth_feature = torch.zeros((2, H, W))

                cx = points_2d[:, 0]# .long()
                cy = points_2d[:, 1]# .long()

                depth_feature[0, cy, cx] = depth
                depth_feature[1, cy, cx] = 1 # liulin code
                    
                depth_feature = self.GenerateDepths(depth_features=depth_feature,img_size=(H,W))
                depth_feature_list.append(depth_feature)    
            depth_feature_list = torch.stack(depth_feature_list,dim=0) # 1*6的特征图
            depth_feature_total_list.append(depth_feature_list)
        depth_feature_total_list = torch.stack(depth_feature_total_list,dim=0)

        return depth_feature_total_list # [B,N,C,H,W] # liulin code end
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)        
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
