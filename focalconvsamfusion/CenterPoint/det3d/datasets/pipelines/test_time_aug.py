import torch
import numpy as np
# from mmdet3d.core.points import BasePoints


from ..registry import PIPELINES



def format_list_float_06(l) -> None:
    for index, value in enumerate(l):
        l[index] = float('%.6f' % value)
    return l

def load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    points = np.fromfile(pts_filename, dtype=np.float32)
    return points

def remove_close(points, radius=1.0):
    """Removes point too close within a certain radius from origin.

    Args:
        points (np.ndarray): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray: Points after removing.
    """
    if isinstance(points, np.ndarray):
        points_numpy = points
    elif isinstance(points, BasePoints):
        points_numpy = points.tensor.numpy()
    else:
        raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]

@PIPELINES.register_module
class CorruptionMethods(object):
    """Test-time augmentation with corruptions.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(self):


        # 能作为全局设定存在的，应是指定：
        # 1.用什么corruption. 2.扰动的程度
        from .corruptions_config import Corruptions_mode
        cor = Corruptions_mode()
        self.corruption_type_l, self.corruption_type_c, self.severity = cor.get_corruption()
        # Weather
        if self.corruption_type_c is not None and self.corruption_type_c == 'snow': # snow_sim
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageAddSnow
            self.snow_sim_c = ImageAddSnow(self.severity, seed=2022)
            
        if self.corruption_type_c is not None and self.corruption_type_c == 'rain': # rain_sim
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageAddRain
            self.rain_sim_c = ImageAddRain(self.severity, seed=2022)
        
        if self.corruption_type_c is not None and self.corruption_type_c == 'fog': # fog_sim
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageAddFog
            self.fog_sim_c = ImageAddFog(self.severity, seed=2022)
        
        if self.corruption_type_c is not None and 'sunlight' in self.corruption_type_c: 
            #! sun_sim 可以对点云和图像一起加噪声，sun_sim_mono 只对图像加噪声，还有 scene_glare_noise 可以只对 Lidar 加噪声
            np.random.seed(2022)
            from .Camera_corruptions import ImagePointAddSun, ImageAddSunMono
            # 点云和图像双重加噪
            self.sun_sim = ImagePointAddSun(self.severity)
            # mono的纯图像加噪
            self.sun_sim_c = ImageAddSunMono(self.severity)
        # Sensor
        if self.corruption_type_c is not None and self.corruption_type_c == 'gaussian': # gauss_sim
            from .Camera_corruptions import ImageAddGaussianNoise
            self.gaussian_sim_c = ImageAddGaussianNoise(self.severity, seed=2022)
        if self.corruption_type_c is not None and self.corruption_type_c == 'uniform': # uniform_sim
            from .Camera_corruptions import ImageAddUniformNoise
            self.uniform_sim_c = ImageAddUniformNoise(self.severity) # uniform_sim 不需要设置seed
        if self.corruption_type_c is not None and self.corruption_type_c == 'impulse': # impulse_sim
            from .Camera_corruptions import ImageAddImpulseNoise
            self.impulse_sim_c = ImageAddImpulseNoise(self.severity, seed=2022)
        # Motion
        if self.corruption_type_c is not None and self.corruption_type_c == 'motion_blur': # motion_sim
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageMotionBlurFrontBack, ImageMotionBlurLeftRight
            self.motion_blur_sim_c_leftright = ImageMotionBlurFrontBack(self.severity)
            self.motion_blur_sim_c_frontback = ImageMotionBlurLeftRight(self.severity)
        # Object
        if self.corruption_type_c is not None and self.corruption_type_c == 'shear': # bbox_shear
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageBBoxOperation, ImageBBoxOperationMono
            self.shear_sim_c = ImageBBoxOperation(self.severity)
            self.shear_sim_c_mono = ImageBBoxOperationMono(self.severity)

        if self.corruption_type_c is not None and self.corruption_type_c == 'scale': # bbox_scale
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageBBoxOperation, ImageBBoxOperationMono
            self.scale_sim_c = ImageBBoxOperation(self.severity)
            self.scale_sim_c_mono = ImageBBoxOperationMono(self.severity)
            
        if self.corruption_type_c is not None and self.corruption_type_c == 'rotation': # bbox_rotate
            # 注意这个只是加图像噪声，没有点云干扰
            from .Camera_corruptions import ImageBBoxOperation, ImageBBoxOperationMono
            self.rotate_sim_c = ImageBBoxOperation(self.severity)
            self.rotate_sim_c_mono = ImageBBoxOperationMono(self.severity)

    def __call__(self, results, info):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        #! dxg Clear
        if self.severity == 0:
            return results, info
        
        #! Weather
        # snow offline
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'snow': # snow_sim
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.snow_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.snow_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        ## Lidar 
        if self.corruption_type_l is not None and self.corruption_type_l =='snow': # snow_sim_lidar
            import numpy as np
            from .LiDAR_corruptions import snow_sim, snow_sim_nus
            pl = results['points'].tensor
            # aug_pl = pl[:,:3]
            points_aug = snow_sim_nus(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        # rain offline
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'rain': # rain_sim
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.rain_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.rain_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'rain':
            import numpy as np
            from .LiDAR_corruptions import rain_sim
            pl = results['points'].tensor
            # aug_pl = pl[:,:3]
            points_aug = rain_sim(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
            
        # fog offline
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'fog':
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.fog_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.fog_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'fog': # fog_sim_lidar
            from .LiDAR_corruptions import fog_sim
            pl = results['points'].tensor
            points_aug = fog_sim(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        # sunlight
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'sunlight_f': # sun_sim
            import numpy as np
            # Transfusion读取的图像是使用mmcv.imread()读取的是RGB通道的图像
            # 但是代码写的是读取BGR图像，存BGR图像
            # 这里改为读PIL.JpegImage  转为ndarray处理，再存PIL.JpegImage  
            img = results['img'] # [PIL.JpegImage * 6]
            if 'lidar2image' in results: # bevfusion
                lidar2img = results['lidar2image']
            points_tensor = results['points'].tensor
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                    # len(img)==6 nuscenes
                    # len(img)==5 waymo
                    # 太阳只需要加一个 -- FRONT
                    '''
                    nuscenes:
                    0    CAM_FRONT, //加这个
                    1    CAM_FRONT_RIGHT,
                    2    CAM_FRONT_LEFT,
                    3    CAM_BACK,
                    4    CAM_BACK_LEFT,
                    5    CAM_BACK_RIGHT
                    '''
                    img0_np = np.array(img[0]) # PIL.JpegImage -> ndarry
                    lidar2img0_np = lidar2img[0]
                    lidar2img0_tensor = torch.from_numpy(lidar2img0_np)
                    img0_np_aug, points_aug = self.sun_sim(
                        image=img0_np,
                        points=points_tensor,
                        lidar2img=lidar2img0_tensor,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    img0_PIL_aug = Image.fromarray(img0_np_aug)
                    img[0] = img0_PIL_aug
                    results['img'] = img
                    results['points'].tensor = points_aug
                
        # Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'sunlight_l': # scene_glare_noise
            from .LiDAR_corruptions import scene_glare_noise
            pl = results['points'].tensor
            # aug_pl = pl[:,:3]
            points_aug = scene_glare_noise(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        
        #! Sensor
        # density
        ## lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'density': # density_dec_global
            from .LiDAR_corruptions import density_dec_global
            pl = results['points'].tensor
            points_aug = density_dec_global(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # cutout
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'cutout': # cutout_local
            from .LiDAR_corruptions import cutout_local
            pl = results['points'].tensor
            points_aug = cutout_local(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # crosstalk
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'crosstalk': # lidar_crosstalk_noise
            from .LiDAR_corruptions import lidar_crosstalk_noise
            pl = results['points'].tensor
            aug_pl = pl[:,:3]
            points_aug = lidar_crosstalk_noise(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # FOV Lost
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'fov':
            import numpy as np
            from .LiDAR_corruptions import fov_filter
            pl = results['points'].tensor
            points_aug = fov_filter(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # gaussian_l
        ## Lidar √
        if self.corruption_type_l is not None and self.corruption_type_l == 'gaussian': # gaussian_noise
            from .LiDAR_corruptions import gaussian_noise
            pl = results['lidar']['combined'] # tensor [N, 5] # det3d results['lidar']['combined']
            pl = gaussian_noise(pl, self.severity)
            results['lidar']['combined'] = pl
        # uniform_l
        ## lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'uniform':
            from .LiDAR_corruptions import uniform_noise
            pl = results['points'].tensor
            points_aug = uniform_noise(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # impulse_l
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'impulse':
            from .LiDAR_corruptions import impulse_noise
            pl = results['points'].tensor
            points_aug = impulse_noise(pl.numpy(), self.severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # gaussian_c
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'gaussian':
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.gaussian_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img[:,:,[2,1,0]]
                image_aug_rgb = self.gaussian_sim_c(image=img_rgb_255_np_uint8)
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        # uniform_c
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'uniform':
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.uniform_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img[:,:,[2,1,0]]
                image_aug_rgb = self.uniform_sim_c(image=img_rgb_255_np_uint8)
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        # impulse_c
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'impulse':
            import numpy as np
            img = results['img'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.impulse_sim_c(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img[:,:,[2,1,0]]
                image_aug_rgb = self.impulse_sim_c(image=img_rgb_255_np_uint8)
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        
        #! Motion
        # compensation
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'compensation': # fulltrajectory_noise
            import numpy as np
            from pyquaternion import Quaternion
            from .LiDAR_corruptions import fulltrajectory_noise
            pl = results['points'].tensor
            aug_pl = pl[:,:3]
            # load pc_pose
            if len(results['sweeps']) != 0:
                # bevfusion是4x4矩阵，Transfusion是和数据集一样的四元数Quaternions
                fir_ego_pose = format_list_float_06(results['ego2global_translation'] + results['ego2global_rotation'])
                fir_sen_glo = format_list_float_06(results['lidar2ego_translation'] + results['lidar2ego_rotation'])

                sec_sweeps = results['sweeps'][0]
                sec_ego_pose = format_list_float_06(sec_sweeps['ego2global_translation']+ sec_sweeps['ego2global_rotation'])
                sec_sen_glo = format_list_float_06(sec_sweeps['sensor2ego_translation']+ sec_sweeps['sensor2ego_rotation'])

                pc_pose = np.array([fir_ego_pose,fir_sen_glo,sec_ego_pose,sec_sen_glo])
                # print(pc_pose.shape)
                points_aug = fulltrajectory_noise(aug_pl.numpy(), pc_pose, self.severity)
                pl[:,:3] = torch.from_numpy(points_aug)
                results['points'].tensor = pl
            else:
                results['points'].tensor = pl
        
        # Moving Obj.
        # 这里和仓库的Readme有出入，实际上只有lidar，没有Camera的噪声！
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'moving_bbox': # moving_noise_bbox
            from .LiDAR_corruptions import moving_noise_bbox
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo #! dxg bevfusion results['ann_info']['gt_bboxes_3d']
                data.append(results['ann_info']['gt_bboxes_3d'])
            points_aug = moving_noise_bbox(pl.numpy(), self.severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # Motion Blur
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'motion_blur': # motion_sim
            import numpy as np
            img = results['img'] # PIL.JpegImage
            if type(img) == list and len(img) == 6:
                # nuscenes
                img_aug = []
                for i in range(6):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    if i % 3 == 0:
                        image_np_aug = self.motion_blur_sim_c_frontback(image=img_np)
                    else:
                        image_np_aug = self.motion_blur_sim_c_leftright(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            elif type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(5):
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    if i == 0:
                        image_np_aug = self.motion_blur_sim_c_frontback(image=img_np)
                    else:
                        image_np_aug = self.motion_blur_sim_c_leftright(image=img_np)
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            else:
                # 判断是 kitti 数据集
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                # print('!!!!!!!!-----------------------------------!!!!!!!!!!')
                # print('attention the front-back image or the leftright image')
                # print(' different in kitti and nus  ')
                # print('!!!!!!!!-----------------------------------!!!!!!!!!!')

                image_aug_rgb = self.motion_sim_frontback(
                    image=img_rgb_255_np_uint8,
                    # watch_img=True,
                    # file_path='2.png'
                )
                image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                results['img'] = image_aug_bgr
        #! Object
        # density
        if self.corruption_type_l is not None and self.corruption_type_l == 'density_bbox': # density_dec_bbox
            from .LiDAR_corruptions import density_dec_bbox
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo/bevfusion
                data.append(results['ann_info']['gt_bboxes_3d'])
            points_aug = density_dec_bbox(pl.numpy(), self.severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # Local Cutout
        if self.corruption_type_l is not None and self.corruption_type_l == 'cutout_bbox':
            import numpy as np
            from .LiDAR_corruptions import cutout_bbox
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo/bevfusion
                data.append(results['ann_info']['gt_bboxes_3d'])
            severity = self.corruption_severity_dict['cutout_bbox']
            # aug_pl = pl[:,:3]
            points_aug = cutout_bbox(pl.numpy(), self.severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # Local Gaussian
        if self.corruption_type_l is not None and self.corruption_type_l == 'gaussian_noise_bbox':
            import numpy as np
            from .LiDAR_corruptions import gaussian_noise_bbox
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            points_aug = gaussian_noise_bbox(pl.numpy(), self.severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        # Shear
        if self.corruption_type_c is not None and self.corruption_type_c == 'shear': # bbox_shear
            import numpy as np
            
            img = results['img'] # PIL.JpegImage
            if 'lidar2image' in results: #! bevfusion
                lidar2img = results['lidar2image']  # nus:各论各的list / kitti: nparray
            
            if 'gt_bboxes_3d' in results:
                bboxes_corners = results['gt_bboxes_3d'].corners
                bboxes_centers = results['gt_bboxes_3d'].center
            else: #! bevfusion
                bboxes_corners = results['ann_info']['gt_bboxes_3d'].corners
                bboxes_centers = results['ann_info']['gt_bboxes_3d'].center
            if type(bboxes_corners) != int:
                # 变换矩阵（和彩新代码统一）
                c = [0.05, 0.1, 0.15, 0.2, 0.25][self.severity - 1]
                b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                transform_matrix = torch.tensor([
                    [1, 0, b],
                    [d, 1, e],
                    [f, 0, 1]
                ]).float()
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    lidar2img_i = torch.from_numpy(lidar2img[i])
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.shear_sim_c(
                                image=img_np,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True
                            )
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
            # kitti
            else:
                # 判断是 kitti 数据集
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                image_aug_rgb = self.shear_sim_c(
                    image=img_rgb_255_np_uint8,
                    bboxes_centers=bboxes_centers,
                    bboxes_corners=bboxes_corners,
                    transform_matrix=transform_matrix,
                    lidar2img=lidar2img
                )
                image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                results['img'] = image_aug_bgr

        # Scale
        if self.corruption_type_c is not None and self.corruption_type_c == 'scale': # bbox_scale
            import numpy as np
            
            img = results['img'] # PIL.JpegImage
            if 'lidar2image' in results: #! bevfusion
                lidar2img = results['lidar2image']  # nus:各论各的list / kitti: nparray
            
            if 'gt_bboxes_3d' in results:
                bboxes_corners = results['gt_bboxes_3d'].corners
                bboxes_centers = results['gt_bboxes_3d'].center
            else: #! bevfusion
                bboxes_corners = results['ann_info']['gt_bboxes_3d'].corners
                bboxes_centers = results['ann_info']['gt_bboxes_3d'].center
            if type(bboxes_corners) != int:
                # 变换矩阵（和彩新代码统一）
                c = [0.1, 0.2, 0.3, 0.4, 0.5][self.severity - 1]
                a = b = d = 1
                import numpy as np
                r = np.random.randint(0, 3)
                t = np.random.choice([-1, 1])
                a += c * t
                b += c * t
                d += c * t
                transform_matrix = torch.tensor([
                    [a, 0, 0],
                    [0, b, 0],
                    [0, 0, d],
                ]).float()
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    lidar2img_i = torch.from_numpy(lidar2img[i])
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.scale_sim_c(
                                image=img_np,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True
                            )
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
        # Rotation
        if self.corruption_type_c is not None and self.corruption_type_c == 'rotation': # bbox_rotate
            import numpy as np
            
            img = results['img'] # PIL.JpegImage
            if 'lidar2image' in results: #! bevfusion
                lidar2img = results['lidar2image']
            
            if 'gt_bboxes_3d' in results:
                bboxes_corners = results['gt_bboxes_3d'].corners
                bboxes_centers = results['gt_bboxes_3d'].center
            else: #! bevfusion
                bboxes_corners = results['ann_info']['gt_bboxes_3d'].corners
                bboxes_centers = results['ann_info']['gt_bboxes_3d'].center
            if type(bboxes_corners) != int:
                # 和彩新代码统一：
                # 仅绕z轴旋转
                theta_base = [4, 8, 12, 16, 20][self.severity - 1]
                theta_degree = np.random.uniform(theta_base - 2, theta_base + 2) * np.random.choice([-1, 1])

                theta = theta_degree / 180 * np.pi
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                # 非mono数据集，绕z轴旋转
                transform_matrix = torch.tensor([
                    [cos_theta, sin_theta, 0],
                    [-sin_theta, cos_theta, 0],
                    [0, 0, 1],
                ]).float()
                
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    lidar2img_i = torch.from_numpy(lidar2img[i])
                    img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                    image_np_aug = self.rotate_sim_c(
                                image=img_np,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True
                            )
                    img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img'] = img_aug
        # Spatial √
        if self.corruption_type_l is not None and self.corruption_type_l == 'spatial_aligment': # spatial_alignment_noise
            from .LiDAR_corruptions import spatial_alignment_noise
            # ori_pose = results['lidar2image'] # bevfusion
            # det3d
            for key, value in results['calib'].items():
                if 'lidar' in key:
                    results['calib'][key] = spatial_alignment_noise(value, self.severity)


        # Temporal
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'temporal_aligment': # temporal_alignment_noise
            ## 替换图片
            import numpy as np
            from PIL import Image
            from .LiDAR_corruptions import temporal_alignment_noise
            frame = temporal_alignment_noise(self.severity)
            img = results['img']
            cam_info = results['cam_sweeps']
            if len(cam_info) < frame - 1:
                assss = 1
            else:
                while (len(cam_info) <= frame-1):
                    frame = frame-1
                if type(img) == list and len(img) == 6:
                    img_aug = []
                    for i in range(6):
                        cam_key = list(cam_info[0].keys())
                        filename = cam_info[frame -1][cam_key[i]]['data_path']
                        # filename = '/data/public/nuscenes/'+results['cam_sweeps'][i][frame-1]['filename']
                        # _img = mmcv.imread(filename)
                        _img = Image.open(filename)
                        img_aug.append(_img)
                    results['img'] = img_aug
            
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'temporal_aligment': # temporal_alignment_noise
            ## 替换lidar
            import numpy as np
            from .LiDAR_corruptions import temporal_alignment_noise
            frame = temporal_alignment_noise(self.severity)
            lidar_info = results['sweeps']
            # print('len of lidar', len(lidar_info))
            if len(lidar_info) < frame-1:
                assss = 1
            else:
                while (len(lidar_info) <= frame-1):
                    frame = frame-1
                    # print(frame)
                sweep = lidar_info[frame-1]
                points_sweep = load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, 5)
                points_sweep = remove_close(points_sweep)
                # print(points_sweep.shape)
                results['points'].tensor = torch.from_numpy(points_sweep)
                # results['points'].tensor[:10000,:] = torch.from_numpy(points_sweep)[:10000,:] #[:,[0, 1, 2, 4]]
        return results, info
    


# class BasePoints:
#     """Base class for Points.

#     Args:
#         tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
#         points_dim (int): Number of the dimension of a point.
#             Each row is (x, y, z). Default to 3.
#         attribute_dims (dict): Dictionary to indicate the meaning of extra
#             dimension. Default to None.

#     Attributes:
#         tensor (torch.Tensor): Float matrix of N x points_dim.
#         points_dim (int): Integer indicating the dimension of a point.
#             Each row is (x, y, z, ...).
#         attribute_dims (bool): Dictionary to indicate the meaning of extra
#             dimension. Default to None.
#         rotation_axis (int): Default rotation axis for points rotation.
#     """

#     def __init__(self, tensor, points_dim=3, attribute_dims=None):
#         if isinstance(tensor, torch.Tensor):
#             device = tensor.device
#         else:
#             device = torch.device("cpu")
#         tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
#         if tensor.numel() == 0:
#             # Use reshape, so we don't end up creating a new tensor that
#             # does not depend on the inputs (and consequently confuses jit)
#             tensor = tensor.reshape((0, points_dim)).to(
#                 dtype=torch.float32, device=device
#             )
#         assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()

#         self.tensor = tensor
#         self.points_dim = points_dim
#         self.attribute_dims = attribute_dims
#         # after modification, axis=2 corresponds to z
#         self.rotation_axis = 2

#     @property
#     def coord(self):
#         """torch.Tensor: Coordinates of each point with size (N, 3)."""
#         return self.tensor[:, :3]

#     @coord.setter
#     def coord(self, tensor):
#         """Set the coordinates of each point."""
#         try:
#             tensor = tensor.reshape(self.shape[0], 3)
#         except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
#             raise ValueError(f"got unexpected shape {tensor.shape}")
#         if not isinstance(tensor, torch.Tensor):
#             tensor = self.tensor.new_tensor(tensor)
#         self.tensor[:, :3] = tensor

#     @property
#     def height(self):
#         """torch.Tensor: A vector with height of each point."""
#         if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
#             return self.tensor[:, self.attribute_dims["height"]]
#         else:
#             return None

#     @height.setter
#     def height(self, tensor):
#         """Set the height of each point."""
#         try:
#             tensor = tensor.reshape(self.shape[0])
#         except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
#             raise ValueError(f"got unexpected shape {tensor.shape}")
#         if not isinstance(tensor, torch.Tensor):
#             tensor = self.tensor.new_tensor(tensor)
#         if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
#             self.tensor[:, self.attribute_dims["height"]] = tensor
#         else:
#             # add height attribute
#             if self.attribute_dims is None:
#                 self.attribute_dims = dict()
#             attr_dim = self.shape[1]
#             self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
#             self.attribute_dims.update(dict(height=attr_dim))
#             self.points_dim += 1

#     @property
#     def color(self):
#         """torch.Tensor: A vector with color of each point."""
#         if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
#             return self.tensor[:, self.attribute_dims["color"]]
#         else:
#             return None

#     @color.setter
#     def color(self, tensor):
#         """Set the color of each point."""
#         try:
#             tensor = tensor.reshape(self.shape[0], 3)
#         except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
#             raise ValueError(f"got unexpected shape {tensor.shape}")
#         if tensor.max() >= 256 or tensor.min() < 0:
#             warnings.warn("point got color value beyond [0, 255]")
#         if not isinstance(tensor, torch.Tensor):
#             tensor = self.tensor.new_tensor(tensor)
#         if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
#             self.tensor[:, self.attribute_dims["color"]] = tensor
#         else:
#             # add color attribute
#             if self.attribute_dims is None:
#                 self.attribute_dims = dict()
#             attr_dim = self.shape[1]
#             self.tensor = torch.cat([self.tensor, tensor], dim=1)
#             self.attribute_dims.update(
#                 dict(color=[attr_dim, attr_dim + 1, attr_dim + 2])
#             )
#             self.points_dim += 3

#     @property
#     def shape(self):
#         """torch.Shape: Shape of points."""
#         return self.tensor.shape

#     def shuffle(self):
#         """Shuffle the points.

#         Returns:
#             torch.Tensor: The shuffled index.
#         """
#         idx = torch.randperm(self.__len__(), device=self.tensor.device)
#         self.tensor = self.tensor[idx]
#         return idx

#     def rotate(self, rotation, axis=None):
#         """Rotate points with the given rotation matrix or angle.

#         Args:
#             rotation (float, np.ndarray, torch.Tensor): Rotation matrix
#                 or angle.
#             axis (int): Axis to rotate at. Defaults to None.
#         """
#         if not isinstance(rotation, torch.Tensor):
#             rotation = self.tensor.new_tensor(rotation)
#         assert (
#             rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
#         ), f"invalid rotation shape {rotation.shape}"

#         if axis is None:
#             axis = self.rotation_axis

#         if rotation.numel() == 1:
#             rot_sin = torch.sin(rotation)
#             rot_cos = torch.cos(rotation)
#             if axis == 1:
#                 rot_mat_T = rotation.new_tensor(
#                     [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
#                 )
#             elif axis == 2 or axis == -1:
#                 rot_mat_T = rotation.new_tensor(
#                     [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
#                 )
#             elif axis == 0:
#                 rot_mat_T = rotation.new_tensor(
#                     [[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]]
#                 )
#             else:
#                 raise ValueError("axis should in range")
#             rot_mat_T = rot_mat_T.T
#         elif rotation.numel() == 9:
#             rot_mat_T = rotation
#         else:
#             raise NotImplementedError
#         self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T

#         return rot_mat_T

#     @abstractmethod
#     def flip(self, bev_direction="horizontal"):
#         """Flip the points in BEV along given BEV direction."""
#         pass

#     def translate(self, trans_vector):
#         """Translate points with the given translation vector.

#         Args:
#             trans_vector (np.ndarray, torch.Tensor): Translation
#                 vector of size 3 or nx3.
#         """
#         if not isinstance(trans_vector, torch.Tensor):
#             trans_vector = self.tensor.new_tensor(trans_vector)
#         trans_vector = trans_vector.squeeze(0)
#         if trans_vector.dim() == 1:
#             assert trans_vector.shape[0] == 3
#         elif trans_vector.dim() == 2:
#             assert (
#                 trans_vector.shape[0] == self.tensor.shape[0]
#                 and trans_vector.shape[1] == 3
#             )
#         else:
#             raise NotImplementedError(
#                 f"Unsupported translation vector of shape {trans_vector.shape}"
#             )
#         self.tensor[:, :3] += trans_vector

#     def in_range_3d(self, point_range):
#         """Check whether the points are in the given range.

#         Args:
#             point_range (list | torch.Tensor): The range of point
#                 (x_min, y_min, z_min, x_max, y_max, z_max)

#         Note:
#             In the original implementation of SECOND, checking whether
#             a box in the range checks whether the points are in a convex
#             polygon, we try to reduce the burden for simpler cases.

#         Returns:
#             torch.Tensor: A binary vector indicating whether each point is \
#                 inside the reference range.
#         """
#         in_range_flags = (
#             (self.tensor[:, 0] > point_range[0])
#             & (self.tensor[:, 1] > point_range[1])
#             & (self.tensor[:, 2] > point_range[2])
#             & (self.tensor[:, 0] < point_range[3])
#             & (self.tensor[:, 1] < point_range[4])
#             & (self.tensor[:, 2] < point_range[5])
#         )
#         return in_range_flags

#     @abstractmethod
#     def in_range_bev(self, point_range):
#         """Check whether the points are in the given range.

#         Args:
#             point_range (list | torch.Tensor): The range of point
#                 in order of (x_min, y_min, x_max, y_max).

#         Returns:
#             torch.Tensor: Indicating whether each point is inside \
#                 the reference range.
#         """
#         pass

#     @abstractmethod
#     def convert_to(self, dst, rt_mat=None):
#         """Convert self to ``dst`` mode.

#         Args:
#             dst (:obj:`CoordMode`): The target Box mode.
#             rt_mat (np.ndarray | torch.Tensor): The rotation and translation
#                 matrix between different coordinates. Defaults to None.
#                 The conversion from `src` coordinates to `dst` coordinates
#                 usually comes along the change of sensors, e.g., from camera
#                 to LiDAR. This requires a transformation matrix.

#         Returns:
#             :obj:`BasePoints`: The converted box of the same type \
#                 in the `dst` mode.
#         """
#         pass

#     def scale(self, scale_factor):
#         """Scale the points with horizontal and vertical scaling factors.

#         Args:
#             scale_factors (float): Scale factors to scale the points.
#         """
#         self.tensor[:, :3] *= scale_factor

#     def __getitem__(self, item):
#         """
#         Note:
#             The following usage are allowed:
#             1. `new_points = points[3]`:
#                 return a `Points` that contains only one point.
#             2. `new_points = points[2:10]`:
#                 return a slice of points.
#             3. `new_points = points[vector]`:
#                 where vector is a torch.BoolTensor with `length = len(points)`.
#                 Nonzero elements in the vector will be selected.
#             4. `new_points = points[3:11, vector]`:
#                 return a slice of points and attribute dims.
#             5. `new_points = points[4:12, 2]`:
#                 return a slice of points with single attribute.
#             Note that the returned Points might share storage with this Points,
#             subject to Pytorch's indexing semantics.

#         Returns:
#             :obj:`BasePoints`: A new object of  \
#                 :class:`BasePoints` after indexing.
#         """
#         original_type = type(self)
#         if isinstance(item, int):
#             return original_type(
#                 self.tensor[item].view(1, -1),
#                 points_dim=self.points_dim,
#                 attribute_dims=self.attribute_dims,
#             )
#         elif isinstance(item, tuple) and len(item) == 2:
#             if isinstance(item[1], slice):
#                 start = 0 if item[1].start is None else item[1].start
#                 stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
#                 step = 1 if item[1].step is None else item[1].step
#                 item = list(item)
#                 item[1] = list(range(start, stop, step))
#                 item = tuple(item)
#             elif isinstance(item[1], int):
#                 item = list(item)
#                 item[1] = [item[1]]
#                 item = tuple(item)
#             p = self.tensor[item[0], item[1]]

#             keep_dims = list(
#                 set(item[1]).intersection(set(range(3, self.tensor.shape[1])))
#             )
#             if self.attribute_dims is not None:
#                 attribute_dims = self.attribute_dims.copy()
#                 for key in self.attribute_dims.keys():
#                     cur_attribute_dims = attribute_dims[key]
#                     if isinstance(cur_attribute_dims, int):
#                         cur_attribute_dims = [cur_attribute_dims]
#                     intersect_attr = list(
#                         set(cur_attribute_dims).intersection(set(keep_dims))
#                     )
#                     if len(intersect_attr) == 1:
#                         attribute_dims[key] = intersect_attr[0]
#                     elif len(intersect_attr) > 1:
#                         attribute_dims[key] = intersect_attr
#                     else:
#                         attribute_dims.pop(key)
#             else:
#                 attribute_dims = None
#         elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
#             p = self.tensor[item]
#             attribute_dims = self.attribute_dims
#         else:
#             raise NotImplementedError(f"Invalid slice {item}!")

#         assert (
#             p.dim() == 2
#         ), f"Indexing on Points with {item} failed to return a matrix!"
#         return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

#     def __len__(self):
#         """int: Number of points in the current object."""
#         return self.tensor.shape[0]

#     def __repr__(self):
#         """str: Return a strings that describes the object."""
#         return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

#     @classmethod
#     def cat(cls, points_list):
#         """Concatenate a list of Points into a single Points.

#         Args:
#             points_list (list[:obj:`BasePoints`]): List of points.

#         Returns:
#             :obj:`BasePoints`: The concatenated Points.
#         """
#         assert isinstance(points_list, (list, tuple))
#         if len(points_list) == 0:
#             return cls(torch.empty(0))
#         assert all(isinstance(points, cls) for points in points_list)

#         # use torch.cat (v.s. layers.cat)
#         # so the returned points never share storage with input
#         cat_points = cls(
#             torch.cat([p.tensor for p in points_list], dim=0),
#             points_dim=points_list[0].tensor.shape[1],
#             attribute_dims=points_list[0].attribute_dims,
#         )
#         return cat_points

#     def to(self, device):
#         """Convert current points to a specific device.

#         Args:
#             device (str | :obj:`torch.device`): The name of the device.

#         Returns:
#             :obj:`BasePoints`: A new boxes object on the \
#                 specific device.
#         """
#         original_type = type(self)
#         return original_type(
#             self.tensor.to(device),
#             points_dim=self.points_dim,
#             attribute_dims=self.attribute_dims,
#         )

#     def clone(self):
#         """Clone the Points.

#         Returns:
#             :obj:`BasePoints`: Box object with the same properties \
#                 as self.
#         """
#         original_type = type(self)
#         return original_type(
#             self.tensor.clone(),
#             points_dim=self.points_dim,
#             attribute_dims=self.attribute_dims,
#         )

#     @property
#     def device(self):
#         """str: The device of the points are on."""
#         return self.tensor.device

#     def __iter__(self):
#         """Yield a point as a Tensor of shape (4,) at a time.

#         Returns:
#             torch.Tensor: A point of shape (4,).
#         """
#         yield from self.tensor

#     def new_point(self, data):
#         """Create a new point object with data.

#         The new point and its tensor has the similar properties \
#             as self and self.tensor, respectively.

#         Args:
#             data (torch.Tensor | numpy.array | list): Data to be copied.

#         Returns:
#             :obj:`BasePoints`: A new point object with ``data``, \
#                 the object's other properties are similar to ``self``.
#         """
#         new_tensor = (
#             self.tensor.new_tensor(data)
#             if not isinstance(data, torch.Tensor)
#             else data.to(self.device)
#         )
#         original_type = type(self)
#         return original_type(
#             new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims
#         )
