import mmcv
import warnings
from copy import deepcopy
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str



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


@PIPELINES.register_module()
class CorruptionMethods(object):
    """Test-time augmentation with corruptions.
    """

    def __init__(self):
        # 能作为全局设定存在的，应是指定：
        # 1.用什么corruption. 2.扰动的程度
        from .corruptions_config import Corruptions_mode
        cor = Corruptions_mode()
        self.sam_ad = False
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
  
    def camera_sim_init(self):
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
        
        if self.corruption_type_c is not None and 'sunlight_f' in self.corruption_type_c: 
            #! sun_sim 可以对点云和图像一起加噪声，sun_sim_mono 只对图像加噪声，还有 scene_glare_noise 可以只对 Lidar 加噪声
            import numpy as np
            np.random.seed(2022)
            from .Camera_corruptions import ImagePointAddSun, ImageAddSunMono
            # 点云和图像双重加噪
            print("***************************************************************")
            self.sun_sim = ImagePointAddSun(self.severity)

        
            
    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        if self.sam_ad:
            import os
            import random
            import numpy as np
            from PIL import Image
            severity = random.randint(1,5)
            cor_list = ['snow', 'rain', 'fog', 'sunlight']
            cor_i = random.randint(0,3)
            cor = cor_list[cor_i]
            print(f'{cor}_{severity}')
            img = results['img_filename'] # PIL.JpegImage
            if cor == 'snow':
                from .Camera_corruptions import ImageAddSnow
                snow_sim_c = ImageAddSnow(severity, seed=2022)
                img_aug = []
                for i in range(len(img)):
                    path_tmp = results['img_filename'][0].split('/')
                    path_tmp[4] = path_tmp[4] + '_cor' # sample_cor
                    path_tmp[5] = path_tmp[5] + f'_{cor}_{severity}' # CAM_FRONT_corruption_severity
                    img_cor_p = '/'.join(path_tmp[:-1])
                    img_cor_f = '/'.join(path_tmp)
                    if os.path.isfile(img_cor_f): # 尝试读离线数据
                        image_np_aug = mmcv.imread(img_cor_f)
                    else: # 如果没有离线数据就生成在线数据并保存
                        img_np_aug = img[i]
                        img_np = np.array(img[i])
                        image_np_aug = snow_sim_c(image=img_np)
                        # 保存离线版
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        os.makedirs(img_cor_p, exist_ok=True)
                        Image.Image.save(img_PIL_aug,fp=img_cor_f)
                    img_aug.append(image_np_aug)
                results['img_filename'] = img_aug

            if cor == 'rain':
                from .Camera_corruptions import ImageAddRain
                rain_sim_c = ImageAddRain(severity, seed=2022)
                img_aug = []
                for i in range(len(img)):
                    path_tmp = results['img_filename'][0].split('/')
                    path_tmp[4] = path_tmp[4] + '_cor' # sample_cor
                    path_tmp[5] = path_tmp[5] + f'_{cor}_{severity}' # CAM_FRONT_corruption_severity
                    img_cor_p = '/'.join(path_tmp[:-1])
                    img_cor_f = '/'.join(path_tmp)
                    if os.path.isfile(img_cor_f): # 尝试读离线数据
                        image_np_aug = mmcv.imread(img_cor_f)
                    else: # 如果没有离线数据就生成在线数据并保存
                        img_np_aug = img[i]
                        img_np = np.array(img[i])
                        image_np_aug = rain_sim_c(image=img_np)
                        # 保存离线版
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        os.makedirs(img_cor_p, exist_ok=True)
                        Image.Image.save(img_PIL_aug,fp=img_cor_f)
                    img_aug.append(image_np_aug)
                results['img_filename'] = img_aug
                

            if cor == 'fog':
                from .Camera_corruptions import ImageAddFog
                fog_sim_c = ImageAddFog(severity, seed=2022)
                img_aug = []
                for i in range(len(img)):
                    path_tmp = results['img_filename'][0].split('/')
                    path_tmp[4] = path_tmp[4] + '_cor' # sample_cor
                    path_tmp[5] = path_tmp[5] + f'_{cor}_{severity}' # CAM_FRONT_corruption_severity
                    img_cor_p = '/'.join(path_tmp[:-1])
                    img_cor_f = '/'.join(path_tmp)
                    if os.path.isfile(img_cor_f): # 尝试读离线数据
                        image_np_aug = mmcv.imread(img_cor_f)
                    else: # 如果没有离线数据就生成在线数据并保存
                        img_np_aug = img[i]
                        img_np = np.array(img[i])
                        image_np_aug = fog_sim_c(image=img_np)
                        # 保存离线版
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        os.makedirs(img_cor_p, exist_ok=True)
                        Image.Image.save(img_PIL_aug,fp=img_cor_f)
                    img_aug.append(image_np_aug)
                results['img_filename'] = img_aug
                
            if cor == 'sunlight':
                # 点云和图像双重加噪
                import numpy as np
                from .Camera_corruptions import ImagePointAddSun, ImageAddSunMono
                np.random.seed(2022)
                if 'lidar2img' in results: # mmdet3d transfuison
                    lidar2img = results['lidar2img']
                points_tensor = results['points'].tensor
                sun_sim = ImagePointAddSun(severity)
                img_aug = []
                for i in range(len(img)):
                    path_tmp = results['img_filename'][i].split('/')
                    path_tmp[4] = path_tmp[4] + '_cor' # sample_cor
                    path_tmp[5] = path_tmp[5] + f'_{cor}_{severity}' # CAM_FRONT_corruption_severity
                    img_cor_p = '/'.join(path_tmp[:-1])
                    img_cor_f = '/'.join(path_tmp)
                    if os.path.isfile(img_cor_f): # 尝试读离线数据
                        image_np_aug = mmcv.imread(img_cor_f)
                    else: # 如果没有离线数据就生成在线数据并保存
                        img_np = np.array(img[i])
                        lidar2img0_np = lidar2img[0]
                        lidar2img0_tensor = torch.from_numpy(lidar2img0_np)
                        image_np_aug, points_aug = sun_sim(image=img_np, points=points_tensor, lidar2img=lidar2img0_tensor)
                        # image_np_aug = sun_sim.sun_sim_img(image=img_np, lidar2img=None, severity=severity)
                        # 保存离线版
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        os.makedirs(img_cor_p, exist_ok=True)
                        Image.Image.save(img_PIL_aug,fp=img_cor_f)
                    img_aug.append(image_np_aug)
                results['img_filename'] = img_aug
                
                
            return results
        
        
        #! dxg Clear
        if self.severity == 0:
            return results
        #! Weather
        #########################
        #         snow
        #########################
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'snow': # snow_sim
            import numpy as np
            from PIL import Image
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    if self.offline_flag:
                        import os
                        path_tmp = results['img_filename'][0].split('/')
                        path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                        path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_c}_{self.severity}' # CAM_FRONT_corruption_severity
                        img_cor_f = '/'.join(path_tmp)
                        if os.path.isfile(img_cor_f):
                            # img_PIL_aug = Image.open(img_cor_f)
                            img_np_aug = mmcv.imread(img_cor_f)
                        else:
                            img_np_aug = img[i]
                        img_aug.append(img_np_aug)
                    else:
                        img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                        image_np_aug = self.snow_sim_c(image=img_np)
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        img_aug.append(image_np_aug)
                        if self.save_cor_flag:
                            import os
                            path_tmp = results['img_filename'][0].split('/')
                            path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                            path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_c}_{self.severity}' # CAM_FRONT_corruption_severity
                            img_cor_p = '/'.join(path_tmp[:-1])
                            img_cor_f = '/'.join(path_tmp)
                            os.makedirs(img_cor_p, exist_ok=True)
                            Image.Image.save(img_PIL_aug,fp=img_cor_f)
                results['img_filename'] = img_aug
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l =='snow': # snow_sim_lidar
            import numpy as np
            from .LiDAR_corruptions import snow_sim, snow_sim_nus
            pl = results['points'].tensor
            if self.offline_flag:
                import os
                import pickle
                path_tmp = results['pts_filename'].split('/')
                path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # LIDAR_TOP_corruption_severity
                pc_cor_f = '/'.join(path_tmp)
                if os.path.isfile(pc_cor_f):
                    pl=np.fromfile(pc_cor_f,dtype=np.float32).reshape(-1,5)
                    pl = torch.from_numpy(pl)
            else:
                try:
                    points_aug = snow_sim_nus(pl.numpy(), self.severity)
                    pl = torch.from_numpy(points_aug)
                except:
                    print('snow add err')
                results['points'].tensor = pl
                if self.save_cor_flag:
                    import os
                    import pickle
                    path_tmp = results['pts_filename'].split('/')
                    path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                    path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # LIDAR_TOP_corruption_severity
                    pc_cor_p = '/'.join(path_tmp[:-1])
                    pc_cor_f = '/'.join(path_tmp)
                    os.makedirs(pc_cor_p, exist_ok=True)
                    with open(pc_cor_f + '.pkl', 'wb') as f:
                        pickle.dump(pl, f) # tensor 但是保存为*.pkl
                    pl.numpy().astype(np.float32).tofile(pc_cor_f) # numpy保存，
        
        #########################
        #         rain
        #########################
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'rain': # rain_sim
            import numpy as np
            from PIL import Image
            
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    if self.offline_flag:
                        import os
                        path_tmp = results['img_filename'][0].split('/')
                        path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                        path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_c}_{self.severity}' # CAM_FRONT_corruption_severity
                        img_cor_f = '/'.join(path_tmp)
                        if os.path.isfile(img_cor_f):
                            # img_PIL_aug = Image.open(img_cor_f)
                            img_np_aug = mmcv.imread(img_cor_f)
                        else:
                            img_np_aug = img[i]
                        img_aug.append(img_np_aug)
                    else:
                        img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                        image_np_aug = self.rain_sim_c(image=img_np)
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        img_aug.append(image_np_aug)
                        if self.save_cor_flag:
                            import os
                            path_tmp = results['img_filename'][0].split('/')
                            path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                            path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_c}_{self.severity}' # CAM_FRONT_corruption_severity
                            img_cor_p = '/'.join(path_tmp[:-1])
                            img_cor_f = '/'.join(path_tmp)
                            os.makedirs(img_cor_p, exist_ok=True)
                            Image.Image.save(img_PIL_aug,fp=img_cor_f)
                results['img_filename'] = img_aug
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'rain':
            import numpy as np
            from .LiDAR_corruptions import rain_sim
            pl = results['points'].tensor
            if self.offline_flag:
                import os
                import pickle
                path_tmp = results['pts_filename'].split('/')
                path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # LIDAR_TOP_corruption_severity
                pc_cor_f = '/'.join(path_tmp)
                if os.path.isfile(pc_cor_f):
                    pl=np.fromfile(pc_cor_f,dtype=np.float32).reshape(-1,5)
                    pl = torch.from_numpy(pl)
            else:
                aug_pl = pl[:,:4]
                points_aug = rain_sim(aug_pl.numpy(), self.severity)
                pl[:,:4] = torch.from_numpy(points_aug)
                if self.save_cor_flag:
                    import os
                    import pickle
                    path_tmp = results['pts_filename'].split('/')
                    path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                    path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # LIDAR_TOP_corruption_severity
                    pc_cor_p = '/'.join(path_tmp[:-1])
                    pc_cor_f = '/'.join(path_tmp)
                    os.makedirs(pc_cor_p, exist_ok=True)
                    with open(pc_cor_f + '.pkl', 'wb') as f:
                        pickle.dump(pl, f) # tensor 但是保存为*.pkl
                    pl.numpy().astype(np.float32).tofile(pc_cor_f) # numpy保存，
            results['points'].tensor = pl


        #########################
        #         fog
        #########################
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'fog':
            import numpy as np
            from PIL import Image
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    if self.offline_flag:
                        import os
                        path_tmp = results['img_filename'][0].split('/')
                        path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                        path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_c}_{self.severity}' # CAM_FRONT_corruption_severity
                        img_cor_f = '/'.join(path_tmp)
                        if os.path.isfile(img_cor_f):
                            # img_PIL_aug = Image.open(img_cor_f)
                            img_np_aug = mmcv.imread(img_cor_f)
                        else:
                            img_np_aug = img[i]
                        img_aug.append(img_np_aug)
                    else:
                        img_np = np.array(img[i]) # PIL.JpegImage -> ndarry
                        image_np_aug = self.fog_sim_c(image=img_np)
                        img_PIL_aug = Image.fromarray(image_np_aug)
                        img_aug.append(image_np_aug)
                        if self.save_cor_flag:
                            import os
                            path_tmp = results['img_filename'][i].filename.split('/')
                            path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                            path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # CAM_FRONT_corruption_severity
                            pc_cor_p = '/'.join(path_tmp[:-1])
                            pc_cor_f = '/'.join(path_tmp)
                            os.makedirs(pc_cor_p, exist_ok=True)
                            Image.Image.save(img_PIL_aug,fp=pc_cor_f)
                results['img_filename'] = img_aug
                
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'fog': # fog_sim_lidar
            import numpy as np
            from .LiDAR_corruptions import fog_sim
            pl = results['points'].tensor
            aug_pl = pl[:,:4]
            points_aug = fog_sim(aug_pl.numpy(), self.severity)
            pl[:,:4] = torch.from_numpy(points_aug)
            results['points'].tensor = pl
            if self.save_cor_flag:
                import os
                import pickle
                path_tmp = results['pts_filename'].split('/')
                path_tmp[3] = path_tmp[3] + '_cor' # sample_cor
                path_tmp[4] = path_tmp[4] + f'_{self.corruption_type_l}_{self.severity}' # LIDAR_TOP_corruption_severity
                pc_cor_p = '/'.join(path_tmp[:-1])
                pc_cor_f = '/'.join(path_tmp)
                os.makedirs(pc_cor_p, exist_ok=True)
                with open(pc_cor_f + '.pkl', 'wb') as f:
                    pickle.dump(pl, f) # tensor 但是保存为*.pkl
                pl.numpy().astype(np.float32).tofile(pc_cor_f) # numpy保存，
                
        #########################
        #       sunlight
        #########################
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'sunlight_f': # sun_sim
            import numpy as np
            # Transfusion读取的图像是使用mmcv.imread()读取的是RGB通道的图像
            # 但是代码写的是读取BGR图像，存BGR图像
            # 这里改为读PIL.JpegImage  转为ndarray处理，再存PIL.JpegImage  
            img = results['img_filename'] # [PIL.JpegImage * 6]
            if 'lidar2img' in results: # bevfusion
                lidar2img = results['lidar2img']
            if 'lidar2img' in results: # mmdet3d transfuison
                lidar2img = results['lidar2img']
            points_tensor = results['points'].tensor
            # print(points_tensor.shape)
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
                    # img0_PIL_aug = Image.fromarray(img0_np_aug)
                    img[0] = img0_np_aug
                    results['img_filename'] = img
                    # print(points_aug.shape)
                    if points_aug:
                        results['points'].tensor = points_aug
        
        if self.corruption_type_l is not None and self.corruption_type_l == 'density': # density_dec_global
            from .LiDAR_corruptions import density_dec_global
            pl = results['points'].tensor
            #import pdb;pdb.set_trace()
            #print("111111111111111111")
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
        ## Lidar
        if self.corruption_type_l is not None and self.corruption_type_l == 'gaussian': # gaussian_noise
            from .LiDAR_corruptions import gaussian_noise
            pl = results['points'].tensor # tensor [N,5]
            points_aug = gaussian_noise(pl.numpy(), self.severity)
            #import pdb;pdb.set_trace()
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
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
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = img[i] # PIL.JpegImage -> ndarry
                    image_np_aug = self.gaussian_sim_c(image=img_np)
                    # img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img_filename'] = img_aug
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
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = img[i] # PIL.JpegImage -> ndarry
                    image_np_aug = self.uniform_sim_c(image=img_np)
                    #img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img_filename'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img[:,:,[2,1,0]]
                image_aug_rgb = self.uniform_sim_c(image=img_rgb_255_np_uint8)
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img_filename'] = image_aug_bgr
        # impulse_c
        ## Camera
        if self.corruption_type_c is not None and self.corruption_type_c == 'impulse':
            import numpy as np
            img = results['img_filename'] # PIL.JpegImage
            # nuscenes
            if type(img) == list and len(img) == 6 or type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(len(img)):
                    img_np = img[i] # PIL.JpegImage -> ndarry
                    image_np_aug = self.impulse_sim_c(image=img_np)
                    #img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img_filename'] = img_aug
            # kitti
            else:
                img_rgb_255_np_uint8 = img[:,:,[2,1,0]]
                image_aug_rgb = self.impulse_sim_c(image=img_rgb_255_np_uint8)
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img_filename'] = image_aug_bgr
        
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
            img = results['img_filename'] # PIL.JpegImage
            if type(img) == list and len(img) == 6:
                # nuscenes
                img_aug = []
                for i in range(6):
                    img_np = img[i] # PIL.JpegImage -> ndarry
                    if i % 3 == 0:
                        image_np_aug = self.motion_blur_sim_c_frontback(image=img_np)
                    else:
                        image_np_aug = self.motion_blur_sim_c_leftright(image=img_np)
                    #img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img_filename'] = img_aug
            elif type(img) == list and len(img) == 5:
                img_aug = []
                for i in range(5):
                    img_np = img[i] # PIL.JpegImage -> ndarry
                    if i == 0:
                        image_np_aug = self.motion_blur_sim_c_frontback(image=img_np)
                    else:
                        image_np_aug = self.motion_blur_sim_c_leftright(image=img_np)
                    #img_PIL_aug = Image.fromarray(image_np_aug)
                    img_aug.append(img_PIL_aug)
                results['img_filename'] = img_aug
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
                results['img_filename'] = image_aug_bgr
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

                     
        
        return results