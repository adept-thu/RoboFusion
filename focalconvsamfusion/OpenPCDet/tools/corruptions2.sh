#!/usr/bin/env bash

# CUDA 选择
# CUDA_VISIBLE_DEVICES=0, 1, 2, 3
gpu_num=2 #gpu数量

# test method samv3
# cfg_file=cfgs/kitti_models/sam_onlinev3.yaml
# batch_size=1 # total batch size
# ckpt=../checkpoints/sam_onlinev3/v3_avg_optim/ckpt/checkpoint_epoch_73.pth

cfg_file=../tools/cfgs/kitti_models/mobilesam_onlinev4.yaml
batch_size=16 # total batch size
ckpt=../output/kitti_models/mobilesam_onlinev4/wavemobilesamv6/ckpt/checkpoint_epoch_80.pth


# corruptions
weather=(snow rain fog glare) # glare=sunglight
sensor=(density cutout crosstalk gaussian_l uniform_l impulse_l gaussian_c uniform_c impulse_c)
motion=(compensation moving_bbox motion_blur)
object=(density_bbox cutout_bbox gaussian_bbox_l uniform_bbox_l impulse_bbox_l shear scale rotation)
alignment=(spatial_aligment temporal_aligment)
all=(None ${weather[@]} ${sensor[@]} ${motion[@]} ${object[@]} ${alignment[@]})

# setting
demo1=(glare)

lidar_only_kitti=()
fusion_kitti=()

for corruptions in ${demo1[@]}
do
if [ $corruptions == snow ]
then
corruptions_l=snow
corruptions_c=snow
elif [ $corruptions == rain ]
then
corruptions_l=rain
corruptions_c=rain
elif [ $corruptions == fog ]
then
corruptions_l=fog
corruptions_c=fog
elif [ $corruptions == glare ]
then
corruptions_l=glare
corruptions_c=glare
elif [ $corruptions == density ]
then
corruptions_l=density
corruptions_c=None
elif [ $corruptions == cutout ]
then
corruptions_l=cutout
corruptions_c=None
elif [ $corruptions == crosstalk ]
then
corruptions_l=crosstalk
corruptions_c=None
elif [ $corruptions == gaussian_l ]
then
corruptions_l=gaussian_l
corruptions_c=None
elif [ $corruptions == uniform_l ]
then
corruptions_l=uniform_l
corruptions_c=None
elif [ $corruptions == impulse_l ]
then
corruptions_l=impulse_l
corruptions_c=None
elif [ $corruptions == gaussian_c ]
then
corruptions_l=None
corruptions_c=gaussian_c
elif [ $corruptions == uniform_c ]
then
corruptions_l=None
corruptions_c=uniform_c
elif [ $corruptions == impulse_c ]
then
corruptions_l=None
corruptions_c=impulse_c
elif [ $corruptions == compensation ]
then
corruptions_l=compensation
corruptions_c=None
elif [ $corruptions == moving_bbox ]
then
corruptions_l=moving_bbox
corruptions_c=moving_bbox
elif [ $corruptions == motion_blur ]
then
corruptions_l=None
corruptions_c=motion_blur
elif [ $corruptions == density_bbox ]
then
corruptions_l=density_bbox
corruptions_c=density_bbox
elif [ $corruptions == cutout_bbox ]
then
corruptions_l=cutout_bbox
corruptions_c=None
elif [ $corruptions == gaussian_bbox_l ]
then
corruptions_l=gaussian_bbox_l
corruptions_c=None
elif [ $corruptions == uniform_bbox_l ]
then
corruptions_l=uniform_bbox_l
corruptions_c=None
elif [ $corruptions == impulse_bbox_l ]
then
corruptions_l=impulse_bbox_l
corruptions_c=None
elif [ $corruptions == shear_bbox ]
then
corruptions_l=shear_bbox
corruptions_c=shear_bbox
elif [ $corruptions == scale_bbox ]
then
corruptions_l=scale_bbox
corruptions_c=scale_bbox
elif [ $corruptions == rotation_bbox ]
then
corruptions_l=rotation_bbox
corruptions_c=rotation_bbox
elif [ $corruptions == spatial_aligment ]
then
corruptions_l=spatial_aligment
corruptions_c=None
elif [ $corruptions == temporal_aligment ]
then
corruptions_l=temporal_aligment
corruptions_c=None
else
corruptions_l=None
corruptions_c=None
fi

echo ">>>>>>>>>>>>>>>>>>>>>>" "corruptions" $corruptions
echo ">>>>>>>>>>>>>>>>>>>>>>" "Lidar" $corruptions_l
echo ">>>>>>>>>>>>>>>>>>>>>>" "Camera" $corruptions_c

for severity in $(seq 3 3); do

echo ">>>>>>>>>>>>>>>>>>>>>>" "severity" $severity
echo "scripts/dist_test1.sh $gpu_num --cfg_file $cfg_file \
    --batch_size $batch_size --ckpt $ckpt --corruptions $corruptions_l $corruptions_c \
    --severity $severity --extra_tag $corruptions &"
scripts/dist_test1.sh $gpu_num --cfg_file $cfg_file  \
    --batch_size $batch_size --ckpt $ckpt --corruptions $corruptions_l $corruptions_c \
    --severity $severity --extra_tag $corruptions &

# echo ">>>>>>>>>>>>>>>>>>>>>>" "severity" $severity
# echo "python test.py --cfg_file $cfg_file \
#     --batch_size $batch_size --ckpt $ckpt --corruptions $corruptions_l $corruptions_c \
#     --severity $severity --extra_tag $corruptions &"

# CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file $cfg_file  \
#     --batch_size $batch_size --ckpt $ckpt --corruptions $corruptions_l $corruptions_c \
#     --severity $severity --extra_tag $corruptions &
wait

done
done