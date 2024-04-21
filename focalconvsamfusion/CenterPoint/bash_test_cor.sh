#####################################
#               选择模型
#####################################
CONFIG="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_focal_multimodal_1_4_data"
# ckpt=./work_dirs/configs/nusc/voxelnet/sam_onlinev3_1_4_data/latest.pth
ckpt=../checkpoints/centerpoint_focal_multimodal_1_4_data.pth


#####################################
#          选择批量处理的噪声
#          下面列出了全部的噪声
#####################################

# corruptions
weather=(snow rain fog sunlight)
sensor=(density cutout crosstalk gaussian_l uniform_l impulse_l gaussian_c uniform_c impulse_c)
motion=(compensation moving_bbox motion_blur)
object=(density_bbox cutout_bbox gaussian_bbox_l uniform_bbox_l impulse_bbox_l shear scale rotation)
alignment=(spatial_aligment temporal_aligment)
all=(${weather[@]} ${sensor[@]} ${motion[@]} ${object[@]} ${alignment[@]})

cor_list=(gaussian_l spatial_aligment)

#####################################
#          缩短推理时间的参数
# 
#####################################
workers=4
batch_size=4
NUM_GPUS=4

for corruptions in ${cor_list[@]} # 遍历指定子集
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
elif [ $corruptions == sunlight ]
then
corruptions_l=sunlight_f
corruptions_c=sunlight_f
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
corruptions_l=gaussian
corruptions_c=None
elif [ $corruptions == uniform_l ]
then
corruptions_l=uniform
corruptions_c=None
elif [ $corruptions == impulse_l ]
then
corruptions_l=impulse
corruptions_c=None
elif [ $corruptions == gaussian_c ]
then
corruptions_l=None
corruptions_c=gaussian
elif [ $corruptions == uniform_c ]
then
corruptions_l=None
corruptions_c=uniform
elif [ $corruptions == impulse_c ]
then
corruptions_l=None
corruptions_c=impulse
elif [ $corruptions == compensation ]
then
corruptions_l=compensation
corruptions_c=None
elif [ $corruptions == moving_bbox ]
then
corruptions_l=moving_bbox
corruptions_c=None
elif [ $corruptions == motion_blur ]
then
corruptions_l=None
corruptions_c=motion_blur
elif [ $corruptions == density_bbox ]
then
corruptions_l=density_bbox
corruptions_c=None
elif [ $corruptions == cutout_bbox ]
then
corruptions_l=cutout_bbox
corruptions_c=None
elif [ $corruptions == gaussian_bbox_l ]
then
corruptions_l=gaussian_bbox
corruptions_c=None
elif [ $corruptions == uniform_bbox_l ]
then
corruptions_l=uniform_bbox
corruptions_c=None
elif [ $corruptions == impulse_bbox_l ]
then
corruptions_l=impulse_bbox
corruptions_c=None
elif [ $corruptions == shear ]
then
corruptions_l=shear
corruptions_c=shear
elif [ $corruptions == scale ]
then
corruptions_l=scale
corruptions_c=scale
elif [ $corruptions == rotation ]
then
corruptions_l=rotation
corruptions_c=rotation
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

for severity in $(seq 1 5); do
echo ">>>>>>>>>>>>>>>>>>>>>>" "severity" $severity
startTime=`date +"%Y-%m-%d %H:%M:%S"`

echo " python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/dist_test.py $CONFIG.py \
 --work_dir ./work_dirs/$CONFIG/eval/$corruptions_l"_"$corruptions_c"_"$severity --checkpoint $ckpt \
 --batch_size $batch_size --worker $workers --severity $severity --corruptions $corruptions_l $corruptions_c"

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/dist_test.py $CONFIG.py \
 --work_dir ./work_dirs/$CONFIG/$corruptions_l"_"$corruptions_c"_"$severity --checkpoint $ckpt \
 --batch_size $batch_size --worker $workers --severity $severity --corruptions $corruptions_l $corruptions_c

wait

endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`
sumTime=$(($et-$st))
echo "Total time is : $sumTime second."
done

done