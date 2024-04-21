#####################################
#               选择模型
#####################################
cfg_file=work_dirs/transfusion_nusc_voxel_LC_fastsam_wave/transfusion_nusc_voxel_LC_fastsam_wave.py
ckpt=work_dirs/transfusion_nusc_voxel_LC_fastsam_wave/epoch_6.pth

# corruptions
weather=(snow rain fog sunlight)
sensor=(density cutout crosstalk gaussian_l uniform_l impulse_l gaussian_c uniform_c impulse_c)
motion=(compensation moving_bbox motion_blur)
object=(density_bbox cutout_bbox gaussian_bbox_l uniform_bbox_l impulse_bbox_l shear scale rotation)
alignment=(spatial_aligment temporal_aligment)
all=(${weather[@]} ${sensor[@]} ${motion[@]} ${object[@]} ${alignment[@]})

# 执行单个噪声
cor_list=(fog)
# 执行多个噪声
# cor_list=(gaussian_c uniform_c impulse_c)
# cor_list 对应 62行 for corruptions in ${cor_list[@]}

#####################################
#          缩短推理时间的参数
# 
#####################################
workers=4
batch_size=16

singleGPU=False # 单卡/多卡推理
NUMGPU=2

# 是否执行不加噪声的推理
clean_flag=False


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=3
# rm -rf ../output/*


if [ $clean_flag == True ]
then
echo ">>>>>>>>>>>>>>>>>>>>>>" "corruptions None"
echo ">>>>>>>>>>>>>>>>>>>>>>" "Lidar None"
echo ">>>>>>>>>>>>>>>>>>>>>>" "Camera None"

echo ">>>>>>>>>>>>>>>>>>>>>>" "severity 0"
echo "python test.py --cfg_file $cfg_file --workers $workers --batch_size $batch_size --ckpt $ckpt --corruptions None None --severity 0 --extra_tag None &"
python test.py --cfg_file $cfg_file --workers $workers --batch_size $batch_size --ckpt $ckpt --corruptions None None --severity 0 --extra_tag None &
wait
fi

for corruptions in ${cor_list[@]} # 遍历指定子集
# for corruptions in ${all[@]} # 遍历全部
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

for severity in $(seq 5 5); do
echo ">>>>>>>>>>>>>>>>>>>>>>" "severity" $severity
startTime=`date +"%Y-%m-%d %H:%M:%S"`

#echo "torchpack dist-run -np $workers python tools/test.py $cfg_file $ckpt --eval bbox --corruptions $corruptions_l $corruptions_c --severity $severity &"
./tools/dist_test2.sh $cfg_file $ckpt 4 --eval bbox --corruptions $corruptions_l $corruptions_c --severity $severity
#torchpack dist-run -np $workers python tools/test.py $cfg_file $ckpt --eval bbox --corruptions $corruptions_l $corruptions_c --severity $severity --save_cor_flag&
wait


endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`
sumTime=$(($et-$st))
echo "Total time is : $sumTime second."
done

done

python tools/corruptions_pp.py