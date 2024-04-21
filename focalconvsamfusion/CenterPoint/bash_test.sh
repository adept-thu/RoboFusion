# CONFIG="configs/nusc/voxelnet/sam_onlinev3_1_4_data"

CONFIG="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_focal_multimodal_1_4_data"
ckpt=../checkpoints/centerpoint_focal_multimodal_1_4_data.pth
EXTRA_TAG=baseline_1_4
# CONFIG="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_focal_multimodal"
NUM_GPUS=4

python ./tools/dist_test.py $CONFIG.py \
 --work_dir ./work_dirs/debug --checkpoint ../checkpoints/centerpoint_focal_multimodal_1_4_data.pth \
 --worker 1 --batch_size 1

# python ./tools/dist_test.py configs/nusc/voxelnet/sam_onlinev3_1_4_data_cor.py \
#  --work_dir ./work_dirs/$CONFIG/eval/$EXTRA_TAG --checkpoint ./work_dirs/$CONFIG/latest.pth --severity 5 --corruptions spatial_aligment None

# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/dist_test.py $CONFIG \
#  --work_dir ./work_dirs/debug --checkpoint $ckpt \
#  --worker 4 --batch_size 4

# echo "python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/dist_test.py $CONFIG \
#  --work_dir ./work_dirs/$CONFIG/eval/$EXTRA_TAG --checkpoint $ckpt \
#  --worker 1 --batch_size 1"