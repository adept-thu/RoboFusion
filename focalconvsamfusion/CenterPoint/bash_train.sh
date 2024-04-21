CONFIG="configs/nusc/voxelnet/sam_onlinev3_1_4_data"

# CONFIG="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_focal_multimodal.py"

NUM_GPUS=4

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ./tools/train.py $CONFIG.py --work_dir ./work_dirs/$CONFIG

# python tools/train.py $CONFIG.py --work_dir ./work_dirs/debug/$CONFIG