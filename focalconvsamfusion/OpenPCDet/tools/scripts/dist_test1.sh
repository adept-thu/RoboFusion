#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=25911 test.py --launcher pytorch ${PY_ARGS}

