#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=10000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT67 Flower102 SDog120)
DATASET_NAMES=(MIT67 Flower102 SDog120)
DATASET_ABBRS=(MIT67 Flower102 SDog120)


for i in 0 
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    CKPT_DIR=results/finetune/conv1/resnet18_${DATASET_ABBR}_lr5e-3_iter30000_wd1e-4_mmt0_1

    python -u load_model.py \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --network resnet18 \
    --checkpoint ${CKPT_DIR}/ckpt.pth \
    # &


done