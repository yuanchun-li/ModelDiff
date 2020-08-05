#!/bin/bash


iter=90000
id=0
splmda=0
layer=5
lr=1e-2
wd=5e-3
mmt=0.9
lmda=5e0

DATASETS=(MIT_67 Flower_102 stanford_dog)
DATASET_NAMES=(MIT67Data Flower102Data SDog120Data)
DATASET_ABBRS=(mit67 flower102 sdog120)

for i in 0
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/distill/feat
    NAME=mbnetv2_${DATASET_ABBR}_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    TEACHER_DIR=results/finetune/layer1/mbnetv2_${DATASET_ABBR}_lr5e-3_iter30000_wd1e-4_mmt0_1

    CUDA_VISIBLE_DEVICES=$1 \
    python -u finetune.py \
    --iterations ${iter} \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name ${NAME} \
    --batch_size 64 \
    --feat_lmda ${lmda} \
    --lr ${lr} \
    --network mbnetv2 \
    --weight_decay ${wd} \
    --beta 1e-2 \
    --test_interval 1000 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --reinit \
    --output_dir ${DIR} \
    # &


done