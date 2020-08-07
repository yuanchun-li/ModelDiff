#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
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

    NAME=vgg16_${DATASET_ABBR}_lr${lr}_iter${iter}_wd${wd}_mmt${mmt}_${id}
    DIR=results/finetune/vgg16/feat0

    CUDA_VISIBLE_DEVICES=$1 \
    python -u finetune.py \
    --iterations ${iter} \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name ${NAME} \
    --batch_size 32 \
    --feat_lmda ${lmda} \
    --lr ${lr} \
    --network vgg16_bn \
    --weight_decay ${wd} \
    --beta 1e-2 \
    --test_interval 1000 \
    --adv_test_interval -1 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --output_dir ${DIR} \
    --ft_begin_module features.0 \
    # &

done