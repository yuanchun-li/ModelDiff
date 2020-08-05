#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
id=1
splmda=0
lmda=0
layer=12345
lr=5e-3
wd=1e-4
mmt=0


DATASETS=(MIT_67 Flower_102 stanford_dog)
DATASET_NAMES=(MIT67Data Flower102Data SDog120Data)
DATASET_ABBRS=(mit67 flower102 sdog120)

for i in 0 
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=mbnetv2_${DATASET_ABBR}_lr${lr}_iter${iter}_wd${wd}_mmt${mmt}_${id}
    DIR=results/finetune/mbnetv2/layer4

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
    --adv_test_interval -1 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --output_dir ${DIR} \
    --ft_begin_module layer4 \
    # &

done