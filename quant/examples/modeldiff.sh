export PYTHONPATH=..:$PYTHONPATH


for TARGET in random
do
    for SIMILAR in teacher root
    do

    CUDA_VISIBLE_DEVICES=$1 \
    python compare.py \
    --similar_mode $SIMILAR \
    --cmp_mode ddv \
    --target_mode $TARGET \

    done
done
