

for TARGET in targeted
do
    for SIMILAR in teacher 
    do

    CUDA_VISIBLE_DEVICES=$1 \
    python compare.py \
    --similar_mode $SIMILAR \
    --cmp_mode ddv \
    --target_mode $TARGET \

    done
done