
# for TARGET in targeted untargeted 
# do
#     for SIMILAR in root teacher
#     do

#     CUDA_VISIBLE_DEVICES=$1 \
#     python compare.py \
#     --similar_mode $SIMILAR \
#     --cmp_mode ddm \
#     --target_mode $TARGET \

#     done
# done


for TARGET in targeted_nobound 
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