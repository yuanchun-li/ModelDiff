
for CMP in ddv 
do
    for SIMILAR in teacher 
    do

    CUDA_VISIBLE_DEVICES=$1 \
    python compare.py \
    --similar_mode $SIMILAR \
    --cmp_mode $CMP \
    --profiling_mode multi_model \

    done
done

