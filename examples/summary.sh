for TARGET in random
do
    for SIMILAR in root
    do

    python summary.py \
    --similar_mode $SIMILAR \
    --cmp_mode ddv \
    --target_mode $TARGET \
    --profiling_mode other_domain

    done
done