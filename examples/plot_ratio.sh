

for TARGET in targeted 
do
    for SIMILAR in root
    do

    python plot_ratio.py \
    --similar_mode $SIMILAR \
    --cmp_mode ddv \
    --target_mode $TARGET \
    --profiling_mode hybrid \
    # &

    done
done