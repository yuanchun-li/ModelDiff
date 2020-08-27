export PYTHONPATH=..:../..:$PYTHONPATH


for SIMILAR in teacher root 
do

CUDA_VISIBLE_DEVICES=$1 \
python compare.py \
--similar_mode $SIMILAR \
--cmp_mode weight \
--profiling_mode normal \
--pair_mode arch \

done


