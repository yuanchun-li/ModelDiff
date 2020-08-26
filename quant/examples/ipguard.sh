export PYTHONPATH=..:../..:$PYTHONPATH

for SIMILAR in teacher root
do

CUDA_VISIBLE_DEVICES=$1 \
python compare.py \
--similar_mode $SIMILAR \
--cmp_mode ipguard \
--profiling_mode hybrid \
--pair_mode dataset \
--target_mode random 

done


