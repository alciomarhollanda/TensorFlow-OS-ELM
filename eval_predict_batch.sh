#!/bin/sh

for units in 8 16 32
do
    for batch_size in 4 8 16 32 64 128 256
    do
        echo "********** units:$units, batch_size:$batch_size"
        eval "CUDA_VISIBLE_DEVICES=-1 python predict_batch.py --iterations 100 --inputs 64 --units $units --outputs 64 --batch_size $batch_size"
    done
done
