#!/bin/sh

for units in 32 64 128
do
    for batch_size in 4 8 16 32 64 128 256
    do
        echo "********** units:$units, batch_size:$batch_size"
        eval "CUDA_VISIBLE_DEVICES=-1 python predict_batch.py --iterations 500 --inputs 784 --units $units --outputs 784 --batch_size $batch_size"
    done
done
