#!/bin/sh

epochs=10
activation="sigmoid"
loss="mean_squared_error"
out="result.txt"

## mnist
for units in 512 1024 2048
do
    for batch_size in 8 16 32
    do
        command="python train.py --dataset mnist --epochs ${epochs} --units ${units} --batch_size ${batch_size} --activation ${activation} --loss ${loss} >> ${out}"
        eval $command
    done
done

## fashion mnist
for units in 512 1024 2048
do
    for batch_size in 8 16 32
    do
        command="python train.py --dataset fashion --epochs ${epochs} --units ${units} --batch_size ${batch_size} --activation ${activation} --loss ${loss} >> ${out}"
        eval $command
    done
done

## digits
for units in 128 256 512
do
    for batch_size in 8 16 32
    do
        command="python train.py --dataset digits --epochs ${epochs} --units ${units} --batch_size ${batch_size} --activation ${activation} --loss ${loss} >> ${out}"
        eval $command
    done
done

## boston
for units in 16 32 64
do
    for batch_size in 8 16 32
    do
        command="python train.py --dataset boston --epochs ${epochs} --units ${units} --batch_size ${batch_size} --activation ${activation} --loss ${loss} >> ${out}"
        eval $command
    done
done
