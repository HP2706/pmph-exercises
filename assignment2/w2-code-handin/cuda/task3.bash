#! /bin/bash


module load cuda

nvcc pbb_main.cu -o pbb_main


arrays=(10003565 1003565 103565 13565)

for array in ${arrays[@]}; do
    echo "Running PBB for array size $array with 1024 threads"
    ./pbb_main $array 1024 >> pbb_results.txt
done