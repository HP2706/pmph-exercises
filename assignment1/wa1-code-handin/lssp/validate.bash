#!/bin/bash

# Validates all 3 predicates
module load cuda
module load futhark
# Check if CUDA is available


if command -v nvidia-smi &> /dev/null; then
    BACKENDS=("cuda" "c")
else
    BACKENDS=("c")
fi

for BACKEND in "${BACKENDS[@]}"; do
    echo "Using backend: $BACKEND"

    echo "Validating lssp-zeros.fut"
    futhark test --backend=$BACKEND lssp-zeros.fut
    echo "Validating lssp-sorted.fut"
    futhark test --backend=$BACKEND lssp-sorted.fut
    echo "Validating lssp-same.fut"
    futhark test --backend=$BACKEND lssp-same.fut
done