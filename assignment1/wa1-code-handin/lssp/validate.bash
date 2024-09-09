#!/bin/bash

# Validates all 3 predicates

module load futhark
module load cuda

# Parse command-line arguments
NOCUDA=false
for arg in "$@"; do
    case $arg in
        -nocuda=True)
        NOCUDA=true
        shift # Remove this argument from processing
        ;;
    esac
done

# Check if CUDA is available and not disabled
if [ "$NOCUDA" = false ] && command -v nvidia-smi &> /dev/null; then
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

    echo "----------------------------------------"
done
