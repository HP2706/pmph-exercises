#!/bin/bash

# Validates all 3 predicates

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    BACKEND="cuda"
else
    BACKEND="c"
fi

echo "Using backend: $BACKEND"

echo "Validating lssp-zeros.fut"
futhark test --backend=$BACKEND lssp-zeros.fut
echo "Validating lssp-sorted.fut"
futhark test --backend=$BACKEND lssp-sorted.fut
echo "Validating lssp-same.fut"
futhark test --backend=$BACKEND lssp-same.fut
