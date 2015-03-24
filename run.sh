#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/ucbtle1/code-packages/abc-sysbio-code/:/home/ucbtle1/code-packages/cuda-sim-code/:/home/ucbtle1/code-packages/sympy-0.7.3
exe=/home/ucbtle1/code-packages/abc-sysbio-code/scripts/run-abc-sysbio

export CUDA_DEVICE=5

python read_input.py; python my_abc.py
