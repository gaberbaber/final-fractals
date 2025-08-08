#!/bin/bash
# This will run on the GPU node
cd $PBS_O_WORKDIR
# module load cuda/12.6
./dla_gpu
