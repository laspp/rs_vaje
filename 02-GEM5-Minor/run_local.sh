#!/bin/sh
#SBATCH --job-name=gem5_simulation    
#SBATCH --output=gem5_log.txt     
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM_PATH=$GEM5_ROOT/build/RISCV


apptainer exec $GEM5_WORKSPACE/gem5_rv.sif  $GEM_PATH/gem5.opt --debug-flags=MinorTrace --debug-file=trace.out --outdir=hello_log cpu_benchmark.py

