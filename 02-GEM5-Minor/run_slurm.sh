#!/bin/sh
#SBATCH --job-name=gem5_simulation    
#SBATCH --output=gem5_log.txt     
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --reservation=fri
#SBATCH --time=00:20:00

GEM5_WORKSPACE=/d/hpc/projects/FRI/GEM5/gem5_workspace
GEM5_ROOT=$GEM5_WORKSPACE/gem5
GEM_PATH=$GEM5_ROOT/build/RISCV

srun apptainer exec $GEM5_WORKSPACE/gem5_rv.sif $GEM_PATH/gem5.opt --outdir=cpu2_stats cpu_benchmark.py


#apptainer exec $GEM5_WORKSPACE/gem5_rv.sif python3 $GEM5_ROOT/util/minorview.py --start-time 0 --end-time 100 trace.out