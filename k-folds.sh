#!/bin/bash -l

#SBATCH -p nodes
#SBATCH -t 20
#SBATCH -D ./
#SBATCH --export=ALL

module load compilers/intel/2019u5
module load mpi/intel-mpi/2019u5/bin

procs=${SLURM_NTASKS:-1}
cores=${SLURM_CPUS_PER_TASK:-1}

export OMP_NUM_THREADS=$cores

k=3
folds=10

make all

echo 
echo
echo =====RUNNING PROGRAMS=====

mpirun -np $procs ./k-folds-complete-icc asteroids.csv "output_$k_$folds.csv" $k $folds

make clear