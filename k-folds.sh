#!/bin/bash -l

#SBATCH -p nodes
#SBATCH -t 20
#SBATCH -D ./
#SBATCH --export=ALL

module load intel/oneapi-hpc-toolkit-2025.1.3.10 
module load mpi/2021.15 

procs=${SLURM_NTASKS:-1}
cores=${SLURM_CPUS_PER_TASK:-1}

export OMP_NUM_THREADS=$cores

k=3
folds=10

make all

echo "Num_Threads = $OMP_NUM_THREADS"
echo "Num_Procs = $procs"

echo 
echo
echo =====RUNNING PROGRAMS=====

mpirun -np $procs ./k-folds-complete-gcc asteroids.csv "output_$k_$folds.csv" $k $folds

make clear