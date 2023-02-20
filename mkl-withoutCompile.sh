#!/bin/bash -l

#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -p course -t 3 

# load modules
module load compilers/intel

# you should ensure you compiled prior to using this batch script
echo Timestamp of a.out
ls -lt a.out 

echo Matrices -O0
icc -qopenmp -O0 ./matrices.c
maxThreads=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
echo running on 1 up to $maxThreads threads
for k in `seq 1 $maxThreads`; do 
      export OMP_NUM_THREADS=$k
      echo using ${OMP_NUM_THREADS} OpenMP threads
      ./a.out
done

echo Matrices -O3
icc -qopenmp -O3 ./matrices.c
maxThreads=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
echo running on 1 up to $maxThreads threads
for k in `seq 1 $maxThreads`; do 
      export OMP_NUM_THREADS=$k
      echo using ${OMP_NUM_THREADS} OpenMP threads
      ./a.out
done

echo Matrices MKL -O0
icc -qopenmp -O0 –mkl=sequential ./matrices-mkl.c
maxThreads=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
echo running on 1 up to $maxThreads threads
for k in `seq 1 $maxThreads`; do 
      export OMP_NUM_THREADS=$k
      echo using ${OMP_NUM_THREADS} OpenMP threads
      ./a.out
done

echo Matrices MKL -O3
icc -qopenmp -O3 –mkl=sequential ./matrices-mkl.c
maxThreads=${SLURM_CPUS_PER_TASK:-1} # if '-c' not used then default to 1
echo running on 1 up to $maxThreads threads
for k in `seq 1 $maxThreads`; do 
      export OMP_NUM_THREADS=$k
      echo using ${OMP_NUM_THREADS} OpenMP threads
      ./a.out
done
