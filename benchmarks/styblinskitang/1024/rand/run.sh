#/bin/bash
#COBALT -t 1:00
#COBALT -n 1024 
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A CSC249ADOA01

echo "Starting HyperSpace run"
echo "Number of MPI ranks: $n_mpi_ranks"

export MKL_ROOT=/opt/intel/mkl
export MKL_INCLUDE=$MKL_ROOT/include
export MKL_INCLUDE=$MKL_ROOT/include
export MKL_LIBRARY=$MKL_ROOT/lib/intel64
source /opt/intel/mkl/bin/mklvars.sh intel64
source /opt/intel/bin/compilervars.sh intel64

export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=1
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_openmp_threads_per_rank=1
export n_hyperthreads_per_core=1
export n_hyperthreads_skipped_between_ranks=1

module load miniconda-3.6/conda-4.4.10
source activate /lus-projects/Candle_ECP/yngtodd/envs

aprun -n $n_mpi_ranks -N 1 python rand.py --results_dir results 
