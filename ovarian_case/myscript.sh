#!/bin/bash
#
# At first lets give job some descriptive name to distinct
# from other jobs running on cluster. Not that %a will have the same
# value as $SLURM_ARRAY_TASK_ID later.
#
#SBATCH -J OCaucTest
#
# Lets redirect jobs out some other file than default slurm-%jobid-out
#SBATCH --output=log/aucTest-%a.log
#SBATCH --error=err/aucTest-%a.err
#
# Well want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# Well want to reserve 2GB/8GB memory for the job
# and 3 days of compute time to finish.
#
#SBATCH --mem=8192
#SBATCH --time=3-00
#SBATCH --partition=sgn,normal,gpu,bigmem,parallel 
#
# These commands will be executed on the compute node:

module load matlab
matlab -singleCompThread -nojvm -nodisplay -nosplash -r "iteration($SLURM_ARRAY_TASK_ID)"