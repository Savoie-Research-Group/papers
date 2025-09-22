#!/bin/sh
#SBATCH --job-name=300K_run1
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 8:00:00
#SBATCH -A bsavoie
#SBATCH -o 300K_run1.out
#SBATCH -e 300K_run1.err

# Load default LAMMPS, compiler, and openmpi packages
module load gcc/12.2.0 
module load openmpi/4.1.4 
module load lammps/20220623 

# cd into the submission directory
echo Running on host `hostname`
echo Time is `date`
t_start=$SECONDS

# Submiting LAMMPS job for eval.in.init
mpirun -np 128 lmp -in eval.in.init >> LAMMPS_run.out &
wait

t_end=$SECONDS
t_diff=$(( ${t_end} - ${t_start} ))
eval "echo $(date -ud "@$t_diff" +'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.')"
echo Completion time is `date`
