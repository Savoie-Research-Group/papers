#!/bin/sh
#SBATCH --job-name=4P_relax_300K_run1
#SBATCH -N 2
#SBATCH -n 256
#SBATCH -t 4:00:00
#SBATCH -A standby
#SBATCH -o 4P_relax_300K_run1.out
#SBATCH -e 4P_relax_300K_run1.err

# Load default LAMMPS, compiler, and openmpi packages
module load gcc/12.2.0 
module load openmpi/4.1.4 
module load lammps/20220623 

# cd into the submission directory
echo Running on host `hostname`
echo Time is `date`
t_start=$SECONDS

# Submiting LAMMPS job for eval.in.init
cd .
mpirun -np $SLURM_NTASKS lmp -in eval.in.init >> 4P_relax.log &
wait

t_end=$SECONDS
t_diff=$(( ${t_end} - ${t_start} ))
eval "echo $(date -ud "@$t_diff" +'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.')"
echo Completion time is `date`
