#!/bin/sh
#SBATCH --job-name=ML_NH3_1C_Angew-1+H_minimized_optimized_r3
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00
#SBATCH -A standby
#SBATCH -o ML_NH3_1C_Angew-1+H_minimized_optimized_r3.out
#SBATCH -e ML_NH3_1C_Angew-1+H_minimized_optimized_r3.err

# Load default LAMMPS, compiler, and openmpi packages
module --force purge
module load gcc/12.2.0
module load openmpi/4.1.4
module load lammps/20220623
echo Running on host `hostname`
echo Time is `date`
t_start=$SECONDS

# Submiting LAMMPS job for eval.in.init
mpirun -np $SLURM_NTASKS lmp -in eval.in.init >> LAMMPS_run.out 
kill -s INT $GPU_PID $CPU_PID

t_end=$SECONDS
t_diff=$(( ${t_end} - ${t_start} ))
eval "echo $(date -ud "@$t_diff" +'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.')"
echo Completion time is `date`
