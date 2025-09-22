#!/bin/bash
#$ -pe smp 64
#$ -q long
#$ -N run1
#$ -o run1.out

module load lammps

t_start=$SECONDS

# Submiting LAMMPS job for plumed.in.init
mpirun -np $NSLOTS lmp -in plumed.in.init > 4P_steer.log &
wait

t_end=$SECONDS
t_diff=$(( ${t_end} - ${t_start} ))
eval "echo $(date -ud "@$t_diff" +'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.')"
echo Completion time is `date`
