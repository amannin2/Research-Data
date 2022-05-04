#!/bin/bash
#$ -N min_c10tfsi
#$ -M amannin2@nd.edu
#$ -m abe
#$ -q long
#$ -pe smp 24

module load gromacs/2018.3
export OMP_NUM_THREADS=24

gmx mdrun -v -deffnm min
