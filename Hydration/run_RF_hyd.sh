#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single node
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (max. of 16)
#SBATCH --ntasks=16
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=48:00:00
#
# Job name
#SBATCH --job-name=rfregession
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

# load anaconda 5.2 (Python 3.6.5, numpy, cython ...)
module load anaconda/python-3.6.8/2019.03

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh 
#----------------------------------------------------------

# Modify the line below to run your program
python RF_09.03_hyd.py

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh 
#----------------------------------------------------------
