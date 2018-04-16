#!/bin/bash

#SBATCH -p short   # queue name
#SBATCH -t 0-8:00       # hours:minutes runlimit after which job will be killed.
#SBATCH -n 8      # number of cores requested
#SBATCH --mem=16G # memory requested
#SBATCH -J modeling         # Job name
#SBATCH -o %j.out       # File to which standard out will be written
#SBATCH -e %j.err       # File to which standard err will be written
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Peter_Shen@hms.harvard.edu

# use keras
source ~/keras/bin/activate

python modeling.py
