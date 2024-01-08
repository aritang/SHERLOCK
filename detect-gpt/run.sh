#!/bin/bash
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -o %J_stdout.log              # Direct the standard output to a file named "<jobID>_stdout.log"
#BSUB -e %J_stderr.log              # Direct the standard error to a file named "<jobID>_stderr.log"
#BSUB -q gpu                        # Specify the queue name if required, here assuming a 'gpu' queue
#BSUB -J T5LARGE_OPT150_VENTI
nvidia-smi >> out

# Define the base directory as the current working directory

# Replace these with your actual proxy details

BASEDIR="$(pwd)"


cd $BASEDIR

echo $BASEDIR

# Your Python command
python run_my.py

