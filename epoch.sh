#!/bin/bash

# Define help message
help="\nUSEAGE:\n
This script takes 1 required and 2 optional argument:
1. Required: Directory where input.deck file is.
2. Optional: Number of processors to use.
2. Optional: log result or not (if 3rd arg == log).
Type ./epoch.sh to show this help."





if [[ -z $1 ]]; then
    echo -e "$help"
	echo 
    exit
else
	dir=$1
fi



# Default to 2 processors if arg not specified
if [[ -z $2 ]]; then
	np=2
else
	np=$2
fi


if [[ $3 == 'log' ]]; then
    export OMP_NUM_THREADS=1
    # nohup echo ${dir} | mpiexec -n ${np} ./bin/epoch1d > run.log &
    # cat nohup.out
    echo ${dir} | mpiexec -n ${np} ./bin/epoch1d > run.log

else
    export OMP_NUM_THREADS=1
    # nohup echo ${dir} | mpiexec -n ${np} ./bin/epoch1d &
    # cat nohup.out
    echo ${dir} | mpiexec -n ${np} ./bin/epoch1d
fi

