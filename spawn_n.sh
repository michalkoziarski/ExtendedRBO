#!/bin/bash

for i in $( seq 1 $1 )
do
	sbatch script.sh runner.py
done