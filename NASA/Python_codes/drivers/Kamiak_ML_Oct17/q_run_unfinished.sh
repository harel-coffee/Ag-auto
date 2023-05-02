#!/bin/bash

cd /home/h.noorazar/NASA/Kamiak_ML_Oct17/qsubs

for runname in 5 6 10 11 12 23 24
do
sbatch ./KNN_Oct17_accuracyScoring$runname.sh
done
