#!/bin/bash

cd /home/h.noorazar/NASA/Kamiak_ML_Oct17/qsubs

for runname in {1..25}
do
sbatch ./KNN_Oct17_accuracyScoring$runname.sh
done
