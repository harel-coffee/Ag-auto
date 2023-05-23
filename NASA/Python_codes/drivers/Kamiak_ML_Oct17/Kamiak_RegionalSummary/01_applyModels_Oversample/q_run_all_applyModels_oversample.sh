#!/bin/bash

cd /home/h.noorazar/NASA/regionalStat/01_applyModels_Oversample/qsubs

for runname in {1..48} ###192
do
sbatch ./applyModels_oversample$runname.sh
done
