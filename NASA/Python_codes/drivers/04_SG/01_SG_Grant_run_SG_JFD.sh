#!/bin/bash

cd /home/hnoorazar/NASA/04_SG/qsubs/
for runname in {1..4}
do
qsub ./q_Grant_JFD$runname.sh
done
