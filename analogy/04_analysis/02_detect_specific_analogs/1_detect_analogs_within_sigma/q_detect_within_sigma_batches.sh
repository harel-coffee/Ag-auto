#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=05:00:00
#PBS -l mem=1gb
#PBS -q fast

cd /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/02_find_analogs_within_sigma/all_qsubs
cat /home/hnoorazar/analog_codes/parameters/to_detect/batch_1 | while read LINE ; do
qsub $LINE
done
