#!/bin/bash

##PBS -l nodes=1:ppn=1,walltime=00:05:00
##PBS -l mem=1gb
##PBS -q fast

cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg
qsub 03_q_create_qsubs_loc_bcc.sh
qsub 03_q_create_qsubs_loc_BNU.sh
qsub 03_q_create_qsubs_loc_Can.sh
qsub 03_q_create_qsubs_loc_CNRM.sh
qsub 03_q_create_qsubs_loc_GFDLG.sh
qsub 03_q_create_qsubs_loc_GFDLM.sh
