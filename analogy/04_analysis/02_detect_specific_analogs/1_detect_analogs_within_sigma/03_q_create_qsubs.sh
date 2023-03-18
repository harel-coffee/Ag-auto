#!/bin/bash

##PBS -l nodes=1:ppn=1,walltime=00:05:00
##PBS -l mem=1gb
##PBS -q fast

cd /home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/02_find_analogs_within_sigma
###########
########### RCP 45
###########
########### w_precip, 1_sigma

cat /home/hnoorazar/analog_codes/parameters/to_detect/NN_sigma_list | while read LINE ; do
cp 03_template.sh ./all_qsubs/q_rcp45_w_precip_1_sigma_$LINE.sh
sed -i s/precip/w_precip/g ./all_qsubs/q_rcp45_w_precip_1_sigma_$LINE.sh
sed -i s/sigma_bd/1/g ./all_qsubs/q_rcp45_w_precip_1_sigma_$LINE.sh
sed -i s/emission/rcp45/g ./all_qsubs/q_rcp45_w_precip_1_sigma_$LINE.sh
sed -i s/int_name/"$LINE"/g ./all_qsubs/q_rcp45_w_precip_1_sigma_$LINE.sh
done

########### w_precip, 2_sigma

cat /home/hnoorazar/analog_codes/parameters/to_detect/NN_sigma_list | while read LINE ; do
cp 03_template.sh ./all_qsubs/q_rcp45_w_precip_2_sigma_$LINE.sh
sed -i s/precip/w_precip/g ./all_qsubs/q_rcp45_w_precip_2_sigma_$LINE.sh
sed -i s/sigma_bd/2/g ./all_qsubs/q_rcp45_w_precip_2_sigma_$LINE.sh
sed -i s/emission/rcp45/g ./all_qsubs/q_rcp45_w_precip_2_sigma_$LINE.sh
sed -i s/int_name/"$LINE"/g ./all_qsubs/q_rcp45_w_precip_2_sigma_$LINE.sh
done

###########
########### RCP 85
########### 
########### w_precip, 1_sigma

cat /home/hnoorazar/analog_codes/parameters/to_detect/NN_sigma_list | while read LINE ; do
cp 03_template.sh ./all_qsubs/q_rcp85_w_precip_1_sigma_$LINE.sh
sed -i s/precip/w_precip/g ./all_qsubs/q_rcp85_w_precip_1_sigma_$LINE.sh
sed -i s/sigma_bd/1/g ./all_qsubs/q_rcp85_w_precip_1_sigma_$LINE.sh
sed -i s/emission/rcp85/g ./all_qsubs/q_rcp85_w_precip_1_sigma_$LINE.sh
sed -i s/int_name/"$LINE"/g ./all_qsubs/q_rcp85_w_precip_1_sigma_$LINE.sh
done

########### w_precip, 2_sigma

cat /home/hnoorazar/analog_codes/parameters/to_detect/NN_sigma_list | while read LINE ; do
cp 03_template.sh ./all_qsubs/q_rcp85_w_precip_2_sigma_$LINE.sh
sed -i s/precip/w_precip/g ./all_qsubs/q_rcp85_w_precip_2_sigma_$LINE.sh
sed -i s/sigma_bd/2/g ./all_qsubs/q_rcp85_w_precip_2_sigma_$LINE.sh
sed -i s/emission/rcp85/g ./all_qsubs/q_rcp85_w_precip_2_sigma_$LINE.sh
sed -i s/int_name/"$LINE"/g ./all_qsubs/q_rcp85_w_precip_2_sigma_$LINE.sh
done

