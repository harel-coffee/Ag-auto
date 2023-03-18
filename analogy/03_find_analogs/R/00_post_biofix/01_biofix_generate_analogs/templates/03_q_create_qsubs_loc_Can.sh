#!/bin/bash

##PBS -l nodes=1:ppn=1,walltime=00:05:00
##PBS -l mem=1gb
##PBS -q fast

cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs
###########
########### RCP 45
###########
########### w_precip, no_gen3

cat /home/hnoorazar/analog_codes/parameters/file_names | while read LINE ; do
cp 03_template_loc_Can.sh ./rcp45_qsubs/Can/q_rcp45_w_precip_$LINE.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/Can/q_rcp45_w_precip_$LINE.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/Can/q_rcp45_w_precip_$LINE.sh
sed -i s/int_file/"$LINE"/g ./rcp45_qsubs/Can/q_rcp45_w_precip_$LINE.sh
done

# ########### no_precip, no_gen3

cat /home/hnoorazar/analog_codes/parameters/file_names | while read LINE ; do
cp 03_template_loc_Can.sh ./rcp45_qsubs/Can/q_rcp45_no_precip_$LINE.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/Can/q_rcp45_no_precip_$LINE.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/Can/q_rcp45_no_precip_$LINE.sh
sed -i s/int_file/"$LINE"/g ./rcp45_qsubs/Can/q_rcp45_no_precip_$LINE.sh
done

###########
########### RCP 85
###########
########## w_precip, no_gen3

cat /home/hnoorazar/analog_codes/parameters/file_names | while read LINE ; do
cp 03_template_loc_Can.sh ./rcp85_qsubs/Can/q_rcp85_w_precip_$LINE.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/Can/q_rcp85_w_precip_$LINE.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/Can/q_rcp85_w_precip_$LINE.sh
sed -i s/int_file/"$LINE"/g ./rcp85_qsubs/Can/q_rcp85_w_precip_$LINE.sh
done

# ########### no_precip, no_gen3

cat /home/hnoorazar/analog_codes/parameters/file_names | while read LINE ; do
cp 03_template_loc_Can.sh ./rcp85_qsubs/Can/q_rcp85_no_precip_$LINE.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/Can/q_rcp85_no_precip_$LINE.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/Can/q_rcp85_no_precip_$LINE.sh
sed -i s/int_file/"$LINE"/g ./rcp85_qsubs/Can/q_rcp85_no_precip_$LINE.sh
done



