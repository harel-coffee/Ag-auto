#!/bin/bash

cd /home/hnoorazar/analog_codes/04_analysis/

########### RCP 45

###### w_precip, w_gen3

for runname in 2026_2050 2051_2075 2076_2095
do
cp template.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_$runname.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_$runname.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_$runname.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_$runname.sh
sed -i s/time_type/"$runname"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_$runname.sh
done

###### no_precip, w_gen3

for runname in 2026_2050 2051_2075 2076_2095
do
cp template.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_$runname.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_$runname.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_$runname.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_$runname.sh
sed -i s/time_type/"$runname"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_$runname.sh
done


###### w_precip, no_gen3

# for runname in 2026_2050 2051_2075 2076_2095
# do
# cp template.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_$runname.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_$runname.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_$runname.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_$runname.sh
# sed -i s/time_type/"$runname"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_$runname.sh
# done

# ###### no_precip, no_gen3

# for runname in 2026_2050 2051_2075 2076_2095
# do
# cp template.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_$runname.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_$runname.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_$runname.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_$runname.sh
# sed -i s/time_type/"$runname"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_$runname.sh
# done

###########
########### RCP 85
########### 
###### w_precip, w_gen3

for runname in 2026_2050 2051_2075 2076_2095
do
cp template.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_$runname.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_$runname.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_$runname.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_$runname.sh
sed -i s/time_type/"$runname"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_$runname.sh
done

###### no_precip, w_gen3

for runname in 2026_2050 2051_2075 2076_2095
do
cp template.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_$runname.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_$runname.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_$runname.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_$runname.sh
sed -i s/time_type/"$runname"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_$runname.sh
done


###### w_precip, no_gen3

# for runname in 2026_2050 2051_2075 2076_2095
# do
# cp template.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_$runname.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_$runname.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_$runname.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_$runname.sh
# sed -i s/time_type/"$runname"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_$runname.sh
# done

# ###### no_precip, no_gen3

# for runname in 2026_2050 2051_2075 2076_2095
# do
# cp template.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_$runname.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_$runname.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_$runname.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_$runname.sh
# sed -i s/time_type/"$runname"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_$runname.sh
# done


