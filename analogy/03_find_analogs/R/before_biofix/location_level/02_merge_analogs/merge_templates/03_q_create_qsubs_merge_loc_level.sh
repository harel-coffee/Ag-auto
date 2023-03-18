#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=2:00:00
#PBS -l mem=2gb
##PBS -q hydro

cd /home/hnoorazar/analog_codes/03_find_analogs/location_level/merge
###########
########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_bcc-csm1-1-m.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_bcc-csm1-1-m.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_bcc-csm1-1-m.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_bcc-csm1-1-m.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_bcc-csm1-1-m.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_bcc-csm1-1-m.sh
sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_bcc-csm1-1-m.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_bcc-csm1-1-m.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_bcc-csm1-1-m.sh
# sed -i s/model_type/"bcc-csm1-1-m"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_bcc-csm1-1-m.sh




############################################ BNU 
########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_BNU-ESM.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_BNU-ESM.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_BNU-ESM.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_BNU-ESM.sh
sed -i s/model_type/"BNU-ESM"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_BNU-ESM.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_BNU-ESM.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_BNU-ESM.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_BNU-ESM.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_BNU-ESM.sh
sed -i s/model_type/"BNU-ESM"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_BNU-ESM.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/model_type/"BNU-ESM"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_BNU-ESM.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/model_type/"BNU-ESM"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_BNU-ESM.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_BNU-ESM.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_BNU-ESM.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_BNU-ESM.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_BNU-ESM.sh
sed -i s/model_type/"BNU-ESM"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_BNU-ESM.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_BNU-ESM.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_BNU-ESM.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_BNU-ESM.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_BNU-ESM.sh
sed -i s/model_type/"BNU-ESM"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_BNU-ESM.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_BNU-ESM.sh
# sed -i s/model_type/"BNU-ESM"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_BNU-ESM.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_BNU-ESM.sh
# sed -i s/model_type/"BNU-ESM"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_BNU-ESM.sh


####################################################### CanESM2
########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CanESM2.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CanESM2.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CanESM2.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CanESM2.sh
sed -i s/model_type/"CanESM2"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CanESM2.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CanESM2.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CanESM2.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CanESM2.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CanESM2.sh
sed -i s/model_type/"CanESM2"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CanESM2.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CanESM2.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CanESM2.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CanESM2.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CanESM2.sh
# sed -i s/model_type/"CanESM2"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CanESM2.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CanESM2.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CanESM2.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CanESM2.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CanESM2.sh
# sed -i s/model_type/"CanESM2"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CanESM2.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CanESM2.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CanESM2.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CanESM2.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CanESM2.sh
sed -i s/model_type/"CanESM2"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CanESM2.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CanESM2.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CanESM2.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CanESM2.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CanESM2.sh
sed -i s/model_type/"CanESM2"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CanESM2.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CanESM2.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CanESM2.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CanESM2.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CanESM2.sh
# sed -i s/model_type/"CanESM2"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CanESM2.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CanESM2.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CanESM2.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CanESM2.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CanESM2.sh
# sed -i s/model_type/"CanESM2"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CanESM2.sh


########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$              CNRM-CM5
########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/model_type/"CNRM-CM5"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_CNRM-CM5.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/model_type/"CNRM-CM5"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_CNRM-CM5.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/model_type/"CNRM-CM5"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_CNRM-CM5.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/model_type/"CNRM-CM5"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_CNRM-CM5.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CNRM-CM5.sh
sed -i s/model_type/"CNRM-CM5"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_CNRM-CM5.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CNRM-CM5.sh
sed -i s/model_type/"CNRM-CM5"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_CNRM-CM5.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CNRM-CM5.sh
# sed -i s/model_type/"CNRM-CM5"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_CNRM-CM5.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CNRM-CM5.sh
# sed -i s/model_type/"CNRM-CM5"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_CNRM-CM5.sh



###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ GFDL-ESM2G

########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/model_type/"GFDL-ESM2G"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2G.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/model_type/"GFDL-ESM2G"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2G.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/model_type/"GFDL-ESM2G"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2G.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/model_type/"GFDL-ESM2G"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2G.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/model_type/"GFDL-ESM2G"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2G.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2G.sh
sed -i s/model_type/"GFDL-ESM2G"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2G.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/model_type/"GFDL-ESM2G"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2G.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2G.sh
# sed -i s/model_type/"GFDL-ESM2G"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2G.sh



###########$$$$$$$$$$$$$$$$$$$$$$$       GFDL-ESM2M

########### RCP 45
###########
########### w_precip, w_gen3
cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/model_type/"GFDL-ESM2M"/g ./rcp45_qsubs/q_rcp45_w_precip_w_gen3_GFDL-ESM2M.sh


########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/gen_3_type/w_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/model_type/"GFDL-ESM2M"/g ./rcp45_qsubs/q_rcp45_no_precip_w_gen3_GFDL-ESM2M.sh

########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/precip_type/w_precip/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/model_type/"GFDL-ESM2M"/g ./rcp45_qsubs/q_rcp45_w_precip_no_gen3_GFDL-ESM2M.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/precip_type/no_precip/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/emission_type/rcp45/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/model_type/"GFDL-ESM2M"/g ./rcp45_qsubs/q_rcp45_no_precip_no_gen3_GFDL-ESM2M.sh

###########
########### RCP 85
########### 
########### w_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/model_type/"GFDL-ESM2M"/g ./rcp85_qsubs/q_rcp85_w_precip_w_gen3_GFDL-ESM2M.sh

########### no_precip, w_gen3

cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/gen_3_type/w_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2M.sh
sed -i s/model_type/"GFDL-ESM2M"/g ./rcp85_qsubs/q_rcp85_no_precip_w_gen3_GFDL-ESM2M.sh

# ########### w_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/precip_type/w_precip/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/model_type/"GFDL-ESM2M"/g ./rcp85_qsubs/q_rcp85_w_precip_no_gen3_GFDL-ESM2M.sh

# ########### no_precip, no_gen3

# cp 03_template_merge_loc.sh ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/precip_type/no_precip/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/gen_3_type/no_gen3/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/emission_type/rcp85/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2M.sh
# sed -i s/model_type/"GFDL-ESM2M"/g ./rcp85_qsubs/q_rcp85_no_precip_no_gen3_GFDL-ESM2M.sh



