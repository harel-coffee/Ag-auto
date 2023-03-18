

library(data.table)
library(dplyr)
#########################
######################### Directories
#########################

data_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/00_databases/usa/"
analog_param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

######################### Reading

bad_CMPOP <- data.table(readRDS(paste0(data_dir, "CMPOP_location_accDDinC_4_biofix.rds")))

biofix_param <- data.table(read.csv(paste0(analog_param_dir, "biofix_param_hi.csv"), 
                                    header=T, as.is=T))

good_CMPOP <- apply_bio_fix_to_CMPOP(bad_CMPOP, biofix_param)