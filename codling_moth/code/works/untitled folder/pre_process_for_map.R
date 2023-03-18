#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)
write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]
	
file_name = paste0("combined_CMPOP_", version)
output = diapause_map_prep(input_dir, 
	                       file_name,
                           param_dir, 
                           location_group_name="LocationGroups.csv")

sub1 = data.table(output[[1]])
sub2 = data.table(output[[2]])
sub3 = data.table(output[[3]])

saveRDS(sub1, paste0(write_dir, "sub1_", version,  ".rds"))
saveRDS(sub2, paste0(write_dir, "sub2_", version, ".rds"))
saveRDS(sub3, paste0(write_dir, "sub3_", version, ".rds"))