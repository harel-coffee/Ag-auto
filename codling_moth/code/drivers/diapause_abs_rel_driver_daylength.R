#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/diap_daylength"
input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]
	
file_name = paste0("combined_CMPOP_", version)
output = diapause_abs_rel_daylength(input_dir, 
	                                file_name,
                                    param_dir, 
                                    location_group_name="LocationGroups.csv")

RelData = data.table(output[[1]])
AbsData = data.table(output[[2]])
sub1 = data.table(output[[3]])
# pre_diap_plot = data.table(output[[4]])

saveRDS(RelData, paste0(write_dir, "diapause_rel_", version, ".rds"))
saveRDS(AbsData, paste0(write_dir, "diapause_abs_", version, ".rds"))
saveRDS(sub1,    paste0(write_dir, "diapause_plot_", version, ".rds"))
# saveRDS(pre_diap_plot, paste0(write_dir, "pre_diap_plot_", version, ".rds"))
