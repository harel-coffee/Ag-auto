#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/diapause_sensitivity/shift/"
input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/diapause_sensitivity/original_data/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]
	
file_name = paste0("combined_CMPOP_", version)
shifts = c(0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3)
for (shift in shifts){
	output = diapause_abs_rel_shifts(input_dir, 
		                             file_name,
	                                 param_dir,
	                                 diap_param=c(102.6077, 1.306483, 16.95815),
	                                 shift,
	                                 location_group_name="LocationGroups.csv")

	RelData = data.table(output[[1]])
	AbsData = data.table(output[[2]])
	sub1 = data.table(output[[3]])
	# pre_diap_plot = data.table(output[[4]])

	saveRDS(RelData, paste0(write_dir, "diapause_rel_", version, "_", shift, ".rds"))
	saveRDS(AbsData, paste0(write_dir, "diapause_abs_", version, "_", shift, ".rds"))
	saveRDS(sub1,    paste0(write_dir, "diapause_plot_", version, "_", shift, ".rds"))
	# saveRDS(pre_diap_plot, paste0(write_dir, "pre_diap_plot_", version, ".rds"))
}

