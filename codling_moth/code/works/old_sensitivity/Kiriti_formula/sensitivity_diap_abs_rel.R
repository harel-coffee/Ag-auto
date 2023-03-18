#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_1/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]
file_name = paste0("combined_CMPOP_", version)

shifts = c("0", "5", "10", "15", "20")
for (shift in shifts){
	input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_1/"
	input_dir = paste0(input_dir, shift, "/")

	write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_1/"
	write_dir = paste0(write_dir, shift, "/")
	
    output = diapause_abs_rel(input_dir, 
	                          file_name,
                              param_dir, 
                              location_group_name="LocationGroups.csv")
    RelData = data.table(output[[1]])
	AbsData = data.table(output[[2]])
	sub1 = data.table(output[[3]])

	saveRDS(RelData, paste0(write_dir, "diapause_rel_data_", version, ".rds"))
	saveRDS(AbsData, paste0(write_dir, "diapause_abs_data_", version, ".rds"))
	saveRDS(sub1,    paste0(write_dir, "diapause_plot_data_", version, ".rds"))
}
	
