#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

args = commandArgs(trailingOnly=TRUE)
version = args[1]

file_name = paste0("combined_CMPOP_", version, ".rds")
shifts = c("0", "5", "10", "15", "20")

for (shift in shifts){
	input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_wider/one_loc/"
	input_dir = paste0(input_dir, shift, "/")
	
	write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_wider/one_loc/"
	write_dir = paste0(write_dir, shift, "/")
	output = generations_func(input_dir, file_name)
	generations_Aug = data.table(output[[1]])
	generations_Nov = data.table(output[[2]])

	saveRDS(generations_Aug, paste0(write_dir, "generations_Aug_combined_CMPOP_", version, ".rds"))
	saveRDS(generations_Nov, paste0(write_dir, "generations_Nov_combined_CMPOP_", version, ".rds"))
}

