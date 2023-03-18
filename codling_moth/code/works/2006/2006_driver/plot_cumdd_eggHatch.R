#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)
source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)


args = commandArgs(trailingOnly=TRUE)
version = args[1]
output_type = args[2]


data_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
plot_path = "/data/hydro/users/Hossein/codling_moth/local/processed/plots/"

plot_cumdd_eggHatch(input_dir=data_dir, file_name ="combined_CMPOP_", version, output_dir=plot_path, output_type)
