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

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
plot_path = "/data/hydro/users/Hossein/codling_moth_new/local/processed/plots/"
if (dir.exists(file.path(plot_path)) == F) {
  dir.create(path = plot_path, recursive = T)
}

cumulative_qrt_boxplot(input_dir=data_dir, file_name ="combined_CMPOP_", version, output_dir=plot_path)