#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)
source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)



data_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
plot_path = "/data/hydro/users/Hossein/codling_moth/local/processed/plots/"

box_width = 0.25 
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")

file_name = "combined_CM_rcp45.rds"
plot_output_name = "adult_emergence_rcp45"
plot_adult_emergence(input_dir, file_name, box_width=.25, plot_path, plot_output_name)

file_name = "combined_CM_rcp85.rds"
plot_output_name = "adult_emergence_rcp85"
plot_adult_emergence(input_dir, file_name, box_width=.25, plot_path, plot_output_name)



