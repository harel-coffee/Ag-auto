#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)
source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)


input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
plot_path = "/data/hydro/users/Hossein/codling_moth/local/processed/plots/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")


##################
################## Number of Larva Generations by August 23
##################

stage = "Larva"
file_name = "generations_combined_CMPOP_rcp85.rds"
plot_generations_Aug23(input_dir,
                      file_name,
                      stage,
                      box_width=.25,
                      plot_path,
                      version="rcp85",
                      color_ord)

file_name = "generations_combined_CMPOP_rcp45.rds"
plot_generations_Aug23(input_dir,
                       file_name,
                       stage,
                       box_width=.25,
                       plot_path,
                       version="rcp45",
                       color_ord)

##################
################## Number of Adult Generations by August 23
##################
stage = "Adult"
file_name = "generations_combined_CMPOP_rcp85.rds"
plot_generations_Aug23(input_dir,
                      file_name,
                      stage,
                      box_width=.25,
                      plot_path,
                      version="rcp85",
                      color_ord)

file_name = "generations_combined_CMPOP_rcp45.rds"
plot_generations_Aug23(input_dir,
                       file_name,
                       stage,
                       box_width=.25,
                       plot_path,
                       version="rcp45",
                       color_ord)

