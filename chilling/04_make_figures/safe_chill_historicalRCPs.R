rm(list=ls())
library(data.table)
library(dplyr)
library(ggmap)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path = "/Users/hn/Documents/GitHub/Kirti/Chilling/4th_draft/chill_plot_core.R"
source(source_path)

##########################################################################################
###                                                                                    ###
###                                    Driver                                          ###
###                                                                                    ###
##########################################################################################
param_dir <- "/Users/hn/Documents/GitHub/Kirti/chilling/parameters/"
locations_wanted <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), header=T, as.is=T)

time_types = c("non_overlapping")     #, "overlapping"
model_types = c("dynamic_model_stats")#, "utah_model_stats"

main_in = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
files_name = c("mid_nov_summary_comp.rds", "mid_oct_summary_comp.rds", "mid_sept_summary_comp.rds",
               "nov_summary_comp.rds", "oct_summary_comp.rds", "sept_summary_comp.rds")

time_type = time_types[1]
model_type = model_types[1]
plot_dir <- "/Users/hn/Desktop/plots/"

setwd(main_in)

datas = data.table(readRDS("sept_summary_comp.rds"))
start = "Sept. 1"

datas <- datas %>% filter(model != "observed")

###### Remove Montana, add Warm, Cool, Oregon
datas <- remove_montana_add_warm_cold(data_dt=datas, LocationGroups_NoMontana=locations_wanted)

climate_levels <- c("Cooler Areas", "Warmer Areas", "Oregon Areas")
datas$warm_cold <- factor(datas$warm_cold, levels = climate_levels)

information = produce_data_4_plots(datas)

safe_jan <- safe_box_plot(information[[1]], due="Jan.", chill_start = start)
safe_feb <- safe_box_plot(information[[4]], due="Feb.", chill_start = start)
safe_mar <- safe_box_plot(information[[7]], due="Mar.", chill_start = start)
safe_apr <- safe_box_plot(information[[10]], due="Apr.", chill_start= start)

# out_dir
box_width = 10
box_height = 8

plot_dir <- "/Users/hn/Desktop/"

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Jan.png")
ggsave(output_name, safe_jan, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Feb.png")
ggsave(output_name, safe_feb, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Mar.png")
ggsave(output_name, safe_mar, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_Apr.png")
ggsave(output_name, safe_apr, path=plot_dir, width=box_width, height=box_height, unit="in", dpi=400)

# means over models
# mean_map_jan = ensemble_map(data=information[[2]], color_col="mean_over_model", due="Jan.")
# mean_map_feb = ensemble_map(data=information[[5]], color_col="mean_over_model", due="Feb.")

# output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_jan.png") 
# ggsave(output_name, mean_map_jan, path=out_dir, width=7, height=4.5, unit="in", dpi=400)

# output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_feb.png") 
# ggsave(output_name, mean_map_feb, path=out_dir, width=7, height=4.5, unit="in", dpi=400)

# medians over models
median_map_jan = ensemble_map(data=information[[3]], color_col="median_over_model", due="Jan.")
median_map_feb = ensemble_map(data=information[[6]], color_col="median_over_model", due="Feb.")
median_map_mar = ensemble_map(data=information[[9]], color_col="median_over_model", due="Mar.")
median_map_apr = ensemble_map(data=information[[12]], color_col="median_over_model", due="Mar.")

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_jan.png") 
ggsave(output_name, median_map_jan, path=plot_dir, width=7, height=4.5, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_feb.png") 
ggsave(output_name, median_map_feb, path=plot_dir, width=7, height=4.5, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_mar.png") 
ggsave(output_name, median_map_mar, path=plot_dir, width=7, height=4.5, unit="in", dpi=400)

output_name = paste0(time_type, "_", unlist(strsplit(model_type, "_"))[1], "_map_apr.png") 
ggsave(output_name, median_map_apr, path=plot_dir, width=7, height=4.5, unit="in", dpi=400)



