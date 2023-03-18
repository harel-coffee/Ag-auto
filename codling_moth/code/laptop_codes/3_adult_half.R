rm(list=ls())
library(data.table)
library(ggpubr)
library(reshape2)
library(dplyr)
library(cowplot)

input_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
plot_path <- "/Users/hn/Desktop/"
versions <- c("rcp45", "rcp85")
stages <- c("larva") # "adult", 
for (stage in stages){
  for (version in versions){
    output_name = paste0(stage, "_flight_", version, "_half.png")
    input_name = paste0("combined_CM_", version, ".rds")
    plot_flight_DoY_half(input_dir=input_dir, input_name=input_name, 
    	                 stage=stage, output_dir=plot_path, 
    	                 output_name=output_name,
    	                 plot_with=7, plot_height=4)
  }
}
