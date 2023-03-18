rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

source_path = "/Users/hn/Documents/GitHub/Kirti/hardiness/R_code/hardiness_core.R"
source(source_path)

param_dir = "/Users/hn/Documents/GitHub/Kirti/hardiness/R_code/parameters/"
input_data_dir = "/Users/hn/Documents/GitHub/Kirti/hardiness/R_code/input_data/"

# read parameters
options(digits=9)
input_params  = data.table(read.csv(paste0(param_dir, "input_parameters", ".csv")))
variety_params= data.table(read.csv(paste0(param_dir,"variety_parameters", ".csv")))

# read input temps
input_temps = data.table(read.csv(paste0(input_data_dir, "input_temps", ".csv")))
Prosser_temps= data.table(read.csv(paste0(input_data_dir, "Prosser_temps", ".csv")))

output <- hardiness_model(data=input_temps, input_params = input_params, 
	                      variety_params = variety_params)
output_dir <- "/Users/hn/Documents/GitHub/Kirti/hardiness/R_code/output_data/"
write.csv(output, 
	      file = paste0(output_dir, "model_output.csv"), 
	      row.names=FALSE)

########################################################################
##################
################## Plot
##################
########################################################################
source_path = "/Users/hn/Documents/GitHub/Kirti/hardiness/R_code/plot_core.R"
source(source_path)

plot_dir <- output_dir
out_name = "model_output"
plot_hardiness(output, plot_dir, out_name)

