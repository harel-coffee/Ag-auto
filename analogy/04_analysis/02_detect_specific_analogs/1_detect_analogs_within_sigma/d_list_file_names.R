.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

# NN_sigma_list <- list.files(path = "/data/hydro/users/Hossein/analog/03_analogs/biofixed/location_level/w_precip_rcp45/", 
#               pattern = "NN_sigma_", all.files = FALSE,
#               full.names = FALSE, recursive = FALSE,
#               ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)
# print(list_files)


out_dir <- "/home/hnoorazar/analog_codes/parameters/to_detect/"


in_dir <- "/data/hydro/users/Hossein/analog/03_analogs/biofixed/detected_4_plots/01_intr_cnty_analogs/1_sigma/"
setwd(in_dir)
getwd()

NN_loc_list <- dir(in_dir, pattern = "NN_loc_year_tb_")
NN_sigma_list <- dir(in_dir, pattern = "NN_sigma_tb_")

print (head(NN_sigma_list))
write.table(NN_sigma_list, 
            file = paste0(out_dir, "NN_sigma_list.csv"), 
            row.names = FALSE,
            col.names = TRUE, 
            sep = ",", na = "")

NN_sigma_list <- data.table(read.csv(paste0(out_dir, "NN_sigma_list.csv"), as.is=T))
setnames(NN_sigma_list, old=c("x"), new=c("file_names"))

NN_sigma_list$file_names = gsub("NN_sigma_tb_", "tb_", NN_sigma_list$file_names)

write.table(NN_sigma_list, 
            file = paste0(out_dir, "NN_sigma_list.csv"), 
            row.names = FALSE,
            col.names = TRUE, 
            sep = ",", na = "")

write.table(NN_sigma_list, file = paste0(param_dir, "NN_sigma_list.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

write.table(NN_sigma_list, file = paste0(param_dir, "NN_sigma_list"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

# NN_loc_list <- data.table(read.csv(paste0(param_dir, "NN_loc_list.csv"), as.is=T))
# setnames(NN_loc_list, old=c("x"), new=c("loc_names"))
# write.table(NN_loc_list, 
#             file = paste0(param_dir, "NN_loc_list.txt"), 
#             sep = "\t", row.names = F, col.names = F, quote=F)
#
#

# write.table(NN_loc_list, 
#             file = paste0(out_dir, "NN_loc_list.csv"), 
#             row.names = FALSE, 
#             col.names = TRUE, 
#             sep = ",", na = "")





