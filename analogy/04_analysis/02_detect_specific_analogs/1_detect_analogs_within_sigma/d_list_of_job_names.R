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

part_1 <- "/home/hnoorazar/analog_codes/00_post_biofix/03_detect_analogs_4_plots/"
in_dir <- paste0(part_1, "02_find_analogs_within_sigma/all_qsubs")
setwd(in_dir)
getwd()

jobs_list <- dir(in_dir, pattern = ".rds.sh")


print (head(jobs_list))
write.table(jobs_list, 
            file = paste0(out_dir, "jobs_list.csv"), 
            row.names = FALSE,
            col.names = TRUE, 
            sep = ",", na = "")

jobs_list <- data.table(read.csv(paste0(out_dir, "jobs_list.csv"), as.is=T))
setnames(jobs_list, old=c("x"), new=c("job_names"))


write.table(jobs_list, 
            file = paste0(out_dir, "jobs_list.csv"), 
            row.names = FALSE,
            col.names = TRUE, 
            sep = ",", na = "")

write.table(jobs_list, file = paste0(out_dir, "jobs_list.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)


batch_1 <- jobs_list[1:285]
batch_2 <- jobs_list[286:570]
batch_3 <- jobs_list[571:855]
batch_4 <- jobs_list[856:1140]
batch_5 <- jobs_list[1141:1224]


write.table(batch_1, file = paste0(out_dir, "batch_1.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

write.table(batch_2, file = paste0(out_dir, "batch_2.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

write.table(batch_3, file = paste0(out_dir, "batch_3.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

write.table(batch_4, file = paste0(out_dir, "batch_4.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)

write.table(batch_5, file = paste0(out_dir, "batch_5.txt"), sep = "\t",
            row.names = FALSE, col.names = FALSE, quote=FALSE)


