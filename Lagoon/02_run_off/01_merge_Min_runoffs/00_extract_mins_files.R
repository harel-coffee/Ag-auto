.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

# Time the processing of this batch of files
start_time <- Sys.time()

################################################################
main_in <- "/data/hydro/users/liumingdata/ForKirtiLagoon/Mingliang_Kirti_Lagoon_shortvar/bcc-csm1-1/historical/"
out_dir <- file.path("/home/hnoorazar/lagoon_codes/parameters/")

list_files <- list.files(main_in)
list_files <- list_files[grep(pattern = "flux_", x = list_files)]
list_files <- data.table(list_files)

list_files$list_files <- gsub("flux_", "", list_files$list_files)

write.table(list_files, 
            file = paste0(out_dir, "min_list_files.csv"),
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")



