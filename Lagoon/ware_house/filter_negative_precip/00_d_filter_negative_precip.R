.libPaths("/data/hydro/R_libs35")
.libPaths()

library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

##### read file
in_dir <- "/data/hydro/users/Hossein/lagoon/00_raw_data/"
file_name <- "raw_RCP85.rds.rds"

A <- readRDS(paste0(in_dir, file_name))
A <- A %>% filter(model == "MRI_CGCM3" & 
                  emission == "RCP 8.5" & 
                  location == "48.71875_-120.96875" &
                  year == 2052 &
                  month== 10) %>%
             data.table()

saveRDS(A, paste0(in_dir, "weird_loc.rds"))
