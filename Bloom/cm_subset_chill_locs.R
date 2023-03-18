# find locations of CM in chill data


library(data.table)
library(dplyr)

dir_base <- "/Users/hn/Documents/GitHub/Ag/"
cm_param_dir <- "/codling_moth/code/parameters/"
bloom_param_dir <- "/Bloom/parameters/"

cm_locs <- read.csv(paste0(dir_base, 
                           cm_param_dir, 
                           "local_list.txt"),
                    header=FALSE) %>% 
           data.table()

bloom_locs <- read.csv(paste0(dir_base, 
                              bloom_param_dir, 
                              "file_list.txt"),
                       header=FALSE) %>% 
           data.table()

setnames(cm_locs, old=c("V1"), new=c("location"))
setnames(bloom_locs, old=c("V1"), new=c("location"))

bloom_locs$location <- gsub("data_", "", bloom_locs$location)

write.table(cm_locs, 
            file = paste0(dir_base, bloom_param_dir,
                          "cm_locs.csv"), 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

