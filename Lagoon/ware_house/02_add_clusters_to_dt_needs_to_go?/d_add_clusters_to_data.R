.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

###########################################################
main_in <- "/data/hydro/users/Hossein/lagoon/01_storm_cumPrecip/"
param_dir <- "/home/hnoorazar/lagoon_codes/parameters/"
###########################################################
obs_clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()

subdir <- c("cum_precip/annual/"); print(subdir)
# subdir <- c("cum_precip/chunky/"); print(subdir)
# subdir <- c("cum_precip/monthly/"); print(subdir)
# subdir <- c("cum_precip/wtr_yr/"); print(subdir)
# subdir <- c("storm/"); print(subdir)

for (sub in subdir){
  in_dir <- file.path(paste0(main_in, sub))
  files_list <- list.files(path=in_dir, pattern=".rds")
  for (file in files_list){
    A <- readRDS(paste0(in_dir, file))
    A <- as.data.frame(A)
    A <- A[,unique(names(A))]
    if ("cluster" %in% colnames(A)){A <- within(A, remove(cluster))}
    if ("cluster.x" %in% colnames(A)){A <- within(A, remove(cluster.x))}
    if ("cluster.y" %in% colnames(A)){A <- within(A, remove(cluster.y))}
    
    A <- data.table(A)
    A <- merge(A, obs_clusters, by="location", all.x=T)
    saveRDS(A, paste0(in_dir, file))
  }
}

end_time <- Sys.time()
print( end_time - start_time)

