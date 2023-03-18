library(data.table)
library(dplyr)

all_storms <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/all_storms.rds")
all_storms <- within(all_storms, remove(cluster))

cluster_labes <- read.csv("/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/precip_elev_5_clusters.csv", as.is=TRUE)

all_storms <- merge(all_storms, cluster_labes, by="location", all.x=TRUE)
all_storms <- na.omit(all_storms)

saveRDS(all_storms, "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/all_storms.rds")