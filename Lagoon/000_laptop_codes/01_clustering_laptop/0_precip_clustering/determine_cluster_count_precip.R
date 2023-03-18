library(data.table)
library(dplyr)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)


dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/cum_precip/"
observed <- readRDS(paste0(dir, "ann_cum_precip_last_day_observed.rds"))

max_clusters = 10
for_elbow = data.table(no_clusters = c(1:max_clusters),
                       total_within_cluster_ss = rep(-666, max_clusters))

for (k in 1:max_clusters){
  set.seed(100)
  output <- cluster_yr_avging(observed, scale=FALSE, no_clusters=k)
  clusters_obj <- output[[2]]
  for_elbow[k, "total_within_cluster_ss"] <- clusters_obj$tot.withinss
}

plot(for_elbow)
