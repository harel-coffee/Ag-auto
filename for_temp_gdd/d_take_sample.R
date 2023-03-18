######################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
#####################################################################

.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)

data_dir = "/data/hydro/users/Hossein/temp_gdd/"

file_name = "modeled_rcp45.rds"
modeled_rcp45 <- data.table(readRDS(paste0(data_dir, file_name)))
modeled_rcp45 = modeled_rcp45[1:100, ]
saveRDS(modeled_rcp45, paste0(data_dir, "sample_modeled_rcp45.rds"))

file_name = "modeled_historical.rds"
modeled_hist <- data.table(readRDS(paste0(data_dir, file_name)))
modeled_hist = modeled_hist[1:100, ]
saveRDS(modeled_hist, paste0(data_dir, "sample_modeled_hist.rds"))



