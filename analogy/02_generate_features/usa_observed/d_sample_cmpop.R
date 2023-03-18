.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)

options(digit=9)
options(digits=9)

write_dir = "/data/hydro/users/Hossein/analog/usa/"
in_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"

all_usa <- data.table(readRDS(paste0(in_dir, "combined_CMPOP.rds")))
all_usa <- all_usa[1:10, ]

saveRDS(all_usa, paste0(write_dir, "all_usa_cmpop_sample.rds"))
