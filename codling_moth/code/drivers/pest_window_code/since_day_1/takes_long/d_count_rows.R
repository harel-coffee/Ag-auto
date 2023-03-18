library(chron)
library(data.table)
library(dplyr)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_rcp"

curr_data = readRDS(paste0(data_dir, name_pref, "45.rds"))
a = dim(curr_data %>% group_by_all %>% count)[1]

write.table(a, file = paste0("unique_row_count.txt"), sep = "\t", row.names = TRUE, col.names = NA)