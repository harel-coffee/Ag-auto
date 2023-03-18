
# generate feature names

param_dir = "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

local_locs <- read.csv(paste0(param_dir, "local_county_fips.csv"), as.is = T)
all_us_locs <- read.csv(paste0(param_dir, "all_us_1300_county_fips_locations.csv"), as.is = T)

local_locs <- local_locs$location
all_us_locs <- all_us_locs$location

local_locs <- gsub("[.]", "_", local_locs)
all_us_locs <- gsub("[.]", "_", all_us_locs)

local_locs <- gsub("-", "", local_locs)
all_us_locs <- gsub("-", "", all_us_locs)

missing_locations <- local_locs[!(local_locs %in% all_us_locs)]
local_locs <- local_locs[local_locs %in% all_us_locs]

local_locs <- paste0("feat_", local_locs, ".rds")

write.table(local_locs, file = paste0(param_dir, "/location_level/", "file_names.txt"), sep = "\t", row.names = F, col.names = F, quote=F)




