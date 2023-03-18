
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

local_locs <- paste0("feat_", local_locs, "_")

years = seq(2026, 2095)

local_locs <- as.character(sapply(local_locs, FUN = function(x) paste0(x, years)))
local_locs <- paste0(local_locs, ".rds")

write.table(local_locs, file = paste0(param_dir, "file_names.txt"), sep = "\t", row.names = F, col.names = F, quote=F)

write.table(missing_locations, file = paste0(param_dir, "missing_locations.txt"), sep = "\t", row.names = F, col.names = F, quote=F)



rcp85_wgen3

write.table(rcp85_no_gen3, file = paste0(param_dir, "q_rcp85_nogen3"), sep="\t", row.names=F, col.names=F, quote=F)



