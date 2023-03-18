
in_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/location_level/"
local_locs <- read.table(paste0(in_dir, "county_avg_file_names"), as.is=T)
local_locs <- local_locs$V1
local_locs <- paste0(local_locs, ".sh")

q_rcp45_w_precip <- local_locs
q_rcp45_no_precip <- local_locs

q_rcp85_w_precip <- local_locs
q_rcp85_no_precip <- local_locs

q_rcp45_w_precip <- paste0("q_rcp45_w_precip_", q_rcp45_w_precip)
q_rcp45_no_precip <- paste0("q_rcp45_no_precip_", q_rcp45_no_precip)

q_rcp85_w_precip <- paste0("q_rcp85_w_precip_", q_rcp85_w_precip)
q_rcp85_no_precip<- paste0("q_rcp85_no_precip_", q_rcp85_no_precip)

param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/location_level/post_biofix/"
write.table(q_rcp85_w_precip, file = paste0(param_dir, "q_rcp85_w_precip_cnty_avg"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp85_no_precip, file = paste0(param_dir, "q_rcp85_no_precip_cnty_avg"), sep="\t", row.names=F, col.names=F, quote=F)

write.table(q_rcp45_w_precip, file = paste0(param_dir, "q_rcp45_w_precip_cnty_avg"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_no_precip, file = paste0(param_dir, "q_rcp45_no_precip_cnty_avg"), sep="\t", row.names=F, col.names=F, quote=F)





