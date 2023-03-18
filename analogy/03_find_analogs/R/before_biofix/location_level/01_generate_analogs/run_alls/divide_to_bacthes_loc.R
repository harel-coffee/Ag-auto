

local_locs <- read.table("/Users/hn/Documents/GitHub/Kirti/analogy/parameters/location_level/file_names", as.is=T)
local_locs <- local_locs$V1
local_locs <- paste0(local_locs, ".sh")

q_rcp45_w_precip_w_gen3 <- local_locs
q_rcp45_w_precip_no_gen3 <- local_locs
q_rcp45_no_precip_w_gen3 <- local_locs
q_rcp45_no_precip_no_gen3 <- local_locs

q_rcp85_w_precip_w_gen3 <- local_locs
q_rcp85_w_precip_no_gen3 <- local_locs
q_rcp85_no_precip_w_gen3 <- local_locs
q_rcp85_no_precip_no_gen3 <- local_locs

q_rcp45_w_precip_w_gen3 <- paste0("q_rcp45_w_precip_w_gen3_", q_rcp45_w_precip_w_gen3)
q_rcp45_w_precip_no_gen3 <- paste0("q_rcp45_w_precip_no_gen3_", q_rcp45_w_precip_no_gen3)
q_rcp45_no_precip_w_gen3 <- paste0("q_rcp45_no_precip_w_gen3_", q_rcp45_no_precip_w_gen3)
q_rcp45_no_precip_no_gen3 <- paste0("q_rcp45_no_precip_no_gen3_", q_rcp45_no_precip_no_gen3)


q_rcp85_w_precip_w_gen3 <- paste0("q_rcp85_w_precip_w_gen3_", q_rcp85_w_precip_w_gen3)
q_rcp85_w_precip_no_gen3 <- paste0("q_rcp85_w_precip_no_gen3_", q_rcp85_w_precip_no_gen3)
q_rcp85_no_precip_w_gen3 <- paste0("q_rcp85_no_precip_w_gen3_", q_rcp85_no_precip_w_gen3)
q_rcp85_no_precip_no_gen3 <- paste0("q_rcp85_no_precip_no_gen3_", q_rcp85_no_precip_no_gen3)

param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/location_level/"
write.table(q_rcp85_w_precip_w_gen3, file = paste0(param_dir, "q_rcp85_w_precip_w_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp85_w_precip_no_gen3, file = paste0(param_dir, "q_rcp85_w_precip_no_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp85_no_precip_w_gen3, file = paste0(param_dir, "q_rcp85_no_precip_w_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp85_no_precip_no_gen3, file = paste0(param_dir, "q_rcp85_no_precip_no_gen3"), sep="\t", row.names=F, col.names=F, quote=F)

write.table(q_rcp45_w_precip_w_gen3, file = paste0(param_dir, "q_rcp45_w_precip_w_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_w_precip_no_gen3, file = paste0(param_dir, "q_rcp45_w_precip_no_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_no_precip_w_gen3, file = paste0(param_dir, "q_rcp45_no_precip_w_gen3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_no_precip_no_gen3, file = paste0(param_dir, "q_rcp45_no_precip_no_gen3"), sep="\t", row.names=F, col.names=F, quote=F)





