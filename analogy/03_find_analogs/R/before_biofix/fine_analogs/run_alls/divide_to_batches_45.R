q_rcp45_wgen3 <- read.table("/Users/hn/Documents/GitHub/Kirti/analogy/parameters/q_rcp45_wgen3", as.is=T)
q_rcp45_wgen3 <- q_rcp45_wgen3$V1

L <- length(q_rcp45_wgen3)
batch_nos = 8
l = L/batch_nos
q_rcp45_wgen3_batch_1 <- q_rcp45_wgen3[1:l]
q_rcp45_wgen3_batch_2 <- q_rcp45_wgen3[(l+1):(2*l)]
q_rcp45_wgen3_batch_3 <- q_rcp45_wgen3[(2*l+1):(3*l)]
q_rcp45_wgen3_batch_4 <- q_rcp45_wgen3[(3*l+1):(4*l)]
q_rcp45_wgen3_batch_5 <- q_rcp45_wgen3[(4*l+1):(5*l)]
q_rcp45_wgen3_batch_6 <- q_rcp45_wgen3[(5*l+1):(6*l)]
q_rcp45_wgen3_batch_7 <- q_rcp45_wgen3[(6*l+1):(7*l)]
q_rcp45_wgen3_batch_8 <- q_rcp45_wgen3[(7*l+1):(8*l)]

length(q_rcp45_wgen3_batch_1) +
length(q_rcp45_wgen3_batch_2) +
length(q_rcp45_wgen3_batch_3) +
length(q_rcp45_wgen3_batch_4) +
length(q_rcp45_wgen3_batch_5) +
length(q_rcp45_wgen3_batch_6) +
length(q_rcp45_wgen3_batch_7) +
length(q_rcp45_wgen3_batch_8)

param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/broken_down_jobs/"
write.table(q_rcp45_wgen3_batch_1, file = paste0(param_dir, "q_rcp45_wgen3_batch_1"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_2, file = paste0(param_dir, "q_rcp45_wgen3_batch_2"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_3, file = paste0(param_dir, "q_rcp45_wgen3_batch_3"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_4, file = paste0(param_dir, "q_rcp45_wgen3_batch_4"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_5, file = paste0(param_dir, "q_rcp45_wgen3_batch_5"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_6, file = paste0(param_dir, "q_rcp45_wgen3_batch_6"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_7, file = paste0(param_dir, "q_rcp45_wgen3_batch_7"), sep="\t", row.names=F, col.names=F, quote=F)
write.table(q_rcp45_wgen3_batch_8, file = paste0(param_dir, "q_rcp45_wgen3_batch_8"), sep="\t", row.names=F, col.names=F, quote=F)
