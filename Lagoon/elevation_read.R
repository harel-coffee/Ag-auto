# read elevations of Min

in_dir <- "/Users/hn/Documents/01_research_data/large_4_GitHub/Min_DB/"
out_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
elevation <- read.table(paste0(in_dir, "elevation.txt"), header = FALSE)
elevation <- subset(elevation, select=c(V3, V4, V78))


write.table(elevation, file = paste0(out_dir, "elevation.csv"),
            row.names=FALSE, col.names=TRUE, na="", sep=",")