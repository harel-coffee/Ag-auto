.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
options(digit=9)
options(digits=9)


in_dir <- paste0("/data/hydro/users/kraghavendra/", 
                 "hardiness/output_data/Plots/facet", 
                 "/observed/")

files <- list.files(path = , 
                    pattern = "png")

print (length(files))
files <- data.table(files)
setnames(files, old=c("files"), new=c("location"))
files$location = gsub(".png", "", files$location)

x <- sapply(files$location, function(x) strsplit(x, "-")[[1]], 
            USE.NAMES=FALSE)
lat = x[1, ]; long = x[2, ];

files$lat <- as.numeric(lat)
files$long <- -1 * as.numeric(long)
files$location <- paste0(files$lat, " N, ", abs(files$long), " W")
files$color <- files$lat + abs(files$long)

print (dim(files))

out_dir <- "/home/hnoorazar/hardiness_codes/parameters/"

saveRDS(files, paste0(out_dir, "1414_locs.rds"))
