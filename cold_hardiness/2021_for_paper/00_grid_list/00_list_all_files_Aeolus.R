
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(ggplot2)
library(dplyr)

options(digits=9)
options(digit=9)

#######################
####################### Directories
#######################
in_dir <- "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
out_dir <- "/home/hnoorazar/"

if (dir.exists(out_dir) == F) {
  dir.create(path =out_dir, recursive = T)
}

#######################
####################### Body
#######################

######
###### List all files
######
all_grids_list = list.files(path=in_dir, 
                            pattern = "data_", 
                            full.names = FALSE, # a logical value. If TRUE, the directory path 
                                                # is prepended to the file names to give a relative 
                                                # file path. If FALSE, the file names 
                                                # (rather than paths) are returned.

                            recursive = FALSE, # Should the listing recurse into directories?
                            ignore.case = TRUE # Should pattern-matching be case-insensitive?
                            )

######
###### break the file names and extract lat and long.
######
x <- sapply(all_grids_list, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
lat = x[2, ]
long = x[3, ]


######
###### form a datatable
######
no_files <- length(lat)

fileTable <- setNames(data.table(matrix(nrow = no_files, ncol = 2)), 
                      c("lat",  "long")
                      )


fileTable$lat <- lat
fileTable$long <- long

######
###### save the data to disk
######
write.table(fileTable, 
            file = paste0(out_dir, "/all_grids_latLong.csv"), 
            row.names=FALSE, na="", col.names=TRUE, sep=",")


######
###### Subset for Supriya
######  Find the locations that are below the northern border of Wyoming and 
######  on the right side of Wyoming's western border
######

fileTable$lat <- as.numeric((fileTable$lat))
fileTable$long <- as.numeric((fileTable$long))


fileTable <- fileTable %>% 
             filter(lat <= 45.03125) %>% 
             data.table()

fileTable <- fileTable %>% 
             filter(lat >= 41.00348) %>% 
             data.table()

fileTable <- fileTable %>% 
             filter(long >= -111.09375) %>% 
             data.table()

fileTable <- fileTable %>% 
             filter(long <= -104.05400) %>% 
             data.table()

new_out_dir <- "/home/hnoorazar/hardiness_codes/parameters/"

if (dir.exists(new_out_dir) == F) {
  dir.create(path =new_out_dir, recursive = T)
}

write.table(fileTable, 
            file = paste0(new_out_dir, "/WyomingLocsForSupria.csv"), 
            row.names=FALSE, na="", col.names=TRUE, sep=",")



#########

# WyomingLocsForSupria$lat <- as.character(WyomingLocsForSupria$lat)
# WyomingLocsForSupria$long <- as.character(WyomingLocsForSupria$long)

dir <- "/Users/hn/Documents/00_GitHub/Ag/hardiness/R_code/parameters/"
WyomingLocsForSupria <- read.csv(paste0(dir, "WyomingLocsForSupria.csv"), as.is=TRUE)

WyomingLocsForSupria %>%
ggplot() +
geom_polygon(data = states, aes(x=long, y=lat)) +
geom_point(aes_string(x = "long", y = "lat", color="lat"), alpha = 0.1, size=.1) +
theme(axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.ticks.y = element_blank(), 
      axis.ticks.x = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      panel.grid.major = element_line(size = 0.1),
      legend.position="bottom", 
      legend.title = element_blank(),
      strip.text = element_text(size=12, face="bold"),
      plot.margin = margin(t=-0.5, r=0.2, b=0, l=0.2, unit = 'cm')
     ) 

