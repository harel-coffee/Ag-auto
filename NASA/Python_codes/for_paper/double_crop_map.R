rm(list=ls())

library(ggmap)
library(ggpubr)
library(rgdal)
library(sp)
# library(sf)
library(foreign)

library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)

require(grid)   # for the textGrob() function

options(digits=9)
options(digit=9)


SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
SF_dir = "/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/"
pred_dir = "/Users/hn/Documents/01_research_data/NASA/RegionalStatData/"


# yrs=seq(2015, 2018)
# all_centrs = data.table()
# for (yr in yrs){
#   WSDA <- readOGR(paste0(SF_dir, "Eastern_", yr, "/Eastern_", yr, ".shp"),
#                   layer = paste0("Eastern_", yr), 
#                   GDAL1_integer64_policy = TRUE)

#   centroids <- rgeos::gCentroid(WSDA, byid=TRUE)

#   crs <- CRS("+proj=lcc 
#              +lat_1=45.83333333333334 
#              +lat_2=47.33333333333334 
#              +lat_0=45.33333333333334 
#              +lon_0=-120.5 +datum=WGS84")

#   centroid_coord <- spTransform(centroids, CRS("+proj=longlat +datum=WGS84"))

#   centroid_coord_dt <- data.table(centroid_coord@coords)
#   setnames(centroid_coord_dt, old=c("x", "y"), new=c("long", "lat"))
#   centroid_coord_dt$ID = WSDA@data$ID

#  all_centrs = rbind(all_centrs, centroid_coord_dt)
    
#   write.table(centroid_coord_dt, 
#             paste0(SF_data_dir, "Eastern_", yr, "_centroid.csv"), 
#             row.names = FALSE, col.names = TRUE, sep=",")
# }
# write.table(all_centrs, 
#             paste0(SF_data_dir, "all_eastern_centroid.csv"), 
#             row.names = FALSE, col.names = TRUE, sep=",")

all_centrs  = read.csv(paste0(SF_data_dir, "all_eastern_centroid.csv"))

irr_SF_data = read.csv(paste0(SF_data_dir, "irriigated_SF_data_concatenated.csv"))
irr_SF_data <- dplyr::left_join(x = irr_SF_data, y = all_centrs, by = "ID")
head(irr_SF_data, 2)


all_preds = read.csv(paste0(pred_dir, "all_preds.csv"), header=T, sep=",", as.is=T)
all_preds <- dplyr::left_join(x = all_preds, y = all_centrs, by = "ID")

length(unique(irr_SF_data$ID))
length(unique(all_preds$ID))


NDVI_reg_cols = c("SVM_NDVI_regular_preds", "KNN_NDVI_regular_preds", "DL_NDVI_regular_prob_point9", "RF_NDVI_regular_preds")
EVI_reg_cols  = c("SVM_EVI_regular_preds",   "KNN_EVI_regular_preds",  "DL_EVI_regular_prob_point4",  "RF_EVI_regular_preds")
NDVI_SG_cols  = c("SVM_NDVI_SG_preds",           "KNN_NDVI_SG_preds",      "DL_NDVI_SG_prob_point9",      "RF_NDVI_SG_preds")
EVI_SG_cols   = c("SVM_EVI_SG_preds",             "KNN_EVI_SG_preds",       "DL_EVI_SG_prob_point4",       "RF_EVI_SG_preds")

NDVI_reg = all_preds[c("ID", "long", "lat", NDVI_reg_cols)]
EVI_reg  = all_preds[c("ID", "long", "lat", EVI_reg_cols)]
NDVI_SG  = all_preds[c("ID", "long", "lat", NDVI_SG_cols)]
EVI_SG   = all_preds[c("ID", "long", "lat", EVI_SG_cols)]

states <- map_data("state")
states_cluster <- subset(states, region %in% c("washington"))

setnames(NDVI_reg, old = NDVI_reg_cols, new = c("SVM", "KNN", "DL", "RF"))
setnames(EVI_reg,  old = EVI_reg_cols, new = c("SVM", "KNN", "DL", "RF"))

setnames(NDVI_SG, old = NDVI_SG_cols, new = c("SVM", "KNN", "DL", "RF"))
setnames(EVI_SG,  old = EVI_SG_cols, new = c("SVM", "KNN", "DL", "RF"))


setcolorder(NDVI_reg, c("ID", "long", "lat", "SVM", "DL", "KNN", "RF"))
setcolorder(EVI_reg,  c("ID", "long", "lat", "SVM", "DL", "KNN", "RF"))
setcolorder(NDVI_SG,  c("ID", "long", "lat", "SVM", "DL", "KNN", "RF"))
setcolorder(EVI_SG,   c("ID", "long", "lat", "SVM", "DL", "KNN", "RF"))


NDVI_reg$method = "NDVI, 4-step smoothed"
EVI_reg$method  = "EVI, 4-step smoothed"
NDVI_SG$method  = "NDVI, 5-step smoothed"
EVI_SG$method   = "EVI, 5-step smoothed"

NDVI_reg_melt <- melt(data.table(NDVI_reg), id = c("ID", "long", "lat", "method"))
EVI_reg_melt  <- melt(data.table(EVI_reg),  id = c("ID", "long", "lat", "method"))
NDVI_SG_melt  <- melt(data.table(NDVI_SG),  id = c("ID", "long", "lat", "method"))
EVI_SG_melt   <- melt(data.table(EVI_SG),   id = c("ID", "long", "lat", "method"))


NDVI_reg_melt[ , value := as.character(value)]
NDVI_reg_melt[value == "1", value := "single-cropped"]
NDVI_reg_melt[value == "2", value := "double-cropped"]

EVI_reg_melt[ , value := as.character(value)]
EVI_reg_melt[value == "1", value := "single-cropped"]
EVI_reg_melt[value == "2", value := "double-cropped"]

NDVI_SG_melt[ , value := as.character(value)]
NDVI_SG_melt[value == "1", value := "single-cropped"]
NDVI_SG_melt[value == "2", value := "double-cropped"]

EVI_SG_melt[ , value := as.character(value)]
EVI_SG_melt[value == "1", value := "single-cropped"]
EVI_SG_melt[value == "2", value := "double-cropped"]

all_melt = rbind(NDVI_reg_melt, EVI_reg_melt, NDVI_SG_melt, EVI_SG_melt)

all_melt$method <- factor(all_melt$method, 
                            levels = c("NDVI, 4-step smoothed", "EVI, 4-step smoothed", 
                                       "NDVI, 5-step smoothed", "EVI, 5-step smoothed"), 
                            order = TRUE)

color_ord = c("red", "dodgerblue")
xlim_ = c(-125, -117)
ylim_ = c(45, 49)

theme_ = theme(axis.title.y = element_text(color = "black"),
               axis.title.x = element_text(color = "black"),
               axis.ticks.y = element_blank(), 
               axis.ticks.x = element_blank(),
               axis.text.x = element_text(color = "black"),
               axis.text.y = element_text(color = "black"),
               panel.grid.major = element_line(size = 0.1),
               legend.position="bottom", 
               legend.key=element_blank(),
               legend.text=element_text(size=12), #, face="bold"
               legend.title=element_blank(), # element_text(size=15),
               strip.text = element_text(size=12, face="bold"),
               plot.margin = margin(t=0, b=0, r=0.2, l=0.2, unit = 'cm')
               )

all_preds_onMap = all_melt %>% 
                  ggplot() +
                  geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group), fill = "grey", color = "black") +
                  facet_grid(~ variable ~ method) +
                  geom_point(aes_string(x = "long", y = "lat", color = "value"), alpha = 0.4, size=.1) +
                  labs(x = "longitude (degree)", y = "latitude (degree)") + # , color="prediction"
                  scale_y_continuous(breaks = c(45, 49)) + 
                  scale_x_continuous(breaks = c(-125, -118)) + 
                  coord_fixed(xlim = xlim_,  ylim = ylim_, ratio = 1.3) +
                  scale_color_manual(values=color_ord) + # change color of dots on the map
                  guides(color = guide_legend(override.aes = list(size=2))) + # make the dots in legend bigger
                  theme_ 

plot_path <- "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/preds_on_map/"

output_name = "all_preds_onMap.png"
dim_=8
ggsave(output_name, all_preds_onMap, path=plot_path, 
       width=8.5, height=6.2, unit="in", dpi = 200)

