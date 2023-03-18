rm(list=ls())
library(data.table)
library(rgdal)
library(sp)
library(dplyr)
library(ggplot2)
library(maps)
library(purrr)
library(scales) # includes pretty_breaks

##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

double_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                     "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                     "06_list_of_2cropped_IDs_4_website/")


output_dir <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/remote_sensing/plots_for_paper/")
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Parameter setup
#
plot_w = 8
plot_h = 5.5
years <- c(2016, 2017, 2018)

eastern_counties <- c("adams", "asotin", "benton", "chelan", 
                      "columbia", "douglas", "ferry", "franklin", 
                      "garfield", "grant", "kittitas", "klickitat",
                      "lincoln", "okanogan", "pend oreille", "spokane", 
                      "stevens", "walla walla", "whitman", "yakima")
##############################################################################################################
#
#    Body
#
states <- map_data("state")
states_cluster <- subset(states, region %in% c("washington"))

states_cluster <- map_data("county", "washington")

mid_range <- function(x) mean(range(x))
county_names <- do.call(rbind, lapply(split(states_cluster, states_cluster$subregion), function(d) {
  data.frame(lat = mid_range(d$lat), long = mid_range(d$long), subregion = unique(d$subregion))
}))

county_names <- county_names %>%
                filter(subregion %in% eastern_counties)
county_names$subregion <- stringr::str_to_title(county_names$subregion)
########################################################
####
#### Take care of location of county names
####
county_names$lat <- county_names$lat + 0.09

county_names$lat[county_names$subregion=="Adams"] <- county_names$lat[county_names$subregion=="Adams"] - 0.07
county_names$long[county_names$subregion=="Adams"] <- county_names$long[county_names$subregion=="Adams"] + 0.12

county_names$lat[county_names$subregion=="Spokane"] <- county_names$lat[county_names$subregion=="Spokane"] - 0.1
county_names$lat[county_names$subregion=="Asotin"] <- county_names$lat[county_names$subregion=="Asotin"] - 0.15
county_names$lat[county_names$subregion=="Garfield"] <- county_names$lat[county_names$subregion=="Garfield"] + 0.1
county_names$lat[county_names$subregion=="Columbia"] <- county_names$lat[county_names$subregion=="Columbia"] - 0.06
county_names$long[county_names$subregion=="Whitman"] <- county_names$long[county_names$subregion=="Whitman"] + 0.2

county_names$lat[county_names$subregion=="Grant"] <- county_names$lat[county_names$subregion=="Grant"] - 0.05
county_names$long[county_names$subregion=="Grant"] <- county_names$long[county_names$subregion=="Grant"] - 0.22

county_names$lat[county_names$subregion=="Lincoln"] <- county_names$lat[county_names$subregion=="Lincoln"] - 0.1
county_names$long[county_names$subregion=="Lincoln"] <- county_names$long[county_names$subregion=="Lincoln"] + 0.1

county_names$lat[county_names$subregion=="Stevens"] <- county_names$lat[county_names$subregion=="Stevens"] - 0.1
county_names$long[county_names$subregion=="Stevens"] <- county_names$long[county_names$subregion=="Stevens"] + 0.05

county_names$lat[county_names$subregion=="Yakima"] <- county_names$lat[county_names$subregion=="Yakima"] - 0.1
county_names$long[county_names$subregion=="Yakima"] <- county_names$long[county_names$subregion=="Yakima"] - 0.2

county_names$lat[county_names$subregion=="Benton"] <- county_names$lat[county_names$subregion=="Benton"] + 0.1
county_names$long[county_names$subregion=="Benton"] <- county_names$long[county_names$subregion=="Benton"] - 0.12

county_names$lat[county_names$subregion=="Franklin"] <- county_names$lat[county_names$subregion=="Franklin"] + 0.1
county_names$long[county_names$subregion=="Franklin"] <- county_names$long[county_names$subregion=="Franklin"] + 0.27

county_names$long[county_names$subregion=="Walla Walla"] <- county_names$long[county_names$subregion=="Walla Walla"] + 0.03
county_names$lat[county_names$subregion=="Walla Walla"] <- county_names$lat[county_names$subregion=="Walla Walla"] - 0.15

county_names$lat[county_names$subregion=="Chelan"] <- county_names$lat[county_names$subregion=="Chelan"] - 0.1

for (year in years){

  SF_name = paste0("Eastern_", year)
  double_name = paste0("doubleCroppedFields_", year,
                       "_onlyAnnuals_JustIrr_LastSrvFalse_NASSIn_SG_win7_Order3")

  SF <- readOGR(paste0(SF_dir, SF_name, "/", SF_name, ".shp"),
                layer = SF_name, 
                GDAL1_integer64_policy = TRUE)

  double_data <- data.table(read.csv(paste0(double_dir, double_name, ".csv")), as.is=TRUE)
  

  unique_IDs <- unique(double_data$ID)

  SF <- SF[SF@data$ID %in% unique_IDs, ]

  ####### Find Centroids
  SF_centers <- rgeos::gCentroid(SF, byid=TRUE)

  crs <- CRS("+proj=lcc 
             +lat_1=45.83333333333334 
             +lat_2=47.33333333333334 
             +lat_0=45.33333333333334 
             +lon_0=-120.5 +datum=WGS84")

  centroid_coord <- spTransform(SF_centers, 
                                CRS("+proj=longlat +datum=WGS84"))

  centroid_coord_dt <- data.table(centroid_coord@coords)
  setnames(centroid_coord_dt, old=c("x", "y"), new=c("long", "lat"))

  centroidMap <- centroid_coord_dt %>%
                 ggplot() +
                 geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                              fill = "grey", color = "black") +
                 geom_point(aes_string(x="long", y="lat"),  color="blue", size=0.7, alpha=0.4) +
                 geom_text(data=county_names, aes(long, lat, label=subregion), size=3) + # angle = 45
                 ggtitle(paste0("Predicted Double-Cropped Fields in ", year)) + 
                 geom_polygon(data=states_cluster, aes(long, lat, group=group), size=.2, fill=NA) + #colour = "grey60"
                 theme_bw() + 
                 theme(axis.title.y = element_blank(),
                       axis.title.x = element_blank(),
                       axis.ticks.y = element_blank(), 
                       axis.text.x = element_blank(),
                       axis.text.y = element_blank(),
                       axis.ticks.x = element_blank(),
                       panel.grid.major = element_line(size = 0.1),
                       panel.grid.minor = element_blank(),
                       strip.text = element_text(size=12, face="bold"),
                       plot.title = element_text(size=12, face = "bold",
                                                 margin = margin(t=.15, r=.1, b=0, l=0, "line")))

  ggsave(filename = paste0("map_", double_name, ".pdf"),
         plot = centroidMap, 
         width=plot_w, height=plot_h, units = "in", 
         dpi=400, device = "pdf",
         path=output_dir)
}
