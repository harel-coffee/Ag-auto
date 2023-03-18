############################################################################################s
rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

################################################################################
################################################################################
################################################################################
plot_in <- paste0("/Users/hn/Documents/01_research_data/", 
                  "2021_doubleCroppedFieldDetection/01_idx/", 
                  "figures_just_2021/2018/double/")

plot_out <- paste0("/Users/hn/Documents/01_research_data/", 
                   "2021_doubleCroppedFieldDetection/01_idx/")

SF_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                 "2021_doubleCroppedFieldDetection/00_SFs/", 
                 "AdamsFrankBentonGrant_2018_IrrAnn/")


################################################################################
#
# list the plot names
#
plot_list <- list.files(path = plot_in, pattern = ".pdf")
noHarvest_plot_list <- list.files(path = paste0(plot_in, "/notHarvested/"), pattern = ".pdf")

plot_dt <- data.table(plot_list)
noHarvest_plot_dt <- data.table(noHarvest_plot_list)

x <- sapply(plot_dt$plot_list, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
ID_1 = x[2, ]
ID_2 = x[3, ]
ID_3 = x[4, ]
ID_4 = x[5, ]

plot_dt$ID <- paste(ID_1, ID_2, ID_3, ID_4, sep="_")
plot_dt$ID <- stringr::str_replace(plot_dt$ID, ".pdf", replacement="")



x <- sapply(noHarvest_plot_dt$noHarvest_plot_list, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
ID_1 = x[2, ]
ID_2 = x[3, ]
ID_3 = x[4, ]
ID_4 = x[5, ]

noHarvest_plot_dt$ID <- paste(ID_1, ID_2, ID_3, ID_4, sep="_")
noHarvest_plot_dt$ID <- stringr::str_replace(noHarvest_plot_dt$ID, ".pdf", replacement="")

plot_dt$harvest <- "1"
noHarvest_plot_dt$harvest <- "0"

setnames(noHarvest_plot_dt, old=c("noHarvest_plot_list"), new=c("plot_list"))
plot_dt <- rbind(plot_dt, noHarvest_plot_dt)
#
# read the shapefile
#
SF <- readOGR(paste0(SF_dir, "AdamsFrankBentonGrant_2018.shp"),
              layer = "AdamsFrankBentonGrant_2018", 
              GDAL1_integer64_policy = TRUE)

SF <- SF[SF@data$ID %in% plot_dt$ID, ]

SF@data <- within(SF@data, 
                  remove(CropTyp, Acres, Irrigtn, IntlSrD, 
                         LstSrvD, DataSrc, Notes, TRS, RtCrpTy, ExctAcr,
                         CropGrp, Shp_Lng, Shap_Ar))

plot_dt <- dplyr::left_join(x = plot_dt,
                            y = SF@data, 
                            by = "ID")


pullman_lat <- 46.7298
pullman_lon <- -117.1817
plot_dt <- rbind(plot_dt, list("Pullman", "Pullman", 0,"Whitman", pullman_lon, pullman_lat))
plot_dt$row <- rownames(plot_dt)

write.csv(plot_dt, 
          file=paste0(plot_out, "2fields_centroids.csv"), 
          row.names=FALSE)

library(ggmap)
library(maps)
library(qmap)
library(ggrepel)

states <- map_data("state")
states_cluster <- subset(states, region %in% c("washington"))




plot_dt %>% ggplot() +
            geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                         fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
           geom_point(aes_string(x = "cntrd_ln", y = "cntrd_lt"), color = "red", alpha = 0.4, size=1, label = plot_dt$row) +
           coord_fixed(xlim = c(-120.5, -117),  ylim = c(45.5, 48), ratio = 1.3) +
           # ggtitle(paste0("CP accumulation percentage differeces between projections and observed data")) +
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
                 plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                 )


aes(cntrd_ln, cntrd_lt, )

plot <- plot_dt %>% ggplot(aes(x = cntrd_ln, y = cntrd_lt)) +
                    geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                                 fill = "grey", color = "black") +
                    # aes_string to allow naming of column in function 
                    geom_point(alpha = 0.4, size=1) +
                    coord_fixed(xlim = c(-120.1, -117),  ylim = c(45.8, 47.5), ratio = 1.3) +
                    geom_text_repel(aes(cntrd_ln, cntrd_lt,label = row,group = row), 
                                    size = 2, box.padding = 0.1, point.padding = 4) +
    # geom_text(data = plot_dt, aes(x=cntrd_ln, y=cntrd_lt, group=row, label=row), size =2, hjust=0, vjust= -0.8) +
                    # geom_text(aes(label=row)) + 
                    # geom_text_repel() +
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
                          plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                          )
plot

qual = 400
W = 7.5
H = 5.5
ggsave(filename = paste0("repelled_text_missingText.png"), 
       plot=plot, 
       width=W, height=H, units="in", 
       dpi=qual, device="png", path=plot_out)


###############################################################################################
#####
##### Good below
#####
plot <- plot_dt %>% ggplot(aes(x = cntrd_ln, y = cntrd_lt)) +
                    geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                                 fill = "grey", color = "black") +
                    # aes_string to allow naming of column in function 
                    geom_point(alpha = 0.4, size=1) +
                    coord_fixed(xlim = c(-120.1, -117),  ylim = c(45.8, 47.5), ratio = 1.3) +
                    geom_text_repel(aes(cntrd_ln, cntrd_lt,label = row,group = row), 
                                    size = 2, box.padding = 0.05, point.padding = 0.05, max.overlaps=100) +
    # geom_text(data = plot_dt, aes(x=cntrd_ln, y=cntrd_lt, group=row, label=row), size =2, hjust=0, vjust= -0.8) +
                    # geom_text(aes(label=row)) + 
                    # geom_text_repel() +
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
                          plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm'))

ggsave(plot=plot, 
       path=plot_out,
       filename = paste0("repelled_text_missingTextmaxoverlaps100.png"), 
       width=W, height=H, units="in", dpi=qual, device="png")

#####
##### Good above
#####

ggmap(get_googlemap( c(-119.23, 46.62), zoom = 8, scale = 2, maptype ='terrain', color = 'color')) + 
ggplot(data=plot_dt, aes(x = cntrd_ln, y = cntrd_lt)) +
            geom_point(alpha = 0.4, size=1) +
            coord_fixed(xlim = c(-120.1, -117),  ylim = c(45.8, 47.5), ratio = 1.3) +
            geom_text_repel(aes(cntrd_ln, cntrd_lt,label = row,group = row), 
                            size = 2, box.padding = 0.05, point.padding = 0.05, max.overlaps=100) +

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
                  plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm'))



pp <- ggmap(get_googlemap( c(-119.23, 46.62), zoom = 3, scale = 2, maptype ='terrain', color = 'color')) + 
      geom_point(data=plot_dt, aes(x = cntrd_ln, y = cntrd_lt, color=harvest), alpha = 0.4, size=1) +
      coord_fixed(xlim = c(-120.1, -117.1),  ylim = c(45.8, 47.5), ratio = 1.3) +
      geom_text_repel(data=plot_dt, aes(cntrd_ln, cntrd_lt,label = row,group = row), 
                      size = 2, box.padding = 0.05, point.padding = 0.05, max.overlaps=100)

pp <- ggmap(get_googlemap( c(-119.23, 46.62), zoom = 8, scale = 2, maptype ='terrain', color = 'color')) + 
      geom_point(data=plot_dt, aes(x = cntrd_ln, y = cntrd_lt, color=harvest), alpha = 0.4, size=1) +
      coord_fixed(xlim = c(-120.1, -117.45),  ylim = c(45.8, 47.5), ratio = 1.3) +
      geom_text_repel(data=plot_dt, aes(cntrd_ln, cntrd_lt,label = row,group = row), 
                      size = 2, box.padding = 0.05, point.padding = 0.05, max.overlaps=100) + 
      theme(axis.title.y = element_blank(),
            axis.title.x = element_blank(),
            # axis.ticks.y = element_blank(), 
            # axis.ticks.x = element_blank(),
            # axis.text.x = element_blank(),
            # axis.text.y = element_blank(),
            # panel.grid.major = element_line(size = 0.1),
            legend.position="none", 
            # legend.title = element_blank(),
            # strip.text = element_text(size=12, face="bold"),
            plot.margin = margin(t=-0.5, r=0.2, b=0, l=-0.3, unit = 'cm')
            )

ggsave(plot=pp, 
       path=plot_out,
       filename = paste0("road_zoom_colorNoharvest.png"), 
       width=W, height=H, units="in", dpi=qual, device="png")
###############################################################################################

plot_dt %>% ggplot(aes(x = cntrd_ln, y = cntrd_lt)) +
            geom_polygon(data = states_cluster, aes(x=long, y=lat, group = group),
                         fill = "grey", color = "black") +
            # aes_string to allow naming of column in function 
            geom_point(alpha = 0.4, size=1) +
            coord_fixed(xlim = c(-120.1, -117),  ylim = c(45.8, 47.5), ratio = 1.3) +
            geom_text_repel(aes(cntrd_ln, cntrd_lt,label = row)) +
            # geom_text(data = plot_dt, aes(x=cntrd_ln, y=cntrd_lt, group=row, label=row), size =2, hjust=0, vjust= -0.8) +
            # geom_text(aes(label=row)) + 
            # geom_text_repel() +
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
                  plot.margin = margin(t=-0.5, r=0.2, b=-0.5, l=0.2, unit = 'cm')
                  )






