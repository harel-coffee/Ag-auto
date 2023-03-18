rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)
library(purrr)
library(scales) # includes pretty_breaks

##############################################################################################################
#
#    Directory setup
#
data_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/cab_chard_w_Wyoming/"
file_names <- list.files(path = data_dir, pattern = ".csv")

output_dir <- paste0("/Users/hn/Documents/01_research_data/cold_hardiness/plots4Paper/Kirti_models_with_increase_CDI/")
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}


##############################################################################################################
#
#    Parameter setup
#
plotColumns <- c("CDI")
plot_w=16
plot_h=8

##############################################################################################################
#
#    Body
#
states <- map_data("state")

for (fname in file_names){
  data <- read.csv(paste0(data_dir, fname), as.is=TRUE) %>% data.table()

  data <- data.table(within(data, remove("X", "X.1")))
  data <- data.table(unique(data))

  setnames(data, old=c("range",       "scenario"), 
                 new=c("time_period", "emission"))

  grape_variety <- unique(data$variety)

  if (grape_variety %in% c("Merlot", "Riesling")){
     states_cluster <- subset(states, region %in% c("washington"))
     # plot_w=8
     # plot_h=4
     } else{
     states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho", "montana", "wyoming"))
     # plot_w=16
     # plot_h=8
   }

  data$emission[data$emission=="rcp45"] <- "RCP 4.5"
  data$emission[data$emission=="rcp85"] <- "RCP 8.5"
  data$emission <- factor(data$emission, levels=c("RCP 8.5", "RCP 4.5"), ordered=TRUE)

  historical <- data %>% filter(!(emission %in% c("RCP 8.5", "RCP 4.5")))
  future <- data %>% filter((emission %in% c("RCP 8.5", "RCP 4.5")))

  historical$model <- "Observed"
  historical$emission <- "RCP 8.5"
  data <- rbind(future, historical)

  historical$emission <- "RCP 4.5"
  data <- rbind(data, historical)

  time_periods <- sort(unique(data$time_period))
  data$time_period <- factor(data$time_period, levels=time_periods, ordered=TRUE)

  for (vari in plotColumns){

    no_of_models <- length(unique(future$model))

    thresh_25 <- floor(no_of_models/4)
    thresh_50 <- floor(no_of_models/2)
    thresh_75 <- floor(no_of_models * (3/4))
  	############################################################################################
    ####
    #### 25%, 50%, and 75% of models show increase,
    ####
                
    positive_count <- future %>% 
                      group_by(location, lat, long, emission, time_period) %>% 
                      summarise(num_pos = sum(CDI > 0)) %>%
                      data.table()
    

    future_25 <- positive_count %>% 
                 filter(num_pos >= thresh_25) %>%
                 data.table()
    
    future_50 <- positive_count %>% 
                 filter(num_pos >= thresh_50) %>%
                 data.table()

    future_75 <- positive_count %>% 
                 filter(num_pos >= thresh_75) %>%
                 data.table()

    currMap <- future_25 %>%
               ggplot() +
               geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                            fill = "grey", color = "black") +
                # aes_string to allow naming of column in function 
               geom_point(aes_string(x="long", y="lat", color="num_pos"), alpha=0.4) +
               scale_color_viridis_c(option="plasma", name="", direction=-1,
                                     # limits = c(min, max),
                                     breaks = pretty_breaks(n=6)) +
               # coord_fixed(xlim=c(-124.5, -107),  ylim=c(42, 50.5), ratio=1.3) +
               facet_grid(. ~ emission ~ time_period) +
               theme(axis.title.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.ticks.y = element_blank(), 
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     strip.text = element_text(size=12, face="bold"))

    ggsave(filename = paste(grape_variety, vari, "25_Percent_Of_Models.pdf", sep="_"),
           plot = currMap              , 
           width = plot_w, height=plot_h, units = "in", 
           dpi = 400, device = "pdf",
           path = output_dir)

    currMap <- future_50 %>%
               ggplot() +
               geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                            fill = "grey", color = "black") +
                # aes_string to allow naming of column in function 
               geom_point(aes_string(x="long", y="lat", color="num_pos"), alpha=0.4) +
               scale_color_viridis_c(option="plasma", name="", direction=-1,
                                     # limits = c(min, max),
                                     breaks = pretty_breaks(n=6)) +
               # coord_fixed(xlim=c(-124.5, -107),  ylim=c(42, 50.5), ratio=1.3) +
               facet_grid(. ~ emission ~ time_period) +
               theme(axis.title.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.ticks.y = element_blank(), 
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     strip.text = element_text(size=12, face="bold"))

    ggsave(filename = paste(grape_variety, vari, "50_Percent_Of_Models.pdf", sep="_"),
           plot = currMap              , 
           width = plot_w, height=plot_h, units = "in", 
           dpi = 400, device = "pdf",
           path = output_dir)

    currMap <- future_75 %>%
               ggplot() +
               geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                            fill = "grey", color = "black") +
                # aes_string to allow naming of column in function 
               geom_point(aes_string(x="long", y="lat", color="num_pos"), alpha=0.4) +
               scale_color_viridis_c(option="plasma", name="", direction=-1,
                                     # limits = c(min, max),
                                     breaks = pretty_breaks(n=6)) +
               # coord_fixed(xlim=c(-124.5, -107),  ylim=c(42, 50.5), ratio=1.3) +
               facet_grid(. ~ emission ~ time_period) +
               theme(axis.title.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.ticks.y = element_blank(), 
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     strip.text = element_text(size=12, face="bold"))

    ggsave(filename = paste(grape_variety, vari, "75_Percent_Of_Models.pdf", sep="_"),
           plot = currMap              , 
           width = plot_w, height=plot_h, units = "in", 
           dpi = 400, device = "pdf",
           path = output_dir)

    currMap <- positive_count %>%
               ggplot() +
               geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                            fill = "grey", color = "black") +
                # aes_string to allow naming of column in function 
               geom_point(aes_string(x="long", y="lat", color="num_pos"), alpha=0.4) +
               scale_color_viridis_c(option="plasma", name="", direction=-1,
                                     # limits = c(min, max),
                                     breaks = pretty_breaks(n=7)) +
               # coord_fixed(xlim=c(-124.5, -107),  ylim=c(42, 50.5), ratio=1.3) +
               facet_grid(. ~ emission ~ time_period) +
               theme(axis.title.y = element_blank(),
                     axis.title.x = element_blank(),
                     axis.ticks.y = element_blank(), 
                     axis.text.x = element_blank(),
                     axis.text.y = element_blank(),
                     axis.ticks.x = element_blank(),
                     strip.text = element_text(size=12, face="bold"))

    ggsave(filename = paste(grape_variety, vari, "noPositiveModels.pdf", sep="_"),
           plot = currMap              , 
           width = plot_w, height=plot_h, units = "in", 
           dpi = 400, device = "pdf",
           path = output_dir)

  }
}



