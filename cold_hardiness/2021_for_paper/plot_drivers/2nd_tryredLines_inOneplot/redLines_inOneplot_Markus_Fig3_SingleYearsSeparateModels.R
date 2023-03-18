rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)


################################################################################
########################################
########################################
########################################
hardiness_core <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/2021_for_paper/hardiness_core.R"
hardiness_plot_core <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/2021_for_paper/hardiness_plot_core.R"

source(hardiness_core)
source(hardiness_plot_core)

########################################
########
########   directories
########
########################################
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/cold_hardiness/parameters/"

data_dir_main <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/daily_data/"
data_dir <- data_dir_main

output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/Markus_Figrue_3/redLines_inOneplot/separateYearsModels/"
if (!dir.exists(output_dir)) {
   dir.create(output_dir, recursive = T)
}

#
# proper DoY labeling
#
CH_DoY <- data.table(read.csv(paste0(param_dir, "CH_DoY_map.csv"), as.is=TRUE))
toDelete <- seq(0, nrow(CH_DoY), 2)
CH_DoY <-  CH_DoY[-toDelete, ]

grape_varieties <- c("cabernet_sauvignon", "chardonnay", "merlot", "reisling")

########################################
# CH_dt <- data.table::fread(paste0(data_dir, "all_cabernet_sauvignon.csv.gz"))
CH_dt2 <- data.table::fread(paste0(data_dir, "all_chardonnay.csv.gz"))
# CH_dt3 <- data.table::fread(paste0(data_dir, "all_merlot.csv.gz"))
# CH_dt4 <- data.table::fread(paste0(data_dir, "all_reisling.csv.gz"))
# CH_dt <- rbind(CH_dt, CH_dt2, CH_dt3, CH_dt4)
CH_dt <- CH_dt2
rm(CH_dt2, CH_dt3, CH_dt4)
#
# Put time period
#
CH_dt <- CH_dt %>%
         filter(hardiness_year %in% c(2025, 2030, 2035, 2040, 2045)) %>%
         data.table()

CH_dt <- CH_dt %>% filter(CH_DoY != 258)%>%
         data.table()


# CH_dt[CH_dt$emission == "Observed", "time_period"] <- "Observed"
# CH_dt <- CH_dt[!is.na(CH_dt$time_period),] # toss NA time_period (i.e. hardiness year 2-24)

# CH_dt = CH_dt[, .(median_tmin = median(t_min)), by = c("location", "year")]

CH_dt[CH_dt$emission=="rcp45", "emission"] <- "RCP 4.5"
CH_dt[CH_dt$emission=="rcp85", "emission"] <- "RCP 8.5"

CH_dt$emission <- factor(CH_dt$emission, 
                         levels =c("RCP 8.5", "RCP 4.5"), 
                          order=TRUE)

CH_dt$time_period <- factor(CH_dt$hardiness_year, 
                            levels =sort(unique(CH_dt$hardiness_year, )), 
                            order=TRUE)


CH_dt_median <- CH_dt %>%
                group_by(location, emission, hardiness_year, time_period, variety, CH_DoY)%>% 
                summarise(median_tmin=median(t_min), median_tmax=median(t_max), median_HC = median(predicted_Hc)) %>%
                data.table()


#
# Do the following so that weather data have the same length as each variety.
# t_max and t_min are repeated for each variety!
#
# weather <- CH_dt %>% filter(variety == "Merlot") %>% data.table()
line_size = 0.6
a_location <- unique(CH_dt$location)[1]
a_mission  <- unique(CH_dt$emission)[1]
for (a_location in unique(CH_dt$location)){
  for (a_mission in unique(CH_dt$emission)){

    curr_CH <- CH_dt %>% filter(location == a_location & emission == a_mission ) %>% data.table()
    curr_CH_median <- CH_dt_median %>% filter(location == a_location & emission == a_mission) %>% data.table()

    ## Copy the thing so it is in all facets.
    years_vc <- unique(curr_CH$time_period)
    count_vec <- 1:(length(years_vc)-1)
    for (count in count_vec){
      curr_1yr <- curr_CH %>% filter(hardiness_year == years_vc[count]) %>% data.table()
      curr_1yr_med <- curr_CH_median %>% filter(hardiness_year == years_vc[count]) %>% data.table()

      count_vec2 <- (count+1):(length(years_vc))
      for (count2 in count_vec2){

        curr_1yr$time_period <- years_vc[count2]
        curr_1yr_med$time_period <- years_vc[count2]

        #
        # Keep the tmin and tmax the same as original
        # 
        curr_1yr_helper     <- curr_CH        %>% filter(hardiness_year == years_vc[count2]) %>% data.table()
        curr_1yr_med_helper <- curr_CH_median %>% filter(hardiness_year == years_vc[count2]) %>% data.table()


        curr_1yr$t_max <- curr_1yr_helper$t_max
        curr_1yr$t_min <- curr_1yr_helper$t_min

        curr_1yr_med$median_tmax <- curr_1yr_med_helper$median_tmax
        curr_1yr_med$median_tmin <- curr_1yr_med_helper$median_tmin


        curr_CH <- rbind(curr_CH, curr_1yr)
        curr_CH_median <- rbind(curr_CH_median, curr_1yr_med)
      }
    }

    the_theme <- theme(panel.grid.major = element_line(size=0.2),
                       panel.spacing=unit(.5, "cm"),
                       legend.text=element_text(size=12, face="bold"),
                       legend.title = element_blank(),
                       legend.position = "top",
                       strip.text = element_text(face="plain", size=12, color="black"),
                       axis.text = element_text(size=10, color="black"), # face="bold",
                       axis.text.x = element_text(angle = -90),
                       # axis.text.x = element_text(angle=20, hjust = 1),
                       axis.ticks = element_line(color = "black", size = .2),
                       axis.title.x = element_blank(),
                       axis.title.y = element_blank(),
                       
                       # axis.title.y = element_text(size=12, face="bold",
                       #                             margin=margin(t=0, r=10, b=0, l=0)),
                       plot.title = element_text(lineheight=.8, face="bold", size=12)
                       )
    
    a_plt <- ggplot() +
             facet_grid(. ~ model ~ time_period, scales = "free") + 
             geom_ribbon(data=curr_CH, aes(x=CH_DoY, ymax=t_max, ymin=t_min), fill="dodgerblue2") + # , alpha=.2
             # geom_line(aes(x=CH_DoY, y=predicted_Hc, color = "Chardonnay"), size=line_size) + 
             
             # plot HC for 2025
             geom_line(data=curr_CH[curr_CH$hardiness_year==2025, ], aes(x=CH_DoY, y=predicted_Hc, color="2025"), size=line_size) + 

             # plot HC for 2030
             geom_line(data=curr_CH[curr_CH$hardiness_year==2030, ], aes(x=CH_DoY, y=predicted_Hc, color="2030"), size=line_size) + 

             # plot HC for 2035
             geom_line(data=curr_CH[curr_CH$hardiness_year==2035, ], aes(x=CH_DoY, y=predicted_Hc, color="2035"), size=line_size) + 

             # plot HC for 2040
             geom_line(data=curr_CH[curr_CH$hardiness_year==2040, ], aes(x=CH_DoY, y=predicted_Hc, color="2040"), size=line_size) + 

             # plot HC for 2045
             geom_line(data=curr_CH[curr_CH$hardiness_year==2045, ], aes(x=CH_DoY, y=predicted_Hc, color="2045"), size=line_size) + 

             scale_color_manual(values = c('2025' = 'darkgreen',
                                           '2030' = 'red',
                                           '2035' = 'orange',
                                           '2040' = 'slateblue1',
                                           '2045' = 'black')) +
                                labs(color = 'years') + 

             scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels= CH_DoY$letter_day) + 
             # labs(y = "predicted CH                          temperature") + 
             # ggtitle(paste0(unique(weather$location), ", ", unique(CH_dt$emission), ", ", unique(CH_dt$time_period))) +
             
             the_theme


    the_theme <- theme(panel.grid.major = element_line(size=0.2),
                       panel.spacing=unit(.5, "cm"),
                       legend.text=element_text(size=12, face="bold"),
                       legend.title = element_blank(),
                       legend.position = "top",
                       strip.text = element_text(face="plain", size=12, color="black"),
                       axis.text = element_text(size=6, color="black"), # face="bold",
                       # axis.text.x = element_text(angle=20, hjust = 1),
                       axis.ticks = element_line(color = "black", size = .2),
                       axis.title.x = element_blank(),
                       axis.title.y = element_blank(),
                       
                       # axis.title.y = element_text(size=12, face="bold",
                       #                             margin=margin(t=0, r=10, b=0, l=0)),
                       plot.title = element_text(lineheight=.8, face="bold", size=12)
                       )

    a_plt_med <- ggplot() +
                 facet_grid(. ~ time_period, scales = "free") + 
                 geom_ribbon(data=curr_CH_median, aes(x=CH_DoY, ymax=median_tmax, ymin=median_tmin), fill="dodgerblue2") +
                 
                 # plot HC for 2025
                 geom_line(data=curr_CH_median[curr_CH_median$hardiness_year==2025, ], aes(x=CH_DoY, y=median_HC, color="2025"), size=line_size) + 

                 # plot HC for 2030
                 geom_line(data=curr_CH_median[curr_CH_median$hardiness_year==2030, ], aes(x=CH_DoY, y=median_HC, color="2030"), size=line_size) + 

                 # plot HC for 2035
                 geom_line(data=curr_CH_median[curr_CH_median$hardiness_year==2035, ], aes(x=CH_DoY, y=median_HC, color="2035"), size=line_size) + 

                 # plot HC for 2040
                 geom_line(data=curr_CH_median[curr_CH_median$hardiness_year==2040, ], aes(x=CH_DoY, y=median_HC, color="2040"), size=line_size) + 

                 # plot HC for 2045
                 geom_line(data=curr_CH_median[curr_CH_median$hardiness_year==2045, ], aes(x=CH_DoY, y=median_HC, color="2045"), size=line_size) + 

                 scale_color_manual(values = c('2025' = 'darkgreen',
                                               '2030' = 'red',
                                               '2035' = 'orange',
                                               '2040' = 'slateblue1',
                                               '2045' = 'black')) +
                                    labs(color = 'years') + 

                 scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels= CH_DoY$letter_day) + 
                 the_theme



    ggsave(plot = a_plt,
           filename = paste0(gsub("\\.", "", gsub(" ", "", a_mission)), "_", a_location , ".pdf"),
           width = 12.5, height = 60, units = "in", 
           dpi = 400, device = "pdf",
           limitsize = FALSE,
           path = output_dir)

    ggsave(plot = a_plt_med,
           filename = paste0(gsub("\\.", "", gsub(" ", "", a_mission)), "_", a_location, "_med", ".pdf"),
           width = 12.5, height = 3, units = "in", 
           dpi = 400, device = "pdf",
           limitsize = FALSE,
           path = output_dir)

  }
}




