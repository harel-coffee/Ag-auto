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

output_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/Markus_Figrue_3/"
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


CH_dt <- CH_dt %>% filter(model %in% c("Observed", "bcc-csm1-1")) %>% data.table()
CH_dt[CH_dt$emission=="rcp45", "emission"] <- "RCP 4.5"
CH_dt[CH_dt$emission=="rcp85", "emission"] <- "RCP 8.5"

#
# Put time period
#
CH_dt_hist <- CH_dt %>%
              filter(hardiness_year %in% c(1980:2015))

CH_dt_hist_45 <- CH_dt_hist
CH_dt_hist_85 <- CH_dt_hist
CH_dt_hist_45$emission <- "RCP 4.5"
CH_dt_hist_85$emission <- "RCP 8.5"
CH_dt_hist <- rbind(CH_dt_hist_45, CH_dt_hist_85)


CH_dt_F <- CH_dt %>%
           filter(hardiness_year %in% c(2025:2049))


col = 'predicted_Hc'
names(CH_dt_F)[names(CH_dt_F) == col] <- 'val'

CH_dt_merged <- merge(CH_dt_F[c('model', 'location', 'val', 'CH_DoY', 'emission')], 
                      CH_dt_hist[c('location', 'CH_DoY', 'emission', col)], 
                      all.x=T, by=c('location', 'CH_DoY', 'emission'))

CH_dt_merged$diff <- CH_dt_merged$val - CH_dt_merged[, col]

CH_dt_merged <- CH_dt_merged %>%
                group_by(location, emission, CH_DoY)%>% 
                summarise(median_HC_diffs = mean(diff)) %>%
                data.table()


# CH_dt <- CH_dt[!is.na(CH_dt$time_period),] # toss NA time_period (i.e. hardiness year 2-24)
# CH_dt = CH_dt[, .(median_tmin = median(t_min)), by = c("location", "year")]

CH_dt_merged$emission <- factor(CH_dt_merged$emission, 
                               levels =c("RCP 8.5", "RCP 4.5"), 
                               order=TRUE)


CH_dt_merged$location <- factor(CH_dt_merged$location, 
                                levels = sort(unique(CH_dt_merged$location)), 
                                order=TRUE)

#
# Do the following so that weather data have the same length as each variety.
# t_max and t_min are repeated for each variety!
#
# weather <- CH_dt %>% filter(variety == "Merlot") %>% data.table()
line_size = 0.6


the_theme <- theme(panel.grid.major = element_line(size=0.2),
                   panel.spacing=unit(.5, "cm"),
                   legend.text=element_text(size=12, face="bold"),
                   legend.title = element_blank(),
                   legend.position = "bottom",
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

a_plt <- ggplot(data=CH_dt_merged) +
         # geom_ribbon(aes(x=CH_DoY, ymax=median_tmax, ymin=median_tmin), fill="dodgerblue2") + # , alpha=.2
         geom_line(aes(x=CH_DoY, y=median_HC_diffs, color = "Chardonnay"), size=line_size) + 
         scale_color_manual(values = c('Chardonnay' = 'red')) +
                            labs(color = 'grape variety') + 
         scale_x_continuous(breaks = CH_DoY$day_count_since_sept, labels= CH_DoY$letter_day) + 
         # labs(y = "predicted CH                          temperature") + 
         # ggtitle(paste0(unique(weather$location), ", ", unique(CH_dt$emission), ", ", unique(CH_dt$time_period))) +
         facet_grid(. ~ emission ~ location, scale="free") +
         the_theme


ggsave(plot = a_plt,
       filename = paste0("diffBased_BCC_predictedHC_mean.pdf"),
       width=25, height=6, units = "in", 
       dpi = 400, device = "pdf",
       limitsize = FALSE,
       path = output_dir)





