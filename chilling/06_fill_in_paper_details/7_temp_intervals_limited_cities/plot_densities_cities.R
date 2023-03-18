#
# This is obtaied by copying and modifying "plot_densities.R"
# To plot densities for specific locations
#

#####################################################
###                                               ###
###             Sept. thru Apr.                   ###
###                                               ###
#####################################################
rm(list=ls())

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr) # for ggarrange
options(digit=9)
options(digits=9)

start_time <- Sys.time()


base_in <- "/data/hydro/users/Hossein/chill/7_time_intervals/"
data_dir <- file.path(base_in, "RDS_files/")
plot_dir <- file.path(base_in, "plots/AUC/consistent_colors")

                     
# data_dir <- file.path(paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
#                              "chilling/7_temp_int_limit_locs/untitled/RDS_files/"))
# plot_dir <- data_dir

if (dir.exists(file.path(plot_dir)) == F) {
  dir.create(path = plot_dir, recursive = T)
}

###########################################################

quality <- 500 # dpi quality
big_pic_width <- 70
big_pic_height <- 65
small_pic_width <- 50
small_pic_height <- 6 # 4

#####################################################
#####################################################
#####################################################

plot_dens <- function(data, month_name){
    color_ord = c("grey47", "dodgerblue", "olivedrab4", "khaki2")
    color_ord = c("grey47" , "dodgerblue", "olivedrab4", "red") #
    iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)
    x_breaks = c(-30, -20, -10, -2, 0, 4, 6, 8, 10, 13, 16, 20, 30, 40, 50)
    the_theme <- theme(plot.margin = unit(c(t=0.4, r=0.3, b=.3, l=0.1), "cm"),
                       panel.border = element_rect(fill=NA, size=.3),
                       panel.grid.major = element_line(size = 0.05),
                       panel.grid.minor = element_blank(),
                       panel.spacing=unit(.3, "cm"),
                       legend.position="bottom", 
                       legend.title = element_blank(),
                       legend.key.size = unit(.9, "line"),
                       legend.margin=margin(t= -0.1, r=0, b=0, l=0, unit='cm'),
                       legend.spacing.x = unit(.08, 'cm'),
                       legend.text=element_text(size=12),
                       strip.text.x = element_text(face="bold", size=12),
                       strip.text.y = element_text(face="bold", size=12),
                       axis.ticks = element_line(size=.2, color="black"),
                       plot.title = element_text(size=12, face="bold"),
                       axis.title.x = element_text(size=12, face = "bold", margin = margin(t=8, r=0, b=0, l=0)),
                       axis.title.y = element_text(size=12, face = "bold", margin = margin(t=0, r=8, b=0, l=0)),
                       axis.text.x = element_text(size =8, face = "plain", color="black"), # , angle=-90
                       axis.text.y = element_text(size =8, face = "plain", color="black")
                       )
    
    
    gtitle = paste0("The density of hourly temp. over chill season")
    
    obs_plot = ggplot(data, aes(x=Temp, fill=factor(time_period))) + 
               geom_density(alpha=.5, size=.1) + 
               geom_vline(xintercept = iof_breaks, 
                          linetype = "solid", color = "red", size = 0.2) +
               facet_grid( ~ scenario ~ city ) +
               xlab("temperature") + 
               ylab("hourly temp. density") +
               # ggtitle(label = gtitle) +
               scale_fill_manual(values=color_ord,
                                 name="Time\nPeriod", 
                                 labels=c("Historical", "2026-2050", "2051-2075", "2076-2099")) + 
               scale_color_manual(values=color_ord,
                                  name="Time\nPeriod", 
                                  limits = color_ord,
                                  labels=c("Historical", "2026-2050", "2051-2075", "2076-2099")) + 
               scale_x_continuous(name="temperature", breaks=x_breaks, limits=c(-30, 50)) + 
               the_theme
    return(obs_plot)
}

modeled  <- readRDS(paste0(data_dir, "/modeled.rds")) %>% data.table()

# toss out modeled historical
modeled <- modeled %>% filter(scenario != "historical")

modeled$scenario[modeled$scenario=="rcp45"] <- "RCP 4.5"
modeled$scenario[modeled$scenario=="rcp85"] <- "RCP 8.5"
print ("line 102")
print (sort(unique(modeled$year)))
modeled$time_period[modeled$year > 2026 & modeled$year <= 2051] <- "2026-2050"
modeled$time_period[modeled$year > 2051  & modeled$year <= 2076] <- "2051-2075"
modeled$time_period[modeled$year > 2076] <- "2076-2099"

print ("line 109")
print (sort(unique(modeled$year)))
print (sort(unique(modeled$time_period)))
observed <- readRDS(paste0(data_dir, "/observed.rds")) %>% data.table()
observed$time_period <- "Historical"
observed$scenario <- "Historical"
#####################################################################
###
###     If you want to facet by scenario as well, then we need
###     to have historical part for both scenarions, so we do 
###     the following
###
#####################################################################
observed_45 <- observed
observed_85 <- observed
observed_45$scenario <- "RCP 4.5"
observed_85$scenario <- "RCP 8.5"
observed <- rbind(observed_45, observed_85)

#####################################################################
#####################################################################

modeled <- rbind(modeled, observed)
modeled <- na.omit(modeled)
print ("line 119")
print (sort(unique(modeled$time_period)))

# order the climate groups
modeled$time_period <- factor(modeled$time_period, 
                              levels = c("Historical", "2026-2050", "2051-2075", "2076-2099"),
                              order=TRUE)

for (ct in unique(modeled$city)){
  dt_df_ct <- modeled %>% filter(city == ct)

  dt_df_45 <- dt_df_ct %>% filter(scenario %in% c("Historical", "RCP 4.5"))
  dt_df_85 <- dt_df_ct %>% filter(scenario %in% c("Historical", "RCP 8.5"))

  plot_85 <- plot_dens(data=dt_df_85)
  plot_45 <- plot_dens(data=dt_df_45)

  print ("line 134")
  print ("plot_85$time_period")
  print (plot_85$time_period)

  print ("plot_45$time_period")
  print (plot_45$time_period)

  ggsave(filename = paste0("sept_apr1_dens_", gsub(" ", "_", ct), "_rcp85.png"),
         path = plot_dir, 
         plot = plot_85,
         width = 6, height = 4, units = "in",
         dpi = quality, 
         device = "png",
         limitsize = FALSE)

  ggsave(filename = paste0("sept_apr1_dens_", gsub(" ", "_", ct), "_rcp45.png"),
         path = plot_dir, 
         plot = plot_45,
         width = 6, height = 4, units = "in",
         dpi = quality, 
         device = "png",
         limitsize = FALSE)
  print ("line 149")

}

end_time <- Sys.time()
print( end_time - start_time)
