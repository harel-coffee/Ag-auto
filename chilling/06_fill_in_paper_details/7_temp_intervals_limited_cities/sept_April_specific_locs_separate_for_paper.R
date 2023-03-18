##
## This code is made by copying "interval_plots.R" in order
## to produce nice plots to go inside the paper (March 2, 2020).
## 
##
##

rm(list=ls())

base_in <- "/data/hydro/users/Hossein/chill/7_time_intervals/"
input_dir <- file.path(base_in, "RDS_files/")
plot_dir <- file.path(base_in, "plots/interval_TS/sept_thru_apr/")
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr) # for ggarrange
options(digit=9)
options(digits=9)
##############################
############################## Global variables
##############################

# input_dir <- "/Users/hn/Documents/01_research_data/chilling/7_temp_int_limit_locs/untitled/RDS_files/"
# plot_dir <- input_dir
# param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

if (dir.exists(file.path(plot_dir)) == F) {
  dir.create(path = plot_dir, recursive = T)
}

quality <- 500 # dpi quality
big_pic_width <- 15
big_pic_height <- 65
small_pic_width <- 6
small_pic_height <- 8


plot_intervals <- function(data, month_name){
  data$temp_cat <- as.character(data$temp_cat)
  data$temp_cat[data$temp_cat == "(-Inf,-2]"] <- "<= -2"
  data$temp_cat[data$temp_cat == "(16, Inf]"] <- "> 16"
  
  temp_cat_ord <- c("<= -2", "(-2,4]", "(4,6]", "(6,8]", "(8,13]", "(13,16]", "> 16")
  data$temp_cat <- factor(data$temp_cat, order=T, levels = temp_cat_ord)

  the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.1, b=.5, l=0.1), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     # plot.title = element_text(hjust = 0.5),
                     plot.subtitle = element_text(hjust = 0.5),
                     panel.grid.major = element_line(size = 0.05),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.25,"cm"),
                     legend.position="bottom", 
                     legend.title = element_blank(),
                     legend.key.size = unit(1, "line"),
                     legend.text=element_text(size=9),
                     legend.margin=margin(t= -0.2, r = 0, b = 0, l = 0, unit = 'cm'),
                     legend.spacing.x = unit(.05, 'cm'),
                     strip.text = element_text(size=12, face = "bold"),
                     axis.ticks = element_line(color = "black", size = .2),
                     axis.title.x = element_text(face = "bold", size=12, 
                                                 margin = margin(t=6, r=0, b=0, l=0)),
                     axis.text.x = element_text(size = 8, face = "bold", 
                                                color="black"), # , angle=-30
                     axis.title.y = element_text(face = "bold", size = 12, 
                                                 margin = margin(t=0, r=6, b=0, l=0)),
                     axis.text.y = element_text(size = 8, face="bold", color="black")
                     )
  obs_plot = ggplot(data = data) +
             geom_point(aes(x = year, y = no_hours, fill = factor(scenario)),
                            alpha = 0.25, shape = 21, size = 1) +
             geom_smooth(aes(x = year, y = no_hours, color = factor(scenario)),
                             method = "lm", se = F, size=.4) +
             facet_grid(~ temp_cat ~ city, scales = "free") +
             scale_color_viridis_d(option = "plasma", begin = 0, end = .7,
                                   name = "Model scenario", 
                                   aesthetics = c("color", "fill")) +
             ylab("hour count") +
             xlab("year") +
             # labs (title = "No. of hours in Intervals of Interest (IoI)",
             #       subtitle = month_name) + 
             # ggtitle(label = "No. of hours in Intervals of Interest (IoI)",
             #         subtitle = month_name) +
             scale_fill_manual(values = c("yellow", "red", "blue")) +
             the_theme

  return(obs_plot)
}


limited_locations <- read.csv(paste0(param_dir, "limited_locations.csv"), header=T, sep=",")

limited_locations$location = paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))
city_names <- limited_locations$city


# iof = interval of interest
iof = c(c(-Inf, -2), c(-2, 4), 
        c(4, 6), c(6, 8), 
        c(8, 13), c(13, 16), 
        c(16, Inf))
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

####################################################################
#####                                                          #####
#####                   Sept. Through Apr.                     #####
#####                                                          #####
####################################################################


############
############ modeled
############
modeled  <- readRDS(paste0(input_dir, "/modeled.rds")) %>% data.table()

# toss out modeled historical
modeled <- modeled %>% filter(scenario != "historical")

modeled$scenario[modeled$scenario=="rcp45"] <- "RCP 4.5"
modeled$scenario[modeled$scenario=="rcp85"] <- "RCP 8.5"

observed <- readRDS(paste0(input_dir, "/observed.rds")) %>% data.table()
observed$scenario <- "Historical"

dt_df <- rbind(modeled, observed)
rm(modeled, observed)

# dt_df <- table(cut(dt_df$Temp, breaks = iof_breaks))

dt_df <- dt_df %>% 
         mutate(temp_cat = cut(Temp, breaks = iof_breaks)) %>% 
         group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
         summarise(no_hours = n()) %>%
         data.table()

cities <- c("Omak", "Wenatchee", "Richland", "Hillsboro", "Hood River", "Corvallis", 
            "Eugene", "Salem", "Walla Walla",  "Yakima")

for (ct in cities){
  A <- dt_df %>% filter(city == ct) %>% data.table()
  sep_apr_plot <- plot_intervals(data=A, month_name = "Sept. through Apr.")
  ggsave(plot = sep_apr_plot,
         filename = paste0("sep_thru_Apr_modeled_", gsub(" ", "_", ct), ".png"),
         path = plot_dir,
         device = "png",
         height = 6, width = 4, 
         units = "in", dpi=quality, limitsize = FALSE)
}



############
############ observed
############

# data <- data.table(readRDS(paste0(input_dir, "sept_thru_jan_observed.rds")))

# data <- table(cut(data$Temp, breaks = iof_breaks))
# sep_jan_plot <- plot_intervals(data, month_name="Sept. through Jan.")
# ggsave(plot = sep_jan_plot,
#        filename = paste0("sep_thru_jan_observed.png"),
#        path = plot_dir,
#        device = "png",
#        height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)


