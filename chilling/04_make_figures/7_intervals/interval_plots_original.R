.libPaths("/data/hydro/R_libs35")
.libPaths()

rm(list=ls())

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr) # for ggarrange
##############################
############################## Global variables
##############################
# data_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/7_temp_int_limit_locs/"
# input_dir<- data_dir
# plot_dir <- "/Users/hn/Desktop/"

base_in <- "/data/hydro/users/Hossein/chill/7_time_intervals/"
data_dir <- file.path(base_in, "RDS_files/")
plot_dir <- file.path(base_in, "plots/interval_TS/")

if (dir.exists(file.path(plot_dir)) == F) {
  dir.create(path = plot_dir, recursive = T)
}

quality <- 300 # dpi quality
big_pic_width <- 15
big_pic_height <- 100

small_pic_width <- 6
small_pic_height <- 14

# iof = interval of interest
iof = c(c(-Inf, -2),
        c(-2, 4),
        c(4, 6),
        c(6, 8),
        c(8, 13),
        c(13, 16),
        c(16, Inf))

iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

month_no = c(1, 2, 3, 9, 10, 11, 12)
month_names = c("Jan", "Feb", "Mar", "Sept", "Oct", "Nov", "Dec")

plot_intervals <- function(data, month_name){
  the_theme <- theme(plot.margin = unit(c(t=0.1, r=0.1, b=.5, l=0.1), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     panel.grid.major = element_line(size = 0.05),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.25,"cm"),
                     legend.position="bottom", 
                     legend.title = element_blank(),
                     legend.key.size = unit(1, "line"),
                     legend.text=element_text(size=9),
                     legend.margin=margin(t= -0.2, r = 0, b = 0, l = 0, unit = 'cm'),
                     legend.spacing.x = unit(.05, 'cm'),
                     plot.title = element_text(size=12, face="bold"),
                     plot.subtitle = element_text(size=10, face="bold"),
                     strip.text.x = element_text(size = 12, face="bold"),
                     strip.text.y = element_text(size = 12, face="bold"),
                     axis.ticks = element_line(color = "black", size = .2),
                     axis.title.x = element_text(face = "bold", size=12, margin = margin(t=6, r=0, b=0, l=0)),
                     axis.title.y = element_text(face = "bold", size = 12, margin = margin(t=0, r=6, b=0, l=0)),
                     axis.text.x = element_text(size = 8, face="bold", color="black", angle=-30),
                     axis.text.y = element_text(size = 8, face="bold", color="black")
                     )
  obs_plot = ggplot(data = data) +
             geom_point(aes(x = year, y = no_hours, fill = factor(scenario)),
                            alpha = 0.25, shape = 21, size = 1) +
             geom_smooth(aes(x = year, y = no_hours, color = factor(scenario)), method = "lm", se = F, size=.6) +
             facet_grid( ~ city ~ temp_cat) +
             scale_color_viridis_d(option = "plasma", begin = 0, end = .7,
                                   name = "Model scenario", 
                                   aesthetics = c("color", "fill")) +
             ylab("No. of hours in a given temp. interval") +
             xlab("year") +
             ggtitle(label = "No. of hours in Intervals of Interest (IoI)",
                     subtitle = month_name) +
             scale_fill_manual(values = c("yellow", "red", "blue")) +
             the_theme
  return(obs_plot)
}

for(month in month_names){
  data = paste0(data_dir, month, ".rds")
  data = data.table(readRDS(data))
  data$scenario[data$scenario == "historical"] = "Historical"
  data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
  data$scenario[data$scenario == "rcp85"] = "RCP 8.5"
  
  data <- data %>% 
          mutate(temp_cat = cut(Temp, breaks = iof_breaks)) %>% 
          group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
          summarise(no_hours = n()) %>%
          data.table()
  
  data$temp_cat = factor(data$temp_cat, order=T)
  
  assign(x = paste0(month, "_plot"),
         value = { plot_intervals(data=data,
                                  month_name=month)})

  data = paste0(data_dir, "observed_", month, ".rds")
  data = data.table(readRDS(data))
  data$scenario[data$scenario == "historical"] = "Historical"
  data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
  data$scenario[data$scenario == "rcp85"] = "RCP 8.5"
  
  data <- data %>% 
          mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
          group_by(chill_season, year, month, scenario, temp_cat, city) %>% 
          summarise(no_hours = n())
  data$temp_cat = factor(data$temp_cat, order=T)
  
  assign(x = paste0("observed_", month, "_plot"),
         value = { plot_intervals(data=data,
                                  month_name=month)})

} ## end for-loop

print ("line 122")

####################################################################
#################
#################   modeled
#################
####################################################################

big_plot <- ggarrange(Sept_plot, observed_Sept_plot,
                      Oct_plot, observed_Oct_plot,
                      Nov_plot, observed_Nov_plot,
                      Dec_plot, observed_Dec_plot,
                      Jan_plot, observed_Jan_plot,
                      Feb_plot, observed_Feb_plot,
                      Mar_plot, observed_Mar_plot,
                      label.x = "year",
                      label.y = "no. of hours in a given temp. interval",
                      ncol = 1, 
                      nrow = 14, 
                      common.legend = T,
                      legend = "bottom")

ggsave(plot = big_plot,
       filename = "7_intervals.png",
       path = plot_dir,
       width = big_pic_width, height = big_pic_height, units = "in",
       dpi=quality, 
       device = "png", 
       limitsize = FALSE)

ggsave(plot = Sept_plot, 
       filename = paste0("modeled_Sept", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, 
       limitsize = FALSE)

ggsave(plot = Oct_plot, 
       filename = paste0("modeled_Oct", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

print ("line 167")

ggsave(plot = Nov_plot, 
       filename = paste0("modeled_Nov", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = Dec_plot,
       filename = paste0("modeled_Dec", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = Jan_plot,
       filename = paste0("modeled_Jan", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = Feb_plot,
       filename = paste0("modeled_Feb", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = Mar_plot, 
       filename = paste0("modeled_Mar", ".png"), 
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, 
       units = "in", dpi=quality, limitsize = FALSE)

print ("line 203")

####################################################################
#################
#################   observed
#################
####################################################################

ggsave(plot = observed_Sept_plot,
       filename = paste0("observed_Sept", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = observed_Oct_plot,
       filename = paste0("observed_Oct", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = observed_Nov_plot,
       filename = paste0("observed_Nov", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = observed_Dec_plot, 
       filename = paste0("observed_Dec", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = observed_Jan_plot,
       filename = paste0("observed_Jan", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

ggsave(plot = observed_Feb_plot,
       filename = paste0("observed_Feb", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

print ("line 248")

ggsave(plot = observed_Mar_plot,
       filename = paste0("observed_Mar", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

####################################################################
#####                                                          #####
#####      observed and modeled months, neck to neck           #####
#####                                                          #####
####################################################################

sept_neck <- ggarrange(Sept_plot, observed_Sept_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = sept_neck,
       filename = paste0("mo_sept", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

oct_neck <- ggarrange(Oct_plot, observed_Oct_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = oct_neck,
       filename = paste0("mo_oct", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

print ("line 288")

nov_neck <- ggarrange(Nov_plot, observed_Nov_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = nov_neck,
       filename = paste0("mo_nov", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

dec_neck <- ggarrange(Dec_plot, observed_Dec_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")

print ("line 311")

ggsave(plot = dec_neck,
       filename = paste0("mo_dec", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

jan_neck <- ggarrange(Jan_plot, observed_Jan_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = jan_neck,
       filename = paste0("mo_jan", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

print ("line 332")

feb_neck <- ggarrange(Feb_plot, observed_Feb_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = feb_neck,
       filename = paste0("mo_feb", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

mar_neck <- ggarrange(Mar_plot, observed_Mar_plot,
                       label.x = "year",
                       label.y = "No. of hours in a given temp. interval",
                       ncol = 1, 
                       nrow = 2, 
                       common.legend = T,
                       legend = "bottom")
ggsave(plot = mar_neck,
       filename = paste0("mo_mar", ".png"),
       path = plot_dir,
       device = "png",
       height = 25, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)
                      
####################################################################
#####                                                          #####
#####                   Sept. Through Dec.                     #####
#####                                                          #####
####################################################################

# iof = interval of interest
iof = c(c(-Inf, -2), c(-2, 4), 
        c(4, 6), c(6, 8), 
        c(8, 13), c(13, 16), 
        c(16, Inf))
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

month_no = c(9, 10, 11, 12)
month_names = c("Sept", "Oct", "Nov", "Dec")
############
############ modeled
############
rm(data)
data <- data.table(readRDS(paste0(data_dir, "sept_thru_dec_modeled.rds")))
data$scenario[data$scenario == "historical"] = "Historical"
data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
data$scenario[data$scenario == "rcp85"] = "RCP 8.5"

data <- data %>% 
          mutate(temp_cat = cut(Temp, breaks = iof_breaks)) %>% 
          group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
          summarise(no_hours = n())

print ("line 389")

sep_Dec_plot <- plot_intervals(data, month_name="Sept. through Dec.")
ggsave(plot = sep_Dec_plot,
       filename = "sep_thru_Dec_modeled.png",
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)
rm(data)
############
############ observed
############

data <- data.table(readRDS(paste0(data_dir, "sept_thru_dec_observed.rds")))
data$scenario[data$scenario == "historical"] = "Historical"
data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
data$scenario[data$scenario == "rcp85"] = "RCP 8.5"

data <- data %>% 
          mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
          group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
          summarise(no_hours = n())

print ("line 412")

sep_Dec_plot <- plot_intervals(data, month_name="Sept. through Dec.")
ggsave(plot = sep_Dec_plot,
       filename = "sep_thru_Dec_observed.png",
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)
rm(data)
####################################################################
#####                                                          #####
#####                   Sept. Through Jan.                     #####
#####                                                          #####
####################################################################


# iof = interval of interest
iof = c(c(-Inf, -2), c(-2, 4), 
        c(4, 6), c(6, 8), 
        c(8, 13), c(13, 16), 
        c(16, Inf))
iof_breaks = c(-Inf, -2, 4, 6, 8, 13, 16, Inf)

month_no = c(9, 10, 11, 12, 1)
month_names = c("Sept", "Oct", "Nov", "Dec", "Jan")
############
############ modeled
############
data <- data.table(readRDS(paste0(data_dir, "sept_thru_jan_modeled.rds")))
data$scenario[data$scenario == "historical"] = "Historical"
data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
data$scenario[data$scenario == "rcp85"] = "RCP 8.5"

data <- data %>% 
          mutate(temp_cat=cut(Temp, breaks = iof_breaks)) %>% 
          group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
          summarise(no_hours = n())


print ("line 451")

sep_jan_plot <- plot_intervals(data, month_name = "Sept. through Jan.")
ggsave(plot = sep_jan_plot,
       filename = paste0("sep_thru_jan_modeled", ".png"),
       path = plot_dir,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)
rm(data)
############
############ observed
############
data <- data.table(readRDS(paste0(data_dir, "sept_thru_jan_observed.rds")))
data$scenario[data$scenario == "historical"] = "Historical"
data$scenario[data$scenario == "rcp45"] = "RCP 4.5"
data$scenario[data$scenario == "rcp85"] = "RCP 8.5"

data <- data %>% 
        mutate(temp_cat=cut(Temp, breaks=iof_breaks)) %>% 
        group_by(chill_season, year, model, month, scenario, temp_cat, city) %>% 
        summarise(no_hours = n())

sep_jan_plot <- plot_intervals(data, month_name="Sept. through Jan.")
ggsave(plot = sep_jan_plot,
       filename = paste0("sep_thru_jan_observed.png"),
       path = ,
       device = "png",
       height = small_pic_height, width = small_pic_width, units = "in", dpi=quality, limitsize = FALSE)

print ("line 480")

