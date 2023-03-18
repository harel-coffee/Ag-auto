rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)


#
# These plots are produced by combined_CM files.
#

plot_adult_emergence <- function(data, box_width = 0.4, annotation=TRUE){

  color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")
  
  data <- subset(data, select = c("Emergence", "time_period", "model", "city", "emission"))

  data = data[, .(Emergence = Emergence), by = c("time_period", "city", "emission")]
  data <- subset(data, select = c("time_period", "city", "Emergence", "emission"))


  time_periods_L <- c("Historical", "2040s", "2060s", "2080s")
  data$time_period = factor(data$time_period, levels=time_periods_L, order=T)

  data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)

  city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
  data$city <- factor(data$city, levels = city_levels, ordered = TRUE)

  ###### Compute medians of each group to annotate in the plot, if possible!!!
  df <- data.frame(data)
  df <- (df %>% group_by(city, time_period, emission))
  medians <- (df %>% summarise(med = median(Emergence)))

  axis_text_size <- 8
  strip_text_size <- 8
  plot_title_font_size <- 10
  legent_font_size <- 8

  the_theme <- theme(plot.margin = unit(c(t = 0.2, r = 0.2, b = -0.2, l = 0.2), "cm"),
                     panel.border = element_rect(fill=NA, size=.3),
                     plot.title = element_text(size = plot_title_font_size, face = "bold"),
                     panel.grid.major = element_line(size = 0.05),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.25,"cm"),
                     legend.position="bottom", 
                     legend.title = element_blank(),
                     legend.key.size = unit(.8, "line"),
                     legend.text = element_text(size = legent_font_size),
                     legend.margin = margin(t= .1, r=0, b=.5, l=0, unit = 'cm'),
                     legend.spacing.x = unit(.05, 'cm'),
                     strip.text.x = element_text(size = strip_text_size, face="bold"),
                     strip.text.y = element_text(size = strip_text_size, face="bold"),
                     axis.ticks = element_line(color = "black", size = .2),
                     axis.ticks.x = element_blank(),
                     axis.text.x = element_text(size = axis_text_size, face = "plain", color="black"),
                     axis.text.y = element_text(size = axis_text_size, face="plain", color="black"),
                     axis.title.x = element_text(face = "bold", size = axis_text_size, margin = margin(t=6, r=0, b=0, l=0)),
                     axis.title.y = element_text(face = "bold", size = axis_text_size, margin = margin(t=0, r=6, b=0, l=0)),
                    )

  if (annotation == TRUE){
    p <- ggplot(data = data, aes(x=time_period, y=Emergence, fill=time_period))+
         geom_boxplot(outlier.size=-.15, notch=FALSE, width = box_width, lwd=.25) +
         scale_x_discrete(expand=c(-.2, 2), limits = levels(data$time_period[1])) +
         scale_y_continuous(breaks = round(seq(40, 170, by = 20), 1)) +
         labs(x="", y="day of year", color = "time_period") +
         facet_grid(~ emission ~ city, scales="free") + 
         the_theme +
         scale_fill_manual(values=color_ord, name="Time\nPeriod", labels=time_periods_L) + 
         scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord, labels=time_periods_L) +
         coord_flip() + 
         geom_text(data = medians, 
                       aes(label = sprintf("%1.1f", medians$med), y=medians$med),
                       size= 2.3, vjust = -1.2, # hjust = .5,
                    ) # + ggtitle(label = plot_title)
       } else {
        p <- ggplot(data = data, aes(x=time_period, y=Emergence, fill=time_period))+
         geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25) +
         scale_x_discrete(expand=c(-.2, 2), limits = levels(data$time_period[1])) +
         scale_y_continuous(breaks = round(seq(40, 170, by = 20), 1)) +
         labs(x="", y="day of year", color = "time_period") +
         facet_grid(~ emission ~ city, scales="free") + 
         the_theme +
         scale_fill_manual(values=color_ord, name="Time\nPeriod", labels=time_periods_L) + 
         scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord, labels=time_periods_L) +
         coord_flip()
  }
  return (p)
}


############################################################################################

data_dir = "/Users/hn/Documents/01_research_data/codling_moth_2021/01_combined_files/"
plot_path = "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/"

############################################################################################
#
#   Read, clean up, and process the data for plot purposes
#

CM_rcp45 <- readRDS(paste0(data_dir, "LM_combined_CM_rcp45.rds"))
CM_rcp85 <- readRDS(paste0(data_dir, "LM_combined_CM_rcp85.rds"))

CM_observed <- readRDS(paste0(data_dir, "LO_combined_CM_observed.rds"))
CM_observed$time_period <- "Historical"
# 
#  change emission in observed so that thet appear
#  in both emission scenario plots
#

CM_observed_45 <- CM_observed
CM_observed_85 <- CM_observed

CM_observed_45$emission <- "RCP 4.5"
CM_observed_85$emission <- "RCP 8.5"

data <- rbind(CM_rcp45, CM_rcp85, 
              CM_observed_45, CM_observed_85)

data$time_period <- as.character(data$time_period)
time_levels = c("Historical", "2040s", "2060s", "2080s")
data$time_period <- factor(data$time_period, levels = time_levels, ordered = TRUE)

data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)
A_dt = data
rm(data)
############################################################################################

for (annot in c(TRUE, FALSE)){
  p <- plot_adult_emergence(data = A_dt, box_width = 0.4, annotation=annot)
  file_Name <- paste0("adultEmergence_annot", annot, ".png")
  plot_width = 9
  plot_height = 4

  ggsave(filename = file_Name, 
         path = plot_path, 
         plot = p,
         width = plot_width, height = plot_height, units = "in",
         dpi = 600, 
         device = "png")

}
