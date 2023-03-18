rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)

########################################################################################

plot_No_generations <- function(data, stage, dead_line,
                                box_width = 0.4, color_ord = c("grey47", "dodgerblue", "olivedrab4", "red"),
                                annotation = TRUE
                                ){

  time_period_L <- c("Historical", "2040s", "2060s", "2080s")
  data$time_period = factor(data$time_period, levels = time_period_L, order=T)
  data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)

  city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
  data$city <- factor(data$city, levels = city_levels, ordered = TRUE)
  
  if (stage=="Larva"){ var = "NumLarvaGens" } else { var = "NumAdultGens" }

  data <- subset(data, select = c("time_period", "city", "emission", var))
  ######
  ###### Compute medians of each group to annotate in the plot, if possible!!!
  ######
  df <- data.frame(data)
  df <- (df %>% group_by(city, time_period, emission))
  medians <- (df %>% summarise(med = median(!!sym(var))))
  rm(df)
  y_lab = paste0("No. of ", tolower(stage), " generations")

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
    box_plot = ggplot(data = data, aes(x = time_period, y = !!sym(var), fill = time_period)) + 
             geom_boxplot(outlier.size=-.15, notch=F, width = box_width, lwd=.25) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(-.2, 2), limits = levels(data$time_period[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(x="", y=y_lab, color = "Time Period") +
             facet_grid(~emission ~ city, scales="free") + 
             the_theme + 
             scale_fill_manual(values=color_ord, name="Time\nPeriod") + 
             scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord) + 
             coord_flip() +
             geom_text(data = medians, 
                       aes(label = sprintf("%1.1f", medians$med), y=medians$med),
                       size= 2.3, vjust = -1.2, # hjust = .5,
                    ) # +
             # ggtitle(label = title)
    } else {
      box_plot = ggplot(data = data, aes(x = time_period, y = !!sym(var), fill = time_period)) + 
             geom_boxplot(outlier.size = - 0.05, notch=F, width = box_width, lwd = 0.25) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(-.2, 2), limits = levels(data$time_period[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(x="", y=y_lab, color = "Time Period") +
             facet_grid(~emission ~ city, scales="free") + 
             the_theme + 
             scale_fill_manual(values=color_ord, name="Time\nPeriod") + 
             scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord) + 
             coord_flip()
  }

  return(box_plot)
}

########################################################################################
#
#     Directories
#
input_dir = "/Users/hn/Documents/01_research_data/codling_moth_2021/02_Diap_Gens_Vertdd/"
plot_path = "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper"

############################################################################################

color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")

stages = c("Larva", "Adult")
dead_lines = c("Aug") # , "Nov"
emissions = c("RCP 4.5", "RCP 8.5")

# file_pref = "generations_" 
# file_mid = "_combined_CMPOP_"

for (annot in c(TRUE, FALSE)){
  for (dead_line in dead_lines){
    file_name = paste0("generations_" , dead_line, ".rds")
    for (stage in stages){

      ############################################################################################
      #
      #   Read, clean up, and process the data for plot purposes
      #

      file_name = paste0("generations_" , dead_line, ".rds")
      data <- data.table(readRDS(paste0(input_dir, file_name)))

      data_modeled <- data %>% filter(emission %in% c("RCP 4.5", "RCP 8.5")) %>% data.table()
      data_obs <- data %>% filter(emission == "Observed") %>% data.table()
      data_obs$time_period <- "Historical"  # do this to avoid putting the date in plots!

      data_obs_45 <- data_obs
      data_obs_85 <- data_obs

      data_obs_45$emission <- "RCP 4.5"
      data_obs_85$emission <- "RCP 8.5"

      data <- data.table(rbind(data_modeled, data_obs_45, data_obs_85))
      ############################################################################################

      p <- plot_No_generations(data=data, stage=stage, dead_line = dead_line,
                               box_width = 0.4, color_ord=color_ord, 
                               annotation=annot)
      
      file_Name <- paste0("NoGen_", stage, "_by_", dead_line, "_", annot, "annot.png")

      plot_width = 9
      plot_height = 4
      ggsave(filename = file_Name, 
             path = plot_path, 
             plot = p,
             width = plot_width, height = plot_height, units = "in",
             dpi = 600, 
             device = "png")
            
        }
  }
}




