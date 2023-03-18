rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)
################################################################################################

input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
plot_path = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
setwd(input_dir)

#
# These plots are produced by combined_CM files.
#

plot_adult_emergence <- function(em){
  the_theme <- theme(plot.margin = margin(t=.5, r=0.5, b=0, l=0.1, unit = 'cm'),
                     panel.grid.major = element_line(size = 0.1),
                     panel.grid.minor = element_blank(),
                     panel.spacing=unit(.5, "cm"),
                     legend.margin=margin(t=.5, r=0, b=0, l=0, unit = 'cm'),
                     legend.title = element_blank(),
                     legend.position="bottom", 
                     legend.key.size = unit(1.5, "line"),
                     legend.spacing.x = unit(.05, 'cm'),
                     legend.text=element_text(size=12),
                     plot.title = element_text(size=12, face="bold"),
                     strip.text = element_text(size=12, face = "bold"),
                     axis.text.x = element_text(size= 10, face = "bold", color="black"),
                     axis.title.x = element_text(size=12, face = "bold",  margin = margin(t=8, r=0, b=0, l=0)),
                     axis.title.y = element_blank(),
                     axis.ticks.y = element_blank()
                     ) 
  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")

  color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")
  plot_dpi <- 700
  box_width = 0.5
  if (em=="rcp45") {plot_title <- "RCP 4.5"} else {plot_title <- "RCP 8.5"}
  print ("line 41")
  data <- data.table(readRDS(paste0("combined_CM_", em, ".rds")))
  print ("line 43")
  data <- subset(data, select = c("Emergence", "ClimateGroup", 
                                  "ClimateScenario", 
                                  "CountyGroup"))

  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

  data = data[, .(Emergence = Emergence), by = c("ClimateGroup", "CountyGroup")]
  data <- subset(data, select = c("ClimateGroup", "CountyGroup", "Emergence"))

  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)

  ###### Compute medians of each group to annotate in the plot, if possible!!!
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(Emergence)))

  p <- ggplot(data = data, aes(x=ClimateGroup, y=Emergence, fill=ClimateGroup))+
       geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25) +
       scale_x_discrete(expand=c(-.2, 2), limits = levels(data$ClimateGroup[1])) +
       scale_y_continuous(breaks = round(seq(40, 170, by = 20), 1)) +
       labs(x="Time period", y="Julian day", color = "Climate Group") +
       facet_wrap(~CountyGroup) +
       the_theme +
       scale_fill_manual(values=color_ord, name="Time\nPeriod", labels=new_ClimateGroup) + 
       scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord, labels=new_ClimateGroup) +
       coord_flip() + 
       geom_text(data = medians, 
                 aes(label = sprintf("%1.0f", medians$med), y=medians$med),
                     size= 3.2, vjust = -.8, # hjust = .5,
                    ) +
       ggtitle(label = plot_title)
  return (p)
}

adult_emerge_rcp45 <- plot_adult_emergence(em="rcp45")
adult_emerge_rcp85 <- plot_adult_emergence(em="rcp85")
adult_emerge <- ggpubr::ggarrange(plotlist = list(adult_emerge_rcp45, adult_emerge_rcp85),
                                  ncol = 2, nrow = 1,
                                  common.legend=T, legend = "bottom")

A <- annotate_figure(adult_emerge, 
                     top = text_grob("Julian day of adult emergence", color="black", face="bold", size=15)
                     # fig.lab = "Julian day of adult emergence", fig.lab.size = 35, fig.lab.face = "bold")
                     )
ggsave("adult_emerge.png", A, path="./", width=10, height=4, unit="in", dpi=600)


