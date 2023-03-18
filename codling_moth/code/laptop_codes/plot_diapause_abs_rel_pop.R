rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)

input_dir = "/Users/hn/Documents/01_research_data/codling_moth/overlaping/"
plot_path = "/Users/hn/Documents/01_research_data/codling_moth/overlaping/"


#### Relative

plot_rel_diapause <- function(input_dir, file_name_extension, version, plot_path){
  if (version == "rcp45") {plot_title <- "RCP 4.5"} else {plot_title <- "RCP 8.5"}
  file_name = paste0(input_dir, file_name_extension)
  data <- data.table(readRDS(file_name))
  data$CumulativeDDC =data$CumulativeDDF/1.8

  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  data <- data[variable =="RelLarvaPop" | variable =="RelNonDiap"]
  data <- subset(data, select=c("ClimateGroup", "CountyGroup", "CumulativeDDC", "variable", "value"))
  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)
  data$variable <- factor(data$variable)

  pp = ggplot(data, aes(x=CumulativeDDC, y=value, color=variable, fill=factor(variable))) + 
       labs(x = "cumulative degree days (in C)", y = "relative population", color = "relative population") +
       geom_vline(xintercept=c(213, 1153, 2313, 3443, 4453), linetype="solid", color ="grey", size=.25) +
       geom_hline(yintercept=c(5, 10, 15, 20), linetype="solid", color ="grey", size=.25) +
       annotate(geom="text", x=(700/1.8),  y=16, label="Gen. 1", color="black", angle=30, size = 5) +
       annotate(geom="text", x=(1700/1.8), y=14, label="Gen. 2", color="black", angle=30, size = 5) +
       annotate(geom="text", x=(2900/1.8), y=12, label="Gen. 3", color="black", angle=30, size = 5) +
       annotate(geom="text", x=(3920/1.8), y=10, label="Gen. 4", color="black", angle=30, size = 5) +
       facet_grid(. ~ ClimateGroup ~  CountyGroup, scales = "free") +
       scale_fill_manual(labels = c("total population", "population escaping diapause"), 
                         values=c("steelblue4", "orange"), 
                         name = "relative population") +
       scale_color_manual(labels = c("total population", "population escaping diapause"), 
                          values=c("steelblue4", "orange"), 
                          guide = FALSE) +
       stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                                   fun.ymin=function(z) { 0 }, 
                                   fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.7)+
       scale_x_continuous(limits = c(0, (max(data$CumulativeDDC)+100))) + 
       scale_y_continuous(limits = c(0, 20)) +
       theme(panel.grid.major = element_blank(),
             panel.grid.minor = element_blank(),
             panel.spacing = unit(.25, "cm"),
             legend.title = element_blank(),
             legend.spacing.x = unit(.2, 'cm'),
             legend.position = "bottom",
             legend.key.size = unit(.65, "cm"),
             axis.ticks = element_line(color = "black", size = .2),
             legend.text = element_text(size=20),
             strip.text = element_text(size=20, face="bold"),
             axis.text = element_text(face="bold", size=16, color="black"),
             plot.title = element_text(size=22, face="bold"),
             axis.title.x = element_text(face="bold", size=20, margin=margin(t=10, r=0, b=0, l=0), color="black"),
             axis.title.y = element_text(face="bold", size=20, margin=margin(t=0, r=10, b=0, l=0), color="black")
              ) +
       ggtitle(label = plot_title)

  # plot_name = paste0("diapause_rel_", version,".png")
  return(pp)
}

file_name_extension = "diapause_rel_data_rcp85.rds"
version = "rcp85"
rel_85 <- plot_rel_diapause(input_dir, file_name_extension, version, plot_path)

file_name_extension = "diapause_rel_data_rcp45.rds"
version = "rcp45"
rel_45 <- plot_rel_diapause(input_dir, file_name_extension, version, plot_path)

ggsave("rel_45.png", rel_45, device="png", path=plot_path, width=10, height=7, unit="in", dpi=400)
ggsave("rel_85.png", rel_85, device="png", path=plot_path, width=10, height=7, unit="in", dpi=400)


rel <- ggpubr::ggarrange(plotlist = list(rel_45, rel_85),
                         ncol = 2, nrow = 1,
                         common.legend = TRUE, legend = "bottom")
ggsave("rel.png", rel, device="png", path=plot_path, width=20, height=8, unit="in", dpi=350)


ggsave("rel_45.pdf", rel_45, device="pdf", path=plot_path, width=10, height=7, unit="in", dpi=400)
ggsave("rel_85.pdf", rel_85, device="pdf", path=plot_path, width=10, height=7, unit="in", dpi=400)
ggsave("rel.pdf", rel, device="pdf", path=plot_path, width=20, height=8, unit="in", dpi=350)

# # A <- annotate_figure(adult_emerge, 
# #                      top = text_grob("Julian day of adult emergence", color="black", face="bold", size=35)
# #                      # fig.lab = "Julian day of adult emergence", fig.lab.size = 35, fig.lab.face = "bold")
# #                      )
# #### Absolute
# # file_name_extension = "diapause_abs_rcp85.rds"
# # version = "rcp85"
# # plot_abs_diapause(input_dir, file_name_extension, version, plot_path)

# file_name_extension = "diapause_abs_rcp45.rds"
# version = "rcp45"
# plot_abs_diapause(input_dir, file_name_extension, version, plot_path)

# A <- annotate_figure(adult_emerge, 
#                      top = text_grob("Julian day of adult emergence", color="black", face="bold", size=35)
#                      # fig.lab = "Julian day of adult emergence", fig.lab.size = 35, fig.lab.face = "bold")
#                      )
