rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)

########################################################################################

plot_No_generations <- function(input_dir, file_name,
                                stage, dead_line,
                                box_width=.7, plot_with = 6.5, plot_height = 2.5,
                                plot_path, version,
                                color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
                                ){
  if (version=="rcp45"){ title <- "RCP 4.5" } else {title <- "RCP 8.5"}

  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)
  
  if (stage=="Larva"){ var = "NumLarvaGens" } else { var = "NumAdultGens" }

  data <- subset(data, select = c("ClimateGroup", "CountyGroup", var))
  ######
  ###### Compute medians of each group to annotate in the plot, if possible!!!
  ######
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(!!sym(var))))
  rm(df)
  y_lab = paste0("No. of generations")

  box_plot = ggplot(data = data, aes(x = ClimateGroup, y = !!sym(var), fill = ClimateGroup)) + 
             geom_boxplot(outlier.size=-.15, lwd=0.25, notch=F, width=box_width) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(0, 2), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(x="", y=y_lab, color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             theme(plot.title = element_text(size=12, face="bold"),
                   plot.margin = margin(t=.5, r=0.5, b=0, l=0.1, unit = 'cm'),
                   panel.grid.major = element_line(size = 0.1),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.5, "cm"),
                   legend.margin=margin(t=.5, r=0, b=0, l=0, unit='cm'),
                   legend.title = element_blank(),
                   legend.position="bottom", 
                   legend.key.size = unit(1.5, "line"),
                   legend.spacing.x = unit(.05, 'cm'),
                   legend.text=element_text(size=12),
                   strip.text = element_text(size=12, face = "bold"),
                   axis.text.x = element_text(size =10, face = "bold", color="black"),
                   axis.text.y  = element_blank(),
                   axis.title.y = element_text(face = "bold", size=8, margin=margin(t=0, r=1.5, b=0, l=0)),
                   axis.title.x = element_text(size=12, face = "bold",margin = margin(t=8, r=0, b=0, l=0)),
                   axis.ticks.y = element_blank()
                  ) +
              scale_fill_manual(values=color_ord, name="Time\nPeriod") + 
              scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord) + 
              coord_flip() +
              ggtitle(label = title)

  return(box_plot)
}


plot_No_generations_annotation <- function(input_dir, file_name,
                                stage, dead_line,
                                box_width=.7, plot_with = 6.5, plot_height = 2.5,
                                plot_path, version,
                                color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
                                ){
  if (version=="rcp45"){ title <- "RCP 4.5" } else {title <- "RCP 8.5"}

  file_name <- paste0(input_dir, file_name)
  data <- data.table(readRDS(file_name))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  
  old_ClimateGroup <- c("Historical", "2040's", "2060's", "2080's")
  new_ClimateGroup <- c("Historical", "2040s", "2060s", "2080s")
  
  data[data$ClimateGroup == old_ClimateGroup[2]]$ClimateGroup = new_ClimateGroup[2]
  data[data$ClimateGroup == old_ClimateGroup[3]]$ClimateGroup = new_ClimateGroup[3]
  data[data$ClimateGroup == old_ClimateGroup[4]]$ClimateGroup = new_ClimateGroup[4]
  data$ClimateGroup = factor(data$ClimateGroup, levels=new_ClimateGroup, order=T)
  
  if (stage=="Larva"){ var = "NumLarvaGens" } else { var = "NumAdultGens" }

  data <- subset(data, select = c("ClimateGroup", "CountyGroup", var))
  ######
  ###### Compute medians of each group to annotate in the plot, if possible!!!
  ######
  df <- data.frame(data)
  df <- (df %>% group_by(CountyGroup, ClimateGroup))
  medians <- (df %>% summarise(med = median(!!sym(var))))
  rm(df)
  y_lab = paste0("No. of generations")
  the_theme<-theme(plot.margin = margin(t=.5, r=0.5, b=0, l=0.1, unit = 'cm'),
                   panel.grid.major = element_line(size = 0.1),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.5, "cm"),
                   legend.margin=margin(t=.5, r=0, b=0, l=0, unit='cm'),
                   legend.title = element_blank(),
                   legend.position="bottom", 
                   legend.key.size = unit(1.5, "line"),
                   legend.spacing.x = unit(.05, 'cm'),
                   plot.title = element_text(size=12, face="bold"),
                   strip.text = element_text(size=12, face = "bold"),
                   legend.text=element_text(size=12),
                   axis.text.x = element_text(size =10, face = "bold", color="black"),
                   axis.text.y  = element_blank(),
                   axis.title.y = element_text(size=12, face = "bold", margin = margin(t=0, r=1.5, b=0, l=0)),
                   axis.title.x = element_text(size=12, face = "bold", margin = margin(t=8, r=0, b=0, l=0)),
                   axis.ticks.y = element_blank()
                  ) 
  box_plot = ggplot(data = data, aes(x = ClimateGroup, y = !!sym(var), fill = ClimateGroup)) + 
             geom_boxplot(outlier.size=-.15, notch=F, width=.4, lwd=.25) +
             # The bigger the number in expand below, the smaller the space between y-ticks
             scale_x_discrete(expand=c(-.2, 2), limits = levels(data$ClimateGroup[1])) +
             scale_y_continuous(limits = c(.5, 4), breaks=seq(1, 5, by=1)) + 
             labs(x="", y=y_lab, color = "Climate Group") +
             facet_wrap(~CountyGroup) +
             the_theme + 
             scale_fill_manual(values=color_ord, name="Time\nPeriod") + 
             scale_color_manual(values=color_ord, name="Time\nPeriod", limits = color_ord) + 
             coord_flip() +
             geom_text(data = medians, 
                aes(label = sprintf("%1.1f", medians$med), y=medians$med),
                    size= 3.2, vjust = -.8, # hjust = .5,
                    ) +
             ggtitle(label = title)

  return(box_plot)
}
########################################################################################
input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
plot_path = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
setwd(input_dir)

color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")

stages = c("Larva", "Adult")
dead_lines = c("Aug") # , "Nov"
versions = c("rcp45", "rcp85")

file_pref = "generations_" 
file_mid = "_combined_CMPOP_"
file_end = ".rds"

quality = 700
plot_with = 6.5
plot_height = 2.5

for (dead_line in dead_lines){
  for (version in versions){
    file_name = paste0(file_pref, dead_line, file_mid)
    file_name = paste0(file_name, version, file_end)
    for (stage in stages){
      p <- plot_No_generations(input_dir, file_name,
                               stage, dead_line = dead_line,
                               box_width=.7, plot_with = plot_with, 
                               plot_height= plot_height, plot_path,
                               version=version, color_ord)

      p_ann <- plot_No_generations_annotation(input_dir, file_name,
                                               stage, dead_line = dead_line,
                                               box_width=.7, plot_with = plot_with, 
                                               plot_height= plot_height, plot_path,
                                               version=version, color_ord)
      assign(x = paste0(stage, "_", dead_line, "_", version), value ={p})
      assign(x = paste0(stage, "_", dead_line, "_ann_", version), value ={p_ann})
            
        }
    }
}

larva <- ggpubr::ggarrange(plotlist = list(Larva_Aug_rcp45, Larva_Aug_rcp85),
                           ncol = 2, nrow = 1,
                           common.legend = TRUE, legend = "bottom")

larva_ann <- ggpubr::ggarrange(plotlist = list(Larva_Aug_ann_rcp45, Larva_Aug_ann_rcp85),
                               ncol = 2, nrow = 1,
                               common.legend = TRUE, legend = "bottom")

if (dead_line=="Aug"){
    plot_title <- paste0("Number of larval generations by Aug. 23")
  } else{
    plot_title <- paste0("Number of larval generations by Nov. 5")
}

A <- annotate_figure(larva,
                     top = text_grob(plot_title, color = "black", face = "bold", size = 15)
                     # fig.lab = plot_title, fig.lab.size = 35, fig.lab.face = "bold", fig.lab.pos = c("top.left")
                     )

A_ann <- annotate_figure(larva_ann,
                         top = text_grob(plot_title, color = "black", face = "bold", size = 15)
                         # fig.lab = plot_title, fig.lab.size = 35, fig.lab.face = "bold", fig.lab.pos = c("top.left")
                         )
plot_dpi <- 300
ggsave("larva_gen_aug.png", A, path="./", width=10, height=4, unit="in", dpi=quality)
ggsave("larva_gen_aug_ann.png", A_ann, path="./", width=10, height=4, unit="in", dpi=quality)


adult <- ggpubr::ggarrange(plotlist = list(Adult_Aug_rcp45, Adult_Aug_rcp85),
                           ncol = 2, nrow = 1,
                           common.legend = TRUE, legend = "bottom")

adult_ann <- ggpubr::ggarrange(plotlist = list(Adult_Aug_ann_rcp45, Adult_Aug_ann_rcp85),
                           ncol = 2, nrow = 1,
                           common.legend = TRUE, legend = "bottom")

if (dead_line=="Aug"){
    plot_title <- paste0("Number of adult generations by Aug. 23")
  } else{
    plot_title <- paste0("Number of adult generations by Nov. 5")
}

A <- annotate_figure(adult,
                     top = text_grob(plot_title, color = "black", face = "bold", size = 15)
                     # fig.lab = plot_title, fig.lab.size = 35, fig.lab.face = "bold", fig.lab.pos = c("top.left")
                     )

A_ann <- annotate_figure(adult_ann,
                         top = text_grob(plot_title, color = "black", face = "bold", size = 15)
                         # fig.lab = plot_title, fig.lab.size = 35, fig.lab.face = "bold", fig.lab.pos = c("top.left")
                         )
plot_dpi <- 300
ggsave("adult_gen_aug.png", A, path="./", width=10, height=4, unit="in", dpi=quality)
ggsave("adult_gen_aug_ann.png", A_ann, path="./", width=10, height=4, unit="in", dpi=quality)



