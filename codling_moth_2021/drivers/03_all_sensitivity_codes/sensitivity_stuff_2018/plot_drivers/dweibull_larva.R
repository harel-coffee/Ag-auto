################## Larva dweibull
library(data.table)
library(dplyr)
library(ggpubr)
par(mar=c(1, 1, 1, 1))
dev.off() 

param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
param_name = "CodlingMothparameters.txt"
params = read.table(paste0(param_dir, param_name), header=TRUE, sep=",")

param_name_shift_10_percent = "CodlingMothparameters_0.1.txt"
params_shift_10_percent = read.table(paste0(param_dir, param_name_shift_10_percent), header=TRUE, sep=",")

param_name_shift_20_percent <- "CodlingMothparameters_0.2.txt"
params_shift_20_percent = read.table(paste0(param_dir, param_name_shift_20_percent), header=TRUE, sep=",")

########################################################################
########################################################################
########################## Functions
##########################
dw_larva_Gen1 <- function(x, shape=params[5, "shape"], scale=params[5, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen2 <- function(x, shape=params[6, "shape"], scale=params[6, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen3 <- function(x, shape=params[7, "shape"], scale=params[7, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen4 <- function(x, shape=params[8, "shape"], scale=params[8, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}
##################################################
#########################
######################### S H I F T 10 %
#########################
##################################################
dw_larva_Gen1_shift_10_percent <- function(x, shape=params_shift_10_percent[5, "shape"], 
                                              scale=params_shift_10_percent[5, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen2_shift_10_percent <- function(x, shape=params_shift_10_percent[6, "shape"], 
                                              scale=params_shift_10_percent[6, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen3_shift_10_percent <- function(x, shape=params_shift_10_percent[7, "shape"], 
                                              scale=params_shift_10_percent[7, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen4_shift_10_percent <- function(x, shape=params_shift_10_percent[8, "shape"], 
                                              scale=params_shift_10_percent[8, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

##################################################
#########################
######################### S H I F T 20 %
#########################
##################################################
dw_larva_Gen1_shift_20_percent <- function(x, shape=params_shift_20_percent[5, "shape"], 
                                              scale=params_shift_20_percent[5, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen2_shift_20_percent <- function(x, shape=params_shift_20_percent[6, "shape"], 
                                              scale=params_shift_20_percent[6, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen3_shift_20_percent <- function(x, shape=params_shift_20_percent[7, "shape"], 
                                              scale=params_shift_20_percent[7, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

dw_larva_Gen4_shift_20_percent <- function(x, shape=params_shift_20_percent[8, "shape"], 
                                              scale=params_shift_20_percent[8, "scale"]){
  dweibull(x, shape=shape, scale=scale)
}

########################################################################
########################################################################
########################## Accumulative
##########################
pw_larva_1 <- function(x, shape=params[5, "shape"], scale=params[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_2 <- function(x, shape=params[6, "shape"], scale=params[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_3 <- function(x, shape=params[7, "shape"], scale=params[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_4 <- function(x, shape=params[8, "shape"], scale=params[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
##################################################
#########################
######################### S H I F T 10 %
#########################
##################################################
pw_larva_1_shift_10_percent <- function(x, shape=params_shift_10_percent[5, "shape"], 
                                         scale=params_shift_10_percent[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_2_shift_10_percent <- function(x, shape=params_shift_10_percent[6, "shape"], 
                                         scale=params_shift_10_percent[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_3_shift_10_percent <- function(x, shape=params_shift_10_percent[7, "shape"], 
                                           scale=params_shift_10_percent[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_4_shift_10_percent <- function(x, shape=params_shift_10_percent[8, "shape"], 
                                         scale=params_shift_10_percent[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

##################################################
#########################
######################### S H I F T 20 %
#########################
##################################################
pw_larva_1_shift_20_percent <- function(x, shape=params_shift_20_percent[5, "shape"], 
                                         scale=params_shift_20_percent[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_2_shift_20_percent <- function(x, shape=params_shift_20_percent[6, "shape"], 
                                         scale=params_shift_20_percent[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_3_shift_20_percent <- function(x, shape=params_shift_20_percent[7, "shape"], 
                                         scale=params_shift_20_percent[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_4_shift_20_percent <- function(x, shape=params_shift_20_percent[8, "shape"], 
                                         scale=params_shift_20_percent[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
########################## 
########################## Functions
########################################################################
########################################################################
########################## The Theme
##########################
the_theme = theme(panel.grid.major = element_blank(),
                  plot.title = element_text(size=18, face="bold"),
                  plot.subtitle = element_text(size=14, face="bold"),
                  axis.text.x = element_text(size = 15, color="black", face="bold"),
                  axis.text.y = element_text(size = 15, angle=0, color="black", face="bold"),
                  
                  axis.title.x = element_text(face = "bold", size=17, 
                                              margin = margin(t=10, r=0, b=0, l=0)),
                  axis.title.y = element_text(face = "bold", size=17,
                                              margin = margin(t=0, r=10, b=0, l=0)),
                  legend.position="bottom",
                  legend.key.size = unit(1, "line"),
                  legend.text=element_text(size=15)
            )

x_limits = c(-100, 6000)
y_limits = c(0, 0.0025)

colorss = c("black", "dodgerblue", "olivedrab4", "red", "black", "dodgerblue", "olivedrab4", "red")
labelss = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4", "Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")

larva_density = ggplot(data.frame(x=x_limits), aes(x=x)) + the_theme + 
                      geom_path(stat="function", fun=dw_larva_Gen1, aes(colour="black"), linetype=1)+
                      geom_path(stat="function", fun=dw_larva_Gen2, aes(colour="dodgerblue"), linetype=1)+
                      geom_path(stat="function", fun=dw_larva_Gen3, aes(colour="olivedrab4"), linetype=1)+
                      geom_path(stat="function", fun=dw_larva_Gen4, aes(colour="red"), linetype=1)+
                      geom_path(stat="function", fun=dw_larva_Gen1_shift_10_percent, aes(colour="black"), linetype=2)+
                      geom_path(stat="function", fun=dw_larva_Gen2_shift_10_percent, aes(colour="dodgerblue"), linetype=2)+
                      geom_path(stat="function", fun=dw_larva_Gen3_shift_10_percent, aes(colour="olivedrab4"), linetype=2)+
                      geom_path(stat="function", fun=dw_larva_Gen4_shift_10_percent, aes(colour="red"), linetype=2)+
                      scale_x_continuous(name="degree day (in F)", limits=x_limits) + 
                      scale_y_continuous(name="density", limits=y_limits, labels = function(x) format(x*1000, scientific=F)) +
                      scale_colour_identity("", guide="legend", 
                                                labels = labelss, 
                                                breaks = colorss) +
                      # ggtitle(label = "Weibull density corresponding to larva parameters (solid lines) \n and 10% increase of scale (dashed lines)") +
                      labs(subtitle = expression(10^-3), parse=T, face="bold")

A <- annotate_figure(larva_density,
                     bottom = text_grob('Weibull density corresponding to larva parameters (solid lines) \n and 10% increase of scale (dashed lines)', color = "black",
                                         hjust = 1.45, x=1, face="plain", size=12))

# A <- larva_density + annotate("text", x = 0, y =0, 
#                               label = 'atop(bold("This should be bold"),"this should not")',
#                               colour = "red", parse = TRUE)


master_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/laptop_codes/sensitivity/weibull_dist_plot/"
plot_path = master_path
ggsave(filename=paste0("dweibull_larva_0.1", ".png"), 
       plot=larva_density,  path=plot_path, 
       width=7 , height=5 , dpi=500, device="png")

