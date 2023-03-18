################## Larva dweibull
rm(list=ls())
library(chron)
library(EnvStats)
library(grid)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(ggplot2)
library(gridExtra)
library(gtable)
library(egg)
par(mar=c(1, 1, 1, 1))
dev.off() 

param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
params = "CodlingMothparameters.txt"
params = read.table(paste0(param_dir, params), header=TRUE, sep=",")

params_shift_10_percent = "CodlingMothparameters_0.1.txt"
params_shift_10_percent = read.table(paste0(param_dir, params_shift_10_percent), 
                                            header=TRUE, sep=",")

params_shift_20_percent <- "CodlingMothparameters_0.2.txt"
params_shift_20_percent = read.table(paste0(param_dir, params_shift_20_percent), 
                                            header=TRUE, sep=",")

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
pw_larva_Gen1 <- function(x, shape=params[5, "shape"], scale=params[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen2 <- function(x, shape=params[6, "shape"], scale=params[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen3 <- function(x, shape=params[7, "shape"], scale=params[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen4 <- function(x, shape=params[8, "shape"], scale=params[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
##################################################
#########################
######################### S H I F T 10 %
#########################
##################################################
pw_larva_Gen1_shift_10_percent <- function(x, shape=params_shift_10_percent[5, "shape"], 
                                              scale=params_shift_10_percent[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen2_shift_10_percent <- function(x, shape=params_shift_10_percent[6, "shape"], 
                                              scale=params_shift_10_percent[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen3_shift_10_percent <- function(x, shape=params_shift_10_percent[7, "shape"], 
                                              scale=params_shift_10_percent[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen4_shift_10_percent <- function(x, shape=params_shift_10_percent[8, "shape"], 
                                              scale=params_shift_10_percent[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

##################################################
#########################
######################### S H I F T 20 %
#########################
##################################################
pw_larva_Gen1_shift_20_percent <- function(x, shape=params_shift_20_percent[5, "shape"], 
                                              scale=params_shift_20_percent[5, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen2_shift_20_percent <- function(x, shape=params_shift_20_percent[6, "shape"], 
                                              scale=params_shift_20_percent[6, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen3_shift_20_percent <- function(x, shape=params_shift_20_percent[7, "shape"], 
                                              scale=params_shift_20_percent[7, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_larva_Gen4_shift_20_percent <- function(x, shape=params_shift_20_percent[8, "shape"], 
                                              scale=params_shift_20_percent[8, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
########################## 
########################## Functions
########################################################################
########################################################################
dw_x_limits = c(-100, 6000)
dw_y_limits = c(0, 0.0025)

colorss = c("dodgerblue", "olivedrab4", "red")
labelss = c("original", 
            "10% change of scale param.", 
            "20% change of scale param.")

dw_theme = theme(panel.grid.major = element_blank(), 
                  # panel.grid.minor = element_blank(),
                  axis.text.x = element_text(size = 9, color="black"),
                  axis.title.x = element_text(face = "plain", 
                                              size=12, 
                                              margin = margin(t=10, r=0, b=0, l=0)),
                  axis.text.y = element_text(size = 9, angle=0, color="black"),
                  axis.title.y = element_text(face = "plain", 
                                              size=12, 
                                              margin = margin(t=0, r=10, b=0, l=0)),
                  legend.position=c(.8,.85),
                  legend.title=element_blank()
                  )

dw_gen_1 <- ggplot(data.frame(x=c(150, 1400)), aes(x=x)) + 
            geom_path(stat="function", fun=dw_larva_Gen1, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen1_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen1_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(150, 1400)) + 
            scale_y_continuous(#name="Weibull density", 
                               name = element_blank(),
                               limits=dw_y_limits, 
                               labels = function(x) format(x*1000, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                      labels = labelss, 
                                      breaks = colorss) +
            labs(subtitle = expression(10^-3), parse=T) + 
            theme_bw() + 
            dw_theme

dw_gen_2 <- ggplot(data.frame(x=c(700, 2900)), aes(x=x)) + 
            geom_path(stat="function", fun=dw_larva_Gen2, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen2_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen2_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(700, 2900)) + 
            scale_y_continuous(#name="Weibull density", 
                name = element_blank(),
                limits=dw_y_limits, labels = function(x) format(x*1000, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                  labels = labelss, 
                                  breaks = colorss) +
            labs(subtitle = expression(10^-3), parse=T) + 
            theme_bw() + 
            dw_theme

dw_gen_3 <- ggplot(data.frame(x=c(1700, 4300)), aes(x=x)) + 
            geom_path(stat="function", fun=dw_larva_Gen3, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen3_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen3_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(1700, 4300)) + 
            scale_y_continuous(#name="Weibull density", 
                name = element_blank(),
                limits=dw_y_limits, labels = function(x) format(x*1000, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                  labels = labelss, 
                                  breaks = colorss) +
            labs(subtitle = expression(10^-3), parse=T) + 
            theme_bw() + 
            dw_theme

dw_gen_4 <- ggplot(data.frame(x=c(2700, 5600)), aes(x=x)) + 
            geom_path(stat="function", fun=dw_larva_Gen4, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen4_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=dw_larva_Gen4_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(2700, 5600)) + 
            scale_y_continuous(#name="Weibull density", 
                name = element_blank(),
                limits=dw_y_limits, labels = function(x) format(x*1000, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                  labels = labelss, 
                                  breaks = colorss) +
            labs(subtitle = expression(10^-3), parse=T) + 
            theme_bw() + 
            dw_theme
############################################################
####################
#################### pw plots
####################
############################################################
pw_theme = theme(panel.grid.major = element_blank(), 
                 # panel.grid.minor = element_blank(),
                 axis.text.x = element_text(size = 9, color="black"),
                 axis.title.x = element_text(face = "plain", 
                                             size=12, 
                                             margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.y = element_text(size = 9, angle=0, color="black"),
                 axis.title.y = element_text(face = "plain", 
                                             size=12, 
                                             margin = margin(t=0, r=10, b=0, l=0)),
                 # legend.position="bottom",
                 legend.position=c(.86,.15),
                 legend.title=element_blank(),
                 legend.text=element_text(size=5)
                 )

pw_x_limits = c(-100, 6000)
pw_y_limits = c(0, 1)

pw_gen_1 <- ggplot(data.frame(x=c(150, 1400)), aes(x=x)) + 
            geom_path(stat="function", fun=pw_larva_Gen1, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen1_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen1_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(150, 1400)) + 
            scale_y_continuous(# name="", 
                               name = element_blank(),
                               limits=pw_y_limits, 
                               labels = function(x) format(x*100, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                      labels = labelss, 
                                      breaks = colorss) +
            labs(subtitle = expression(10^-2), parse=T) + 
            theme_bw() + 
            pw_theme

pw_gen_2 <- ggplot(data.frame(x=c(700, 2900)), aes(x=x)) + 
            geom_path(stat="function", fun=pw_larva_Gen2, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen2_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen2_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(700, 2900)) + 
            scale_y_continuous(#name="Weibull cumulative distribution", 
                               name = element_blank(),
                               limits=pw_y_limits, 
                               labels = function(x) format(x*100, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                      labels = labelss, 
                                      breaks = colorss) +
            labs(subtitle = expression(10^-2), parse=T) + 
            theme_bw() + 
            pw_theme

pw_gen_3 <- ggplot(data.frame(x=c(1700, 4300)), aes(x=x)) + 
            geom_path(stat="function", fun=pw_larva_Gen3, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen3_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen3_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(1700, 4300)) + 
            scale_y_continuous(#name="Weibull cumulative distribution", 
                               name = element_blank(),
                               limits=pw_y_limits, 
                               labels = function(x) format(x*100, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                      labels = labelss, 
                                      breaks = colorss) +
            labs(subtitle = expression(10^-2), parse=T) + 
            theme_bw() + 
            pw_theme

pw_gen_4 <- ggplot(data.frame(x=c(2700, 5500)), aes(x=x)) + 
            geom_path(stat="function", fun=pw_larva_Gen4, aes(colour="dodgerblue"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen4_shift_10_percent, aes(colour="olivedrab4"), linetype=1)+
            geom_path(stat="function", fun=pw_larva_Gen4_shift_20_percent, aes(colour="red"), linetype=1)+
            scale_x_continuous(name="Degree days", limits=c(2700, 5500)) + 
            scale_y_continuous(#name="Weibull cumulative distribution", 
                               name = element_blank(),
                               limits=pw_y_limits, 
                               labels = function(x) format(x*100, scientific=F)) +
            scale_colour_identity("", guide="legend", 
                                      labels = labelss, 
                                      breaks = colorss) +
            labs(subtitle = expression(10^-2), parse=T) + 
            theme_bw() + 
            pw_theme

########################################################
##################
################## Put the plots together in a panel
##################
########################################################
# all_plots <- grid.arrange(dw_gen_1, pw_gen_1,
#                           dw_gen_2, pw_gen_2, 
#                           dw_gen_3, pw_gen_3,
#                           dw_gen_4, pw_gen_4,
#                           nrow = 4)

# all_plots <- ggarrange(dw_gen_1, pw_gen_1, 
#                        dw_gen_2, pw_gen_2, 
#                        dw_gen_2, pw_gen_2, 
#                        dw_gen_3, pw_gen_3,
#                        dw_gen_4, pw_gen_4)
########################################################
grid_arrange_shared_legend <- function(..., ncol = length(list(...)),
                                            nrow = 1,
                                            position = c("bottom", "right")) {
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
       legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
       lheight <- sum(legend$height)
       lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x)
        x + theme(legend.position = "none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
    
  combined <- switch( position, "bottom" = arrangeGrob( do.call(arrangeGrob, gl),
        legend,
        ncol = 1,
        heights = unit.c(unit(1, "npc") - lheight, lheight)
      ),
      "right" = arrangeGrob(
        do.call(arrangeGrob, gl),
        legend,
        ncol = 2,
        widths = unit.c(unit(1, "npc") - lwidth, lwidth)
      )
    )
    
    grid.newpage()
    grid.draw(combined)
    
    # return gtable invisibly
    invisible(combined)
    
  }

all_plots <- grid_arrange_shared_legend(dw_gen_1, pw_gen_1, 
                           dw_gen_2, pw_gen_2, 
                           dw_gen_3, pw_gen_3,
                           dw_gen_4, pw_gen_4, 
                           nrow = 4, 
                           ncol=2)


master_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/laptop_codes/weibull_dist_plot/"

plot_path = "/Users/hn/Desktop/"
ggsave(filename=paste0("larva_weibull", ".png"), 
	     plot=all_plots, 
	     path=plot_path, 
  	   width=15,
	     height=10, 
	     dpi=500, 
	     device="png")

