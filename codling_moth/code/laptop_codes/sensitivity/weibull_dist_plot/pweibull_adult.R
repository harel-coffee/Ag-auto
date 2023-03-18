################## adult pweibull
library(chron)
library(EnvStats)
library(grid)
# library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(ggplot2)
par(mar=c(1, 1, 1, 1))
dev.off() 

param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
param_name = "CodlingMothparameters.txt"
params = read.table(paste0(param_dir, param_name), header=TRUE, sep=",")

param_name_shift = "CodlingMothparameters_0.1.txt"
params_shift = read.table(paste0(param_dir, param_name_shift), header=TRUE, sep=",")

########################################################################
########################################################################
########################## Functions
##########################
pw_adult_1 <- function(x, shape=params[13, "shape"], scale=params[13, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_2 <- function(x, shape=params[14, "shape"], scale=params[14, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_3 <- function(x, shape=params[15, "shape"], scale=params[15, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_4 <- function(x, shape=params[16, "shape"], scale=params[16, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
######################### S H I F T 10 %
pw_adult_1_shift <- function(x, shape=params_shift[13, "shape"], scale=params_shift[13, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_2_shift <- function(x, shape=params_shift[14, "shape"], scale=params_shift[14, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_3_shift <- function(x, shape=params_shift[15, "shape"], scale=params_shift[15, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}

pw_adult_4_shift <- function(x, shape=params_shift[16, "shape"], scale=params_shift[16, "scale"]){
  pweibull(x, shape=shape, scale=scale)
}
########################## 
########################## Functions
########################################################################
########################################################################
########################## The Theme
##########################
the_theme = theme(panel.grid.major = element_blank(), 
                  # panel.grid.minor = element_blank(),
                  axis.text.x = element_text(size = 9, color="black"),
                  axis.title.x = element_text(face = "plain", 
                                              size=12, 
                                              margin = margin(t=10, r=0, b=0, l=0)),
                  axis.text.y = element_text(size = 9, angle=90, color="black"),
                  axis.title.y = element_text(face = "plain", 
                                              size=12, 
                                              margin = margin(t=0, r=10, b=0, l=0)),
                  legend.position="bottom"
            ) + theme_bw()

x_limits = c(-100, 6000)
y_limits = c(0, 1)

colorss = c("grey70", "dodgerblue", "olivedrab4", "red", "grey70", "dodgerblue", "olivedrab4", "red")
labelss = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4", "Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")

adult_density = ggplot(data.frame(x=x_limits), aes(x=x)) + the_theme + 
			  geom_path(stat="function", fun=pw_adult_1, aes(colour="grey70"), linetype=1)+
			  geom_path(stat="function", fun=pw_adult_2, aes(colour="dodgerblue"), linetype=1)+
			  geom_path(stat="function", fun=pw_adult_3, aes(colour="olivedrab4"), linetype=1)+
			  geom_path(stat="function", fun=pw_adult_4, aes(colour="red"), linetype=1)+
			  geom_path(stat="function", fun=pw_adult_1_shift, aes(colour="grey70"), linetype=2)+
			  geom_path(stat="function", fun=pw_adult_2_shift, aes(colour="dodgerblue"), linetype=2)+
			  geom_path(stat="function", fun=pw_adult_3_shift, aes(colour="olivedrab4"), linetype=2)+
			  geom_path(stat="function", fun=pw_adult_4_shift, aes(colour="red"), linetype=2)+
			  scale_x_continuous(name="Degree days", limits=x_limits) + 
			  scale_y_continuous(name="Weibull cumulative distribution", limits=y_limits, labels = function(x) format(x, scientific=F)) +
			  scale_colour_identity("", guide="legend", 
			                        labels = labelss, 
			                        breaks = colorss)

master_path = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/weibull_dist_plot/"
plot_path = master_path
ggsave(filename=paste0("pweibull_adult", ".png"), 
	   plot=adult_density, 
	   path=plot_path, 
	   width=7 ,
	   height=5 , 
	   dpi=1000, 
	   device="png")

