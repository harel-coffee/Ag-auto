
rm(list=ls())
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

library(Hmisc) # for negation of %in%

data_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
file_names = c("overlap_stat.rds", 
               "non_overlap_stat.rds", 
               "observed_stat.rds"
               )

modeled_overlap_stat = data.table(readRDS(paste0(data_dir, file_names[1])))
modeled_non_overlap_stat = data.table(readRDS(paste0(data_dir, file_names[2])))
observed_stat = data.table(readRDS(paste0(data_dir, file_names[3])))

rm(data_dir, file_names)

################################################

modeled_overlap_stat$CountyGroup = as.character(modeled_overlap_stat$CountyGroup)
modeled_overlap_stat[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
modeled_overlap_stat[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

modeled_non_overlap_stat$CountyGroup = as.character(modeled_non_overlap_stat$CountyGroup)
modeled_non_overlap_stat[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
modeled_non_overlap_stat[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

observed_stat$CountyGroup = as.character(observed_stat$CountyGroup)
observed_stat[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
observed_stat[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

################################################

modeled_historical = filter(modeled_non_overlap_stat, scenario=="historical") 

# The reason for the following two lines is that
# I forgot to inclide the 1979 in the overlap case!!!
modeled_overlap_stat = filter(modeled_overlap_stat, scenario!="historical")
modeled_overlap_stat = rbind(modeled_overlap_stat, modeled_historical)

# combone the two historical types, observed and modeled.
observed_modeled = rbind(observed_stat, modeled_historical)
rm(observed_stat, modeled_historical)

# drop the years that do not belong to any climate time perios
# just a sanity check! they are OK! when I checked last. no need for 
# the following line.
observed_modeled <- observed_modeled[!is.na(observed_modeled$ClimateGroup),]

# fix the order of stuff
time_periods = c("Historical", "2040's", "2060's", "2080's")

modeled_overlap_stat$ClimateGroup <- factor(modeled_overlap_stat$ClimateGroup)
modeled_overlap_stat$ClimateGroup <- ordered(modeled_overlap_stat$ClimateGroup, 
                                     levels = time_periods)
modeled_overlap_stat$ClimateGroup[1]

modeled_non_overlap_stat$ClimateGroup <- factor(modeled_non_overlap_stat$ClimateGroup)
modeled_non_overlap_stat$ClimateGroup <- ordered(modeled_non_overlap_stat$ClimateGroup, 
                                                 levels = time_periods)
modeled_non_overlap_stat$ClimateGroup[1]

observed_modeled$ClimateGroup = factor(observed_modeled$ClimateGroup)
observed_modeled$ClimateGroup <- ordered(observed_modeled$ClimateGroup, 
                                         levels = time_periods)
observed_modeled$ClimateGroup[1]
################################################
#######
#######                Plot Area
#######
################################################
the_theme <- theme_bw() + 
             theme(plot.margin = unit(c(t=0.1, r=0.5, b=.1, l=0.1), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(hjust = 0.5),
                   plot.subtitle = element_text(hjust = 0.5),
                   panel.grid.major = element_line(size = 0.2),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   # legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
                   legend.text=element_text(size=7),
                   legend.margin=margin(t= -0.1, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 10),
                   axis.ticks = element_line(color = "black", size = .2),
                   #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                   axis.title.x = element_text(face = "plain", size=12, 
                   	                            margin = margin(t=6, r=0, b=0, l=0)),
                   # axis.title.x=element_blank(),
                   axis.text.x = element_text(size = 10, face = "plain", 
                   	                          color="black"),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_text(face = "plain", size = 12, 
                                               margin = margin(t=0, r=6, b=0, l=0)),
                   axis.text.y = element_text(size = 10, face="plain", color="black")
                   # axis.title.y = element_blank()
                   )

####################################################
########
######## Over lapped time periods
########
####################################################

warm_cool_overlap <- ggplot(data = modeled_overlap_stat) +
                     geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                                    fill = factor(model),
                                     colour = factor(model),
                                     shape = factor(model)),
                                     alpha = 0.6, 
                                     size = 1) + 
                     scale_shape_manual(values = seq(1:19)) +
                     facet_grid( ~ ClimateGroup ~ CountyGroup) + 
                     ggtitle(label = "Time periods overlap in this plot") + 
                     xlab("average of GDD on 31st of Dec.") +
                     ylab("average of daily tmean") +
                          the_theme
warm_cool_overlap

ggsave(filename = "warm_cool_overlap.png", 
	     path = "/Users/hn/Desktop/overlap_periods/", 
	     plot = warm_cool_overlap,
       width = 10, height = 6, units = "in",
       dpi=400, 
       device = "png")


warm_cool_overlap_4585 <- ggplot(data = modeled_overlap_stat) +
                         geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                                        fill = factor(model),
                                         colour = factor(model),
                                         shape = factor(model)),
                                         alpha = 0.6, 
                                         size = 1) + 
                         scale_shape_manual(values = seq(1:19)) +
                         facet_grid( ~ CountyGroup  ~  ClimateGroup ~ scenario) + 
                         ggtitle(label = "Time periods overlap in this plot") + 
                         xlab("average of GDD on 31st of Dec.") +
                         ylab("average of daily tmean") +
                              the_theme
# warm_cool_overlap_4585
ggsave(filename = "warm_cool_overlap_4585.png", 
       path = "/Users/hn/Desktop/overlap_periods/", 
       plot = warm_cool_overlap_4585,
       width = 10, height = 10, units = "in",
       dpi=400, 
       device = "png")

####################################################
########
######## NON Over lapped time periods
########
####################################################

warm_cool_non_overlap <- ggplot(data = modeled_non_overlap_stat) +
                         geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                                        fill = factor(model),
                                         colour = factor(model),
                                         shape = factor(model)),
                                         alpha = 0.6, 
                                         size = 1) + 
                         scale_shape_manual(values = seq(1:19)) +
                         facet_grid( ~ ClimateGroup ~ CountyGroup) + 
                         ggtitle(label = "Time periods do not overlap in this plot") + 
                         xlab("average of GDD on 31st of Dec.") +
                         ylab("average of daily tmean") +
                         the_theme
# warm_cool_non_overlap

ggsave(filename = "warm_cool_non_overlap.png", 
       path = "/Users/hn/Desktop/non_overlap_periods/", 
       plot = warm_cool_non_overlap,
       width = 10, height = 6, units = "in",
       dpi=400, 
       device = "png")


warm_cool_non_overlap_4585 <- ggplot(data = modeled_non_overlap_stat) +
                             geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                                            fill = factor(model),
                                             colour = factor(model),
                                             shape = factor(model)),
                                             alpha = 0.6, 
                                             size = 1) + 
                             scale_shape_manual(values = seq(1:19)) +
                             facet_grid( ~ CountyGroup  ~  ClimateGroup ~ scenario) + 
                             ggtitle(label = "Time periods do not overlap in this plot") + 
                             xlab("average of GDD on 31st of Dec.") +
                             ylab("average of daily tmean") +
                             the_theme
# warm_cool_non_overlap_4585
ggsave(filename = "warm_cool_non_overlap_4585.png", 
       path = "/Users/hn/Desktop/non_overlap_periods/", 
       plot = warm_cool_non_overlap_4585,
       width = 10, height = 10, units = "in",
       dpi=400, 
       device = "png")
####################################################
#################################################### 
#######
#######
#######     Separate the chosen models of Cod. Moth.
#######
#######
#################################################### 
####################################################
the_six_models = c("bcc-csm1-1-m", "BNU-ESM", 
  	               "CanESM2", "CNRM-CM5", 
  	               "GFDL-ESM2G", "GFDL-ESM2M")
# modeled_stat$chosen = 0L
# modeled_stat$chosen[modeled_stat$model %in% the_six_models] = "chosen"
# modeled_stat$chosen[modeled_stat$chosen == "0" ] = "not_chosen"

modeled_stat_1_overlap = modeled_overlap_stat
modeled_stat_1_overlap[(modeled_stat_1_overlap$model %nin% the_six_models), ]$model = "others"
modeled_stat_1_overlap$model = factor(modeled_stat_1_overlap$model)

time_periods = c("Historical", "2040's",  "2060's", "2080's")
modeled_stat_1_overlap$ClimateGroup = factor(modeled_stat_1_overlap$ClimateGroup)
modeled_stat_1_overlap$ClimateGroup <- ordered(modeled_stat_1_overlap$ClimateGroup, 
                                         levels = time_periods)

modeled_chosen_overlap_plot <- ggplot(data = modeled_stat_1_overlap) +
                               geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                  	           #fill = factor(model),
                                	           colour = factor(model),
                                	           shape = factor(model)
                                	           ),
                                         alpha = 1, size = 2) +
                                scale_shape_manual(values = c(4, 8, 15, 16, 17, 18, 21, 22, 3, 42)) +
                                #scale_colour_manual(values=seq(0, 7)) +
                                facet_grid( ~ CountyGroup  ~  ClimateGroup ~ scenario) +
                                ggtitle(label = "Time periods overlap in this plot") + 
                                xlab("average of GDD on 31st of Dec.") +
                                ylab("average of daily tmean") +
                                the_theme
modeled_chosen_overlap_plot

ggsave(filename = "modeled_chosen_overlap_plot.png", 
	     path = "/Users/hn/Desktop/overlap_periods/", 
	     plot = modeled_chosen_overlap_plot,
       width = 10, height = 10, units = "in",
       dpi=400, 
       device = "png")


######## Non Overlap chosen models
###########################################################
modeled_stat_1_non_overlap = modeled_non_overlap_stat
modeled_stat_1_non_overlap[(modeled_stat_1_non_overlap$model %nin% the_six_models), ]$model = "others"
modeled_stat_1_non_overlap$model = factor(modeled_stat_1_non_overlap$model)

time_periods = c("Historical", "2040's",  "2060's", "2080's")
modeled_stat_1_non_overlap$ClimateGroup = factor(modeled_stat_1_non_overlap$ClimateGroup)
modeled_stat_1_non_overlap$ClimateGroup <- ordered(modeled_stat_1_non_overlap$ClimateGroup, 
                                         levels = time_periods)

modeled_chosen_non_overlap_plot <- ggplot(data = modeled_stat_1_non_overlap) +
                               geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                               #fill = factor(model),
                                             colour = factor(model),
                                             shape = factor(model)
                                             ),
                                         alpha = 1, size = 2) +
                                scale_shape_manual(values = c(4, 8, 15, 16, 17, 18, 21, 22, 3, 42)) +
                                #scale_colour_manual(values=seq(0, 7)) +
                                facet_grid( ~ CountyGroup  ~  ClimateGroup ~ scenario) +
                                ggtitle(label = "Time periods do not overlap in this plot") + 
                                xlab("average of GDD on 31st of Dec.") +
                                ylab("average of daily tmean") +
                                the_theme
modeled_chosen_non_overlap_plot

ggsave(filename = "modeled_chosen_non_overlap_plot.png", 
       path = "/Users/hn/Desktop/non_overlap_periods/", 
       plot = modeled_chosen_non_overlap_plot,
       width = 10, height = 10, units = "in",
       dpi=400, 
       device = "png")

###########################################################
########################################################### 
#######
#######
#######     observed comparison with modeled historical
#######
#######
########################################################### 
###########################################################
observed_modeled$scenario[observed_modeled$scenario=="historical"] = "modeled"
observed_modeled_1 = observed_modeled


# drop some columns
# observed_modeled_1[, ClimateGroup := NULL]
# observed_modeled_1[, scenario := NULL]

the_theme <- theme_bw() + 
             theme(plot.margin = unit(c(t=0.1, r=0.5, b=.1, l=0.1), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(hjust = 0.5),
                   plot.subtitle = element_text(hjust = 0.5),
                   panel.grid.major = element_line(size = 0.2),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   # legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
                   legend.text=element_text(size=7),
                   legend.margin=margin(t= -0.1, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 10),
                   axis.ticks = element_line(color = "black", size = .2),
                   #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                   axis.title.x = element_text(face = "plain", size=12, 
                                                margin = margin(t=4, r=0, b=0, l=0)),
                   # axis.title.x=element_blank(),
                   axis.text.x = element_text(size = 8, face = "plain", 
                                              color="black"),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_text(face = "plain", size = 12, 
                                               margin = margin(t=0, r=2, b=0, l=0)),
                   axis.text.y = element_text(size = 8, face="plain", color="black")
                   # axis.title.y = element_blank()
                   )

plot <- ggplot(data = observed_modeled_1) +
        geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
        	           #fill = factor(model),
        	           colour = factor(model),
        	           shape = factor(model)
        	           ),
                   alpha = 1, size = 2) +
        scale_shape_manual(values = c(seq(1, 7), seq(9:20), 8)) +
        xlab("average of GDD on 31st of Dec.") +
        ylab("average of daily tmean") + 
        facet_wrap(~ CountyGroup, scales= "free") +
        the_theme
plot

ggsave(filename = "observed_vs_modeled_hist.png", 
  	   path = "/Users/hn/Desktop/observed_vs_historical_modeled/", 
  	   plot = plot,
       width = 6, height = 5, units = "in",
       dpi=400, 
       device = "png")

####### 
####### 
####### 
the_theme <- theme_bw() + 
             theme(plot.margin = unit(c(t=0.1, r=0.5, b=.1, l=0.1), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(hjust = 0.5),
                   plot.subtitle = element_text(hjust = 0.5),
                   panel.grid.major = element_line(size = 0.2),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
                   legend.text=element_text(size=7),
                   legend.margin=margin(t= -0.1, r = 0, b = 0, l = 0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = 10),
                   axis.ticks = element_line(color = "black", size = .2),
                   #axis.text = element_text(face = "plain", size = 2.5, color="black"),
                   axis.title.x = element_text(face = "plain", size=12, 
                                                margin = margin(t=4, r=0, b=0, l=0)),
                   # axis.title.x=element_blank(),
                   axis.text.x = element_text(size = 8, face = "plain", 
                                              color="black"),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_text(face = "plain", size = 12, 
                                               margin = margin(t=0, r=2, b=0, l=0)),
                   axis.text.y = element_text(size = 8, face="plain", color="black")
                   # axis.title.y = element_blank()
                   )
observed_modeled_2 = observed_modeled_1
observed_modeled_2[(observed_modeled_2$model %nin% c("observed"))]$model = "modeled"

plot <- ggplot(data = observed_modeled_2) +
        geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                     #fill = factor(model),
                     colour = factor(model),
                     shape = factor(model)
                     ),
                   alpha = 1, size = 2) +
        scale_shape_manual(values = seq(20, 1)) +
        xlab("average of GDD on 31st of Dec.") +
        ylab("average of daily tmean") + 
        #scale_colour_manual(values=seq(0, 7)) +
        facet_grid(~ CountyGroup) +
        the_theme
# plot
ggsave(filename = "observed_modeled_2.png", 
       path = "/Users/hn/Desktop/observed_vs_historical_modeled/", 
       plot = plot,
       width = 6, height = 4, units = "in",
       dpi=400, 
       device = "png")


####### 
####### chosen
####### 
the_six_models = c("bcc-csm1-1-m", "BNU-ESM", 
                   "CanESM2", "CNRM-CM5", 
                   "GFDL-ESM2G", "GFDL-ESM2M")
observed_modeled_3 = observed_modeled_1

observed_modeled_3[(observed_modeled_3$model %nin% c("observed", the_six_models))]$model = "others"
observed_modeled_3[(observed_modeled_3$model %nin% c("observed"))]$model = "modeled"

plot <- ggplot(data = observed_modeled_3) +
        geom_point(aes(x = mean_cumm_dd, y = mean_tmean, 
                     #fill = factor(model),
                     colour = factor(model),
                     shape = factor(model)
                     ),
                   alpha = 1, size = 2) +
        scale_shape_manual(values = seq(20, 1)) +
        xlab("average of GDD on 31st of Dec.") +
        ylab("average of daily tmean") + 
        #scale_colour_manual(values=seq(0, 7)) +
        facet_grid(~ CountyGroup) +
        the_theme
# plot
ggsave(filename = "observed_chosen_others.png", 
     path = "/Users/hn/Desktop/observed_vs_historical_modeled/", 
     plot = plot,
       width = 6, height = 4, units = "in",
       dpi=400, 
       device = "png")


