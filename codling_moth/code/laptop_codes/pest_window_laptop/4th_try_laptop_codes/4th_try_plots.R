######## 4th Try
## The plot of 1st day of August cumDD
## and 
## The plot of pop_diff in the 14 days window
##
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/"
part_2 = "my_aeolus_2015/all_local/pest_control/3rd_try_data/"
data_dir = paste0(part_1, part_2)

name_prefix = "three_days_"

climate_scenarios = c("historical", "bcc-csm1-1-m", 
	                  "BNU-ESM", "CanESM2", "CNRM-CM5",
	                  "GFDL-ESM2G", "GFDL-ESM2M")
models = c("45", "85")

for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_prefix, model, ".rds")))
	months = c(4, 6, 8)

	# april_data  = curr_data[curr_data$month == 4]
	# june_data   = curr_data[curr_data$month == 6]
	august = curr_data[curr_data$month == 8]
	august = subset(august, select = c(ClimateScenario, 
		                               ClimateGroup, CountyGroup,
		                               CumDDinF, year))
	august$CountyGroup = as.character(august$CountyGroup)
	august[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
	august[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
	august = melt(august, id = c("ClimateGroup", "CountyGroup",
	                             "ClimateScenario", "year"))

	legend_scenarios = c("historical", "bcc", "BNU", "Can", "CNRM",
		                 "GFDL-G", "GFDL-M")
	fill_colors = c("red", "blue", "deepskyblue2", 
	                "green", "grey70", "dodgerblue", 
	                "olivedrab4")

	the_theme <- theme(# panel.grid.major = element_line(size = 0.2),
			           # panel.spacing=unit(1, "cm"),
			           plot.margin = unit(c(t = 0, r = 1, b = 0, l = 1), "cm"),
			           panel.spacing.x = unit(1, "cm"),
			           # panel.spacing.y = unit(1, "lines"),
			           panel.grid.major = element_blank(),
	                   panel.grid.minor = element_blank(),
			           legend.title = element_blank(),
	  	  	           legend.text = element_text(size=10),
			           legend.position = "bottom",
			           strip.text = element_text(size=12, face="plain"),
			           axis.text = element_text(face="plain", size=10, color="black"),
			           axis.ticks = element_line(color = "black", size = .2),
			           axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
			           axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
			           )
	plot <- ggplot(august, 
		           aes(x=year, y=value, fill=factor(ClimateScenario))) +
			       labs(x = "Year", y = "Cumulative DD (in F)") +
			       facet_grid(. ~ CountyGroup, scales="free") +
				   # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.1) }, 
				                fun.ymax=function(z) { quantile(z,0.9) }, 
				                alpha=0.3) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.25) }, 
				                fun.ymax=function(z) { quantile(z,0.75) }, 
				                alpha=0.8) + 
				   stat_summary(geom="line", 
				                fun.y=function(z) { quantile(z,0.5) }) +
				   scale_color_manual(values=fill_colors, 
	                                  labels = legend_scenarios) +
	               scale_fill_manual(values=fill_colors,  
	                                 labels = legend_scenarios) +
				   # scale_x_continuous(breaks=seq(0, 300, 50)) +
				   scale_y_continuous(limits = c(500, 4000), breaks=seq(500, 4000, by=1000)) + 
				   theme_bw() + 
				   the_theme

	output_dir = "/Users/hn/Desktop/"
	out_name = paste0("august_1st_dd_", model, ".png")
	ggsave(out_name, plot, path=output_dir, width=14, height=7, unit="in", dpi=400)
	############
	############ separate the climate scenarios
	############
	plot_sep <- ggplot(august, 
		           aes(x=year, y=value, fill=factor(ClimateScenario))) +
			       labs(x = "Year", y = "Cumulative DD (in F)") +
			       facet_grid(. ~ ClimateScenario ~ CountyGroup, scales="free") +
				   # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.1) }, 
				                fun.ymax=function(z) { quantile(z,0.9) }, 
				                alpha=0.3) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.25) }, 
				                fun.ymax=function(z) { quantile(z,0.75) }, 
				                alpha=0.8) + 
				   stat_summary(geom="line", 
				                fun.y=function(z) { quantile(z,0.5) }) +
				   scale_color_manual(values = fill_colors, 
	                                  labels = legend_scenarios) +
	               scale_fill_manual(values = fill_colors,  
	                                 labels = legend_scenarios) +
				   # scale_x_continuous(breaks=seq(0, 300, 50)) +
				   scale_y_continuous(limits = c(500, 4000), breaks=seq(500, 4000, by=1000)) + 
				   theme_bw() + 
				   the_theme

	output_dir = "/Users/hn/Desktop/"
	out_name = paste0("august_1st_dd_separate_", model, ".png")
	ggsave(out_name, plot_sep, path=output_dir, width=14, height=7, unit="in", dpi=400)
	#################################
	###########
	########### line plots
	###########
	#################################
	the_theme <- theme(# panel.grid.major = element_line(size = 0.2),
			           # panel.spacing=unit(1, "cm"),
			           plot.margin = unit(c(t = 0, r = 1, b = 0, l = 1), "cm"),
			           panel.spacing.x = unit(1, "cm"),
			           # panel.spacing.y = unit(1, "lines"),
			           panel.grid.major = element_blank(),
	                   panel.grid.minor = element_blank(),
			           legend.title = element_blank(),
	  	  	           legend.text = element_text(size=10),
			           legend.position = "bottom",
			           strip.text = element_text(size=12, face="plain"),
			           axis.text = element_text(face="plain", size=10, color="black"),
			           axis.ticks = element_line(color = "black", size = .2),
			           axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
			           axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
			           )
	line_plot <- ggplot(august, 
	                    aes(x=year, y=value, fill=factor(ClimateScenario), color=factor(ClimateScenario))) +
	                    labs(x = "Year", y = "Cumulative DD (in F)") +
	                    facet_grid(. ~ CountyGroup, scales="free") +
	                    # geom_line(aes(fill=factor(ClimateScenario), color=factor(ClimateScenario) )) +
	                    stat_summary(geom="line", 
	                                 fun.y=function(z) { quantile(z,0.5) }) +
	                    scale_y_continuous(limits=c(500, 4000), breaks=seq(500, 4000, by=1000)) + 
	                    theme_bw() + the_theme

	out_name = paste0("august_1st_dd_line_", model, ".png")
	ggsave(out_name, line_plot, path=output_dir, width=14, height=10, unit="in", dpi=400)

	line_plot_sep <- ggplot(august, 
				     aes(x=year, y=value, fill=factor(ClimateScenario), color=factor(ClimateScenario))) +
				     labs(x = "Year", y = "Cumulative DD (in F)") +
	 			     facet_grid(. ~ClimateScenario ~ CountyGroup, scales="free") +
				     # geom_line(aes(fill=factor(ClimateScenario), color=factor(ClimateScenario) )) +
				     stat_summary(geom="line", 
				                  fun.y=function(z) { quantile(z,0.5) }) +
				     # scale_x_continuous(breaks=seq(0, 300, 50)) +
				     scale_y_continuous(limits=c(500, 4000), breaks=seq(500, 4000, by=1000)) + 
				     theme_bw() + the_theme
	out_name = paste0("august_1st_dd_line_sep_", model, ".png")
	ggsave(out_name, line_plot_sep, path=output_dir, width=14, height=10, unit="in", dpi=400)
}
##############################################################################
#######################################
#######################################       pop diff.
#######################################
##############################################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/"
part_2 = "my_aeolus_2015/all_local/pest_control/400_2nd_try_data/"
data_dir = paste0(part_1, part_2)

start_end_name = "start_end_"

climate_scenarios = c("historical", "bcc-csm1-1-m", 
	                  "BNU-ESM", "CanESM2", "CNRM-CM5",
	                  "GFDL-ESM2G", "GFDL-ESM2M")
models = c("45", "85")
for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, start_end_name, model, ".rds")))
	curr_data = subset(curr_data, select=c(ClimateScenario, ClimateGroup, 
		                                   CountyGroup, pop_delta, year))

	curr_data = melt(curr_data, id = c("ClimateGroup", "CountyGroup",
		                             "ClimateScenario", "year"))

	legend_scenarios = c("historical", "bcc", "BNU", "Can", "CNRM",
			                 "GFDL-G", "GFDL-M")
	fill_colors = c("red", "blue", "deepskyblue2", 
	                "green", "grey70", "dodgerblue", 
	                "olivedrab4")

	the_theme <- theme(# panel.grid.major = element_line(size = 0.2),
			           # panel.spacing=unit(1, "cm"),
			           plot.margin = unit(c(t = 0, r = 1, b = 0, l = 1), "cm"),
			           panel.spacing.x = unit(1, "cm"),
			           # panel.spacing.y = unit(1, "lines"),
			           panel.grid.major = element_blank(),
	                   panel.grid.minor = element_blank(),
			           legend.title = element_blank(),
	  	  	           legend.text = element_text(size=10),
			           legend.position = "bottom",
			           strip.text = element_text(size=12, face="plain"),
			           axis.text = element_text(face="plain", size=10, color="black"),
			           axis.ticks = element_line(color = "black", size = .2),
			           axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
			           axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
			           )
	plot <- ggplot(curr_data, 
		           aes(x=year, y=value, fill=factor(ClimateScenario))) +
			       labs(x = "Year", y = "pop. diff.") +
			       facet_grid(. ~ CountyGroup, scales="free") +
				   # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.1) }, 
				                fun.ymax=function(z) { quantile(z,0.9) }, 
				                alpha=0.3) +
				   stat_summary(geom="ribbon", 
				                fun.y=function(z) { quantile(z,0.5) }, 
				                fun.ymin=function(z) { quantile(z,0.25) }, 
				                fun.ymax=function(z) { quantile(z,0.75) }, 
				                alpha=0.8) + 
				   stat_summary(geom="line", 
				                fun.y=function(z) { quantile(z,0.5) }) +
				   scale_color_manual(values=fill_colors, 
	                                  labels = legend_scenarios) +
	               scale_fill_manual(values=fill_colors,  
	                                 labels = legend_scenarios) +
				   # scale_x_continuous(breaks=seq(0, 300, 50)) +
				   scale_y_continuous(limits = c(0, 1), breaks=seq(0, 1, by=.2)) + 
				   theme_bw() + 
				   the_theme

	output_dir = "/Users/hn/Desktop/"
	out_name = paste0("pop_diff_", model, ".png")
	ggsave(out_name, plot, path=output_dir, width=14, height=7, unit="in", dpi=400)
	############
	############ separate the climate scenarios
	############
	plot_sep <- ggplot(curr_data, 
	           aes(x=year, y=value, fill=factor(ClimateScenario))) +
		       labs(x = "Year", y = "pop. diff.") +
		       facet_grid(. ~ ClimateScenario ~ CountyGroup, scales="free") +
			   # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
			   stat_summary(geom="ribbon", 
			                fun.y=function(z) { quantile(z,0.5) }, 
			                fun.ymin=function(z) { quantile(z,0.1) }, 
			                fun.ymax=function(z) { quantile(z,0.9) }, 
			                alpha=0.3) +
			   stat_summary(geom="ribbon", 
			                fun.y=function(z) { quantile(z,0.5) }, 
			                fun.ymin=function(z) { quantile(z,0.25) }, 
			                fun.ymax=function(z) { quantile(z,0.75) }, 
			                alpha=0.8) + 
			   stat_summary(geom="line", 
			                fun.y=function(z) { quantile(z,0.5) }) +
			   scale_color_manual(values = fill_colors, 
	                              labels = legend_scenarios) +
	           scale_fill_manual(values = fill_colors,  
	                             labels = legend_scenarios) +
			   # scale_x_continuous(breaks=seq(0, 300, 50)) +
			   scale_y_continuous(limits = c(0, 1), breaks=seq(0, 1, by=.2)) + 
			   theme_bw() + 
			   the_theme

	output_dir = "/Users/hn/Desktop/"
	out_name = paste0("pop_diff_cloud_separate_", model, ".png")
	ggsave(out_name, plot_sep, path=output_dir, width=14, height=10, unit="in", dpi=400)

	#################################
	###########
	########### line plots
	###########
	#################################
	the_theme <- theme(# panel.grid.major = element_line(size = 0.2),
			           # panel.spacing=unit(1, "cm"),
			           plot.margin = unit(c(t = 0, r = 1, b = 0, l = 1), "cm"),
			           panel.spacing.x = unit(1, "cm"),
			           # panel.spacing.y = unit(1, "lines"),
			           panel.grid.major = element_blank(),
	                   panel.grid.minor = element_blank(),
			           legend.title = element_blank(),
	  	  	           legend.text = element_text(size=10),
			           legend.position = "bottom",
			           strip.text = element_text(size=12, face="plain"),
			           axis.text = element_text(face="plain", size=10, color="black"),
			           axis.ticks = element_line(color = "black", size = .2),
			           axis.title.x = element_text(face="plain", size=16, margin=margin(t=10, r=0, b=0, l=0)),
			           axis.title.y = element_text(face="plain", size=16, margin=margin(t=0, r=10, b=0, l=0))
			           )
	line_plot <- ggplot(curr_data, 
	                    aes(x=year, y=value, fill=factor(ClimateScenario), color=factor(ClimateScenario))) +
	                    labs(x = "Year", y = "pop. diff.") +
	                    facet_grid(. ~ CountyGroup, scales="free") +
	                    stat_summary(geom="line", 
	                                 fun.y=function(z) { quantile(z,0.5) }) +
	                    scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, by=.2)) + 
	                    theme_bw() + the_theme

	out_name = paste0("pop_diff_line_", model, ".png")
	ggsave(out_name, line_plot, path=output_dir, width=14, height=7, unit="in", dpi=400)

	line_plot_sep <- ggplot(curr_data, 
				     aes(x=year, y=value, fill=factor(ClimateScenario), color=factor(ClimateScenario))) +
				     labs(x = "Year", y = "pop. diff.") +
	 			     facet_grid(. ~ClimateScenario ~ CountyGroup, scales="free") +
				     stat_summary(geom="line", 
				                  fun.y=function(z) { quantile(z,0.5) }) +
				     # scale_x_continuous(breaks=seq(0, 300, 50)) +
				     scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, by=.2)) + 
				     theme_bw() + the_theme
	out_name = paste0("pop_diff_line_sep_", model, ".png")
	ggsave(out_name, line_plot_sep, path=output_dir, width=14, height=10, unit="in", dpi=400)

}