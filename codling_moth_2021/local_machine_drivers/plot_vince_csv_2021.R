# rm(list=ls())

library(data.table)
library(ggplot2)
library(dplyr)
library(ggpubr) # for ggarrange
library("readxl")

file_dir <- file.path("/Users/hn/Documents/00_GitHub/Ag/Vince_CSV/Vince_PestControl/")
out_dir_base <- "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/pest_control/"

file_name = "MDpesticide_effects.csv"
file_name = "Simulation_Results_2021.xls"

if (file_name == "MDpesticide_effects.csv"){
  out_dir <- paste0(out_dir_base, "old_pestSimulations/")
  file_name = paste0(file_dir, file_name)
  data = data.table(read.csv(file_name))
  } else {
    file_name = paste0(file_dir, file_name)
    out_dir <- paste0(out_dir_base, "2021_PestSimulations/")
    data = data.table(read_excel(file_name))
}


##########################################
##### Below we have the plot for 
##### both rcp's combined.
#####
##########################################
needed_cols = c("mdeffect", "ceffectnmd", "ceffectmd",
                "loc", "scenario", "year")
data = subset(data, select=needed_cols)

data$time_period[data$year >= 1979 & data$year <= 2005] <- "Historical"
data$time_period[data$year >= 2025 & data$year <= 2055] <- "2040s"
data$time_period[data$year >= 2045 & data$year <= 2075] <- "2060s"
data$time_period[data$year >= 2065 & data$year <= 2095] <- "2080s"

# There are years between 2006 and 2015 which ... becomes NA
# The second line is a better approach, it just drops
# the rows containing NA in the given column.

# data <- na.omit(data)
data <- data[!is.na(data$time_period),]

time_levels = c("Historical", "2040s", "2060s", "2080s")
data$time_period <- factor(data$time_period, levels = time_levels, ordered = TRUE)

city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
data$loc <- factor(data$loc, levels = city_levels, ordered = TRUE)

setnames(data, old=c("mdeffect", "ceffectnmd", "ceffectmd"), 
               new=c("MD",       "Spray",       "MD & Spray"))

data_45 = filter(data, scenario %in% c("Historical", "RCP4.5"))
data_85 = filter(data, scenario %in% c("Historical", "RCP8.5"))

data_45$scenario = "RCP4.5" # factor(data_45$scenario)
data_85$scenario = "RCP8.5" # factor(data_85$scenario)

data_45 <- data_45 %>% 
           # rename(MD = mdeffect, Spray = ceffectnmd, MD_Spray = ceffectmd) %>% 
           select(-c(year)) %>% # , scenario
           data.table()

data_85 <- data_85 %>% 
           # rename(MD = mdeffect, Spray = ceffectnmd, MD_Spray = ceffectmd) %>% 
           select(-c(year)) %>% # , scenario
           data.table()

data_45 = melt(data_45, id = c("loc", "time_period", "scenario"))
data_85 = melt(data_85, id = c("loc", "time_period", "scenario"))


data_45$variable <- factor(data_45$variable, ordered = TRUE)
data_85$variable <- factor(data_85$variable, ordered = TRUE)

######
###### Plot's cosmetic settings
######
color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
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

vince_45 <- ggplot(data = data_45, aes(x=variable, y=value), group = variable) + 
            geom_boxplot(outlier.size=-.05, notch=FALSE, width=.7, lwd=.25, aes(fill=time_period), 
                         position=position_dodge(width=0.8)) + 
            scale_x_discrete(expand=c(0.2, .1), limits = levels(data$variable[1])) +
            facet_grid(~scenario ~ loc, scales="free") + 
            labs(x="control method", 
                 y="% of untreated control population remaining", 
                 color = "Time Period") + 
            # ggtitle(label = paste0("% of untreated control population RCP 4.5")) + 
            scale_color_manual(values=color_ord,
                               name="Time\nPeriod", 
                               limits = color_ord,
                               labels=c("Historical", "2040s", "2060s", "2080s")) +
            scale_fill_manual(values=color_ord,
                              name="Time\nPeriod", 
                              labels=c("Historical", "2040s", "2060s", "2080s")) +
            the_theme

ggsave(filename = "00_controlMethods_rcp45.png", 
       path = out_dir, 
       plot = vince_45,
       width = 9, height = 3, units = "in",
       dpi = 400, 
       device = "png")


vince_85 <- ggplot(data = data_85, aes(x=variable, y=value), group = variable) + 
                geom_boxplot(outlier.size=-.05, notch=FALSE, width=.7, lwd=.25, aes(fill=time_period), 
                             position=position_dodge(width=0.8)) + 
                scale_x_discrete(expand=c(0.2, .1), limits = levels(data$variable[1])) +
                facet_grid(~scenario ~loc, scales="free") + 
                labs(x="control method", 
                     y="% of untreated control population remaining", 
                     color = "Time Period") + 
                # ggtitle(label = paste0("% of untreated control population RCP 8.5")) +
                scale_color_manual(values=color_ord,
                                   name="Time\nPeriod", 
                                   limits = color_ord,
                                   labels=c("Historical", "2040s", "2060s", "2080s")) +
                scale_fill_manual(values=color_ord,
                                  name="Time\nPeriod", 
                                  labels=c("Historical", "2040s", "2060s", "2080s")) +
                the_theme

ggsave(filename = "00_controlMethods_rcp85.png", 
       path = out_dir, 
       plot = vince_85,
       width = 9, height = 3, units = "in",
       dpi = 400, 
       device = "png")


# big_plot <- ggarrange(vince_45, vince_85,
#                       label.x = "control method", label.y = "% of untreated control population remaining",
#                       ncol = 1, nrow = 2, 
#                       common.legend = T, legend = "bottom")
# ggsave(filename = "ggarrange_controlMethods_rcp45_85.png", 
#        path = out_dir, 
#        plot = big_plot,
#        width = 9, height = 6, units = "in",
#        dpi = 600, 
#        device = "png")
#
# Plot the above all at one with facet
#
#


color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
axis_text_size <- 8
strip_text_size <- 8
plot_title_font_size <- 10
legent_font_size <- 10
the_theme <- theme(plot.margin = unit(c(t = 0.2, r = 0.2, b = -0.2, l = 0.2), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(size = plot_title_font_size, face = "bold"),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
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

data_for_facet <- data %>% 
                 select(-c(year))

data_45 = filter(data_for_facet, scenario %in% c("Historical", "RCP4.5"))
data_85 = filter(data_for_facet, scenario %in% c("Historical", "RCP8.5"))

data_45$scenario <- "RCP 4.5"
data_85$scenario <- "RCP 8.5"

data_for_facet <- rbind(data_85, data_45)
data_for_facet$scenario <- factor(data_for_facet$scenario, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)
data_for_facet <- data.table(data_for_facet)

data_for_facet = melt(data_for_facet, id = c("loc", "time_period", "scenario"))
data_for_facet$variable <- factor(data_for_facet$variable, ordered = TRUE)

vince_box_45_85 <- ggplot(data = data_for_facet, aes(x=variable, y=value), group = variable) + 
                   geom_boxplot(outlier.size=-.15, notch=FALSE, width=.7, lwd=.25, aes(fill=time_period), 
                                position=position_dodge(width=0.8)) + 
                   scale_x_discrete(expand=c(0.2, .1), limits = levels(data$variable[1])) +
                   facet_grid(~ scenario ~loc , scales="free") + 
                   labs(x="control method", y="% of untreated control population remaining", color = "Time Period") + 
                   # ggtitle(label = paste0("% of the untreated control population")) + 
                   scale_color_manual(values=color_ord,
                                      name="Time\nPeriod", 
                                      limits = color_ord,
                                      labels=c("Historical", "2040s", "2060s", "2080s")) +
                   scale_fill_manual(values=color_ord,
                                     name="Time\nPeriod", 
                                     labels=c("Historical", "2040s", "2060s", "2080s")) +
                   the_theme

ggsave(filename = "00_facet_controlMethods_rcp45_85.png", 
       path = out_dir, 
       plot = vince_box_45_85,
       width = 9, height = 5, units = "in",
       dpi = 400, 
       device = "png")

################################################################################
################################################################################
################################################################################

color_ord = c("grey47", "dodgerblue", "olivedrab4", "red")
axis_text_size <- 8
strip_text_size <- 8
plot_title_font_size <- 10
legent_font_size <- 10
the_theme <- theme(plot.margin = unit(c(t = 0.2, r = 0.2, b = -0.2, l = 0.2), "cm"),
                   panel.border = element_rect(fill=NA, size=.3),
                   plot.title = element_text(size = plot_title_font_size, face = "bold"),
                   panel.grid.major = element_line(size = 0.05),
                   panel.grid.minor = element_blank(),
                   panel.spacing=unit(.25,"cm"),
                   legend.position="bottom", 
                   legend.title = element_blank(),
                   legend.key.size = unit(1, "line"),
                   legend.text = element_text(size = legent_font_size),
                   legend.margin = margin(t= .1, r=0, b=.5, l=0, unit = 'cm'),
                   legend.spacing.x = unit(.05, 'cm'),
                   strip.text.x = element_text(size = strip_text_size, face="bold"),
                   strip.text.y = element_text(size = strip_text_size, face="bold"),
                   axis.ticks = element_line(color = "black", size = .2),
                   axis.ticks.x = element_blank(),
                   axis.text.x = element_blank(),
                   axis.text.y = element_text(size = axis_text_size, face = "plain", color="black"),
                   axis.title.x = element_blank(),
                   axis.title.y = element_text(face = "bold", size = axis_text_size, margin = margin(t=0, r=6, b=0, l=0)),
                  )

MS_Spray <- ggplot(data = data_for_facet[data_for_facet$variable=="MD & Spray", ], aes(x=variable, y=value), group = variable) + 
            geom_boxplot(outlier.size=-.04, notch=FALSE, width=.7, lwd=.3, aes(fill=time_period), 
                         position=position_dodge(width=0.8)) + 
            scale_y_continuous(breaks=seq(0, 20, by=5)) +
            facet_grid(~ scenario ~ loc, scales="free") + 
            labs(x = element_blank(), y="% of untreated control population remaining", color = "Time Period") + 
            # ggtitle(label = paste0("Control method is based on MD and spray")) +
            scale_color_manual(values=color_ord,
                               name="Time\nPeriod", 
                               limits = color_ord,
                               labels=c("Historical", "2040s", "2060s", "2080s")) +
            scale_fill_manual(values=color_ord,
                              name="Time\nPeriod", 
                              labels=c("Historical", "2040s", "2060s", "2080s")) +
            the_theme

ggsave(filename = "00_MS_Spray_hiRes.png", 
       path = out_dir, 
       plot = MS_Spray,
       width = 9, height = 5, units = "in",
       dpi = 400,
       device = "png")


#########
######### add text to the above plot This is not updated. It is from 2018
#########

data_back_MDSpray <- data_back[data_back$variable=="MD_Spray", ]
data_back_MDSpray <- within(data_back_MDSpray, remove(variable))

df <- data.frame(data_back_MDSpray)
df <- (df %>% group_by(time_period, RCP, loc))
medians <- (df %>% summarise(med = median(value)))
medians <- data.table(medians)
medians <- setnames(medians, old="time_period", new="variable")


data_back_MDSpray <- setnames(data_back_MDSpray, old="time_period", new="variable")

MS_Spray <- ggplot(data = data_back_MDSpray, aes(x=variable, y=value), group = variable) + 
            geom_boxplot(outlier.size=-.01, notch=FALSE, width=.7, lwd=.55, aes(fill=variable), 
                         position=position_dodge(width=0.8)) + 
            scale_y_continuous(breaks=seq(0, 20, by=5)) +
            facet_grid(~ RCP ~ loc, scales="free") + 
            labs(x = element_blank(),
                 y="% of untreated control population remaining", 
                 color = "Time Period") + 
            # ggtitle(label = paste0("Control method is based on MD and spray")) +
            scale_color_manual(values=color_ord,
                               name="Time\nPeriod", 
                               limits = color_ord,
                               labels=c("Historical", "2040s", "2060s", "2080s")) +
            scale_fill_manual(values=color_ord,
                              name="Time\nPeriod", 
                              labels=c("Historical", "2040s", "2060s", "2080s")) +
            the_theme +
            geom_text(data = medians, 
                        aes(label = sprintf("%1.0f", medians$med), y=medians$med), 
                            colour = "black", fontface = "plain", size=8, 
                            position = position_dodge(0.08), vjust = -0.2)


ggsave(filename = "MS_Spray_hiRes_wText.png", 
       path = out_dir, 
       plot = MS_Spray,
       width = 14, height = 10, units = "in", # width = 14, height = 10 
       dpi = 400,
       device = "png")



#########
######### create table of statistics for it.
#########

data_back_MDSpray <- data_back_MDSpray[!is.na(data_back_MDSpray$value),]


