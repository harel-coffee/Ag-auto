
library(data.table)
library(ggplot2)
library(dplyr)

#################
#################
#################
data_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/01_combined_files/"

CMPOP_rcp45 <- readRDS(paste0(data_dir, "LM_combined_CMPOP_rcp45.rds"))
CMPOP_rcp85 <- readRDS(paste0(data_dir, "LM_combined_CMPOP_rcp85.rds"))
CMPOP_observed <- readRDS(paste0(data_dir, "LO_combined_CMPOP_observed.rds"))

needed_cols <- c("model", "city", "dayofyear", "CumDDinF", "emission", "time_period")
CMPOP_rcp45 <- subset(CMPOP_rcp45, select = needed_cols)
CMPOP_rcp85 <- subset(CMPOP_rcp85, select = needed_cols)
CMPOP_observed <- subset(CMPOP_observed, select = needed_cols)
CMPOP_observed$time_period <- "Historical"

# 
#  change emission in observed so that thet appear
#  in both emission scenario plots
#

CMPOP_observed_45 <- CMPOP_observed
CMPOP_observed_85 <- CMPOP_observed

CMPOP_observed_45$emission <- "RCP 4.5"
CMPOP_observed_85$emission <- "RCP 8.5"

data <- rbind(CMPOP_rcp45, CMPOP_rcp85, 
              CMPOP_observed_45, CMPOP_observed_85)

#
#    old code was wrong. We did not cut the last day of each quarter!
#
# data[, season := as.character(0)]
# data[data[ , data$dayofyear <= 90]]$season = "Qr. 1"
# data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "Qr. 2"
# data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "Qr. 3"
# data[data[ , data$dayofyear >= 274]]$season = "Qr. 4"
# data = within(data, remove(dayofyear))
# data$season = factor(data$season, levels = c("Qr. 1", "Qr. 2", "Qr. 3", "Qr. 4"))
#
#

#
#   We need to look at only the last day. 
#   so, keep only the last day!
#
cumdd_in_QRT_1 <- data %>% filter(dayofyear == 90)  %>% data.table()
cumdd_in_QRT_2 <- data %>% filter(dayofyear == 181) %>% data.table()
cumdd_in_QRT_3 <- data %>% filter(dayofyear == 273) %>% data.table()
cumdd_in_QRT_4 <- data %>% filter(dayofyear == 365) %>% data.table()

cumdd_in_QRT_1$season <- "Qr. 1"
cumdd_in_QRT_2$season <- "Qr. 2"
cumdd_in_QRT_3$season <- "Qr. 3"
cumdd_in_QRT_4$season <- "Qr. 4"

data <- rbind(cumdd_in_QRT_1, cumdd_in_QRT_2, 
              cumdd_in_QRT_3, cumdd_in_QRT_4)
data <- subset(data, select = c("time_period", "city", "season", "emission", "CumDDinF"))

data = melt(data, id = c("time_period", "city", "season", "emission"))
data = within(data, remove(variable))

#
#   Take care of order of factors:
#
time_levels = c("Historical", "2040s", "2060s", "2080s")
data$time_period <- factor(data$time_period, levels = time_levels, ordered = TRUE)

data$season = factor(data$season, 
                     levels = c("Qr. 1", "Qr. 2", "Qr. 3", "Qr. 4"),
                     order = TRUE)

city_levels = c("East Oroville", "Wenatchee", "Quincy", "Wapato", "Richland")
data$city <- factor(data$city, levels = city_levels, ordered = TRUE)

data$emission <- factor(data$emission, levels = c("RCP 4.5", "RCP 8.5"), ordered = TRUE)

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
                   legend.margin = margin(t= -0.6, r=0, b=.5, l=0, unit = 'cm'),
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


bplot <- ggplot(data = data, aes(x=season, y=value), group = season) + 
         geom_boxplot(outlier.size=-.15, notch=FALSE, width=.7, lwd=.25, aes(fill=time_period), 
                      position=position_dodge(width=0.8)) + 

         # scale_y_continuous(limits = c(0, 6000), breaks = seq(0, 6000, by = 1000)) + 
         facet_grid(~ emission ~ city, scales="free") + 
         labs(x="", y="cumulative degree days (in F)", color = "time period") + 
         # theme_bw() +
         scale_color_manual(values=color_ord,
                            name="Time\nPeriod", 
                            limits = color_ord,
                            labels=c("Historical", "2040s", "2060s", "2080s")) +
        scale_fill_manual(values=color_ord,
                          name="Time\nPeriod", 
                          labels=c("Historical", "2040s", "2060s", "2080s")) +
        the_theme

out_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/03_plots_4_paper/"

ggsave(filename = "cumDD_quarterly.png", 
       path = out_dir, 
       plot = bplot,
       width = 9, height = 5, units = "in",
       dpi = 400, 
       device = "png")



