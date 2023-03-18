
rm(list=ls())
library(data.table)
library(dplyr)
library(ggpubr)

input_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
plot_path <- "/Users/hn/Desktop/"
versions <- c("rcp45", "rcp85")
stage <- c("larva") # "adult", 

version = versions[1]

color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")

output_name = paste0(stage, "_flight_", version, "_half.png")
input_name = paste0("combined_CM_", version, ".rds")
data <- readRDS(paste0(input_dir, input_name))

data <- subset(data, select = c("LGen1_0.5", "LGen2_0.5", "LGen3_0.5", "LGen4_0.5",
                                "ClimateGroup", "CountyGroup"))


data$CountyGroup = as.character(data$CountyGroup)
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'


L = c("Gen. 1", "Gen. 2", "Gen. 3", "Gen. 4")
data_melted = melt(data, id = c("ClimateGroup", "CountyGroup"))
data_melted[data_melted$variable=="LGen1_0.5"]$variable = L[1]
data_melted[data_melted$variable=="LGen2_0.5"]$variable = L[2]
data_melted[data_melted$variable=="LGen3_0.5"]$variable = L[3]
data_melted[data_melted$variable=="LGen4_0.5"]$variable = L[4]
data_melted$variable <- factor(data_melted$variable, levels = L, ordered = TRUE)

rm(data)

bplot <- ggplot(data = data_melted, aes(x=variable, y=value), group = variable) + 
         geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                      position=position_dodge(width=0.5)) + 
         scale_y_continuous(limits = c(80, 370), breaks = seq(100, 360, by = 50)) +
         facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
         labs(x="Time period", y="Julian day", color = "Climate Group") + 
         theme(legend.position="bottom", 
               legend.margin=margin(t=-.1, r=0, b=5, l=0, unit = 'cm'),
               legend.title = element_blank(),
               legend.text = element_text(size=7, face="plain"),
               legend.key.size = unit(.5, "cm"), 
               panel.grid.major = element_line(size = 0.1),
               panel.grid.minor = element_line(size = 0.1),
               panel.spacing=unit(.5, "cm"),
               strip.text = element_text(size= 6, face = "plain"),
               axis.text = element_text(face = "plain", size = 4, color="black"),
               axis.ticks = element_line(color = "black", size = .2),
               axis.title.x = element_text(face = "plain", size = 10, 
                                           margin = margin(t=10, r=0, b=0, l=0)),
               axis.text.x = element_text(size = 6, color="black"), # tick text font size
               axis.title.y = element_text(face = "plain", size = 10, 
                                           margin = margin(t=0, r=7, b=0, l=0)),
               axis.text.y  = element_blank(),
               axis.ticks.y = element_blank(),
               plot.margin = unit(c(t=-0.35, r=.7, b=-4.7, l=0.3), "cm")
              ) +
         scale_color_manual(values=color_ord,
                            name="Time\nPeriod", 
                            limits = color_ord,
                            labels=c("Historical", "2040", "2060", "2080")) +
         scale_fill_manual(values=color_ord,
                           name="Time\nPeriod", 
                           labels=c("Historical", "2040", "2060", "2080")) + 
         coord_flip()
#bplot <- add_sub(bplot, label="Gen. 1", x=1.02, y=8, angle=270, size=6, fontface="plain")
ggsave(output_name, bplot, device="png", 
       path=plot_path, width=plot_with, height=plot_height, 
       unit="in", dpi=500)
