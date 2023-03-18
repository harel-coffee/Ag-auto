rm(list=ls())
library(chron)
library(data.table)
library(ggplot2)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(cowplot)

plot_path <- "/Users/hn/Desktop/Kirti/check_point/my_aeolus_rds/"
output_name = "test.png"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "red")

combined_CM45 <- readRDS("/Users/hn/Desktop/Kirti/check_point/my_aeolus_rds/combined_CM_rcp45.rds")
data <- combined_CM45
data <- subset(data, select = c("AGen1_0.25", "AGen1_0.5", "AGen1_0.75", 
                                "AGen2_0.25", "AGen2_0.5", "AGen2_0.75",
                                "AGen3_0.25", "AGen3_0.5", "AGen3_0.75", 
                                "AGen4_0.25", "AGen4_0.5", "AGen4_0.75", 
                                "ClimateGroup", "CountyGroup"))

data$CountyGroup = as.character(data$CountyGroup)
data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

data_melted = melt(data, id = c("ClimateGroup", "CountyGroup"))

L = c('AGen1_0.25','AGen2_0.25', 'AGen3_0.25','AGen4_0.25',
      'AGen1_0.5','AGen2_0.5', 'AGen3_0.5','AGen4_0.5',
      'AGen1_0.75','AGen2_0.75', 'AGen3_0.75','AGen4_0.75')

data_melted$variable <- factor(data_melted$variable, levels = L, ordered = TRUE)
# df <- data.frame(data)
# df <- (df %>% group_by(CountyGroup, ClimateGroup))
# medians <- (df %>% summarise(med = median(AGen1_0.25)))
# medians_vec <- medians$med

bplot <- ggplot(data = data_melted, aes(x=variable, y=value), group = variable) + 
  geom_boxplot(outlier.size=0, notch=FALSE, width=.2, aes(fill=ClimateGroup), position=position_dodge(width=0.5)) + 
  scale_y_continuous(limits = c(80, 370), breaks = seq(100, 360, by = 50)) +
  geom_vline(xintercept=4.5, linetype="solid", color = "grey", size=1)+
  geom_vline(xintercept=8.5, linetype="solid", color = "grey", size=1)+
  annotate("text", x=2.5, y=369, angle=270, label= "boat", size=8, fontface="plain") + 
  # scale_x_discrete(expand=c(0, 2), limits = levels(data_melted$ClimateGroup[1])) +
  facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
  labs(x="Time Period", y="Day of Year", color = "Climate Group", title=factor(data_melted$CountyGroup)) + 
  theme_bw() +
  theme(legend.position="bottom", 
        legend.margin=margin(t=-.1, r=0, b=0, l=0, unit = 'cm'),
        legend.title = element_blank(),
        legend.text = element_text(size=23, face="plain"),
        legend.key.size = unit(2, "cm"), 
        panel.grid.major = element_line(size = 0.5),
        panel.grid.minor = element_line(size = 0.5),
        strip.text = element_text(size= 17, face = "plain"),
        axis.text = element_text(face = "plain", size = 10),
        axis.title.x = element_text(face = "plain", size = 30, 
                                    margin = margin(t=15, r=0, b=0, l=0)),
        axis.text.x = element_text(size = 17),
        axis.title.y = element_text(face = "plain", size = 30, 
                                    margin = margin(t=0, r=15, b=0, l=0)),
        axis.text.y  = element_blank(),
        axis.ticks.y = element_blank(),
        plot.margin = unit(c(t=-0.35, r=2, b=-5, l=0.5), "cm")
  ) +
  scale_color_manual(values=color_ord,
                     name="Time\nPeriod", 
                     limits = color_ord,
                     labels=c("Historical","2040","2060","2080")) +
  scale_fill_manual(values=color_ord,
                    name="Time\nPeriod", 
                    labels=c("Historical","2040","2060","2080")) + 
  coord_flip()

p2 <- add_sub(bplot, label="25%", x=0, vjust=-95, hjust=7.7, angle=270, size=20, fontface = "bold")
p2 <- add_sub(p2, label="50%", x=0, vjust=-95, hjust=15.8, angle=270, size=20, fontface = "bold")
p2 <- add_sub(p2, label="75%", x=0, vjust=-95, hjust=24.2, angle=270, size=20, fontface = "bold")

#p2 <- add_sub(p2, label="Gen. 1", x=0, vjust=-92, hjust=24.2, angle=270, size=20, fontface = "bold")
#p2 <- add_sub(p2, label="Gen. 2", x=0, vjust=-92, hjust=24.2, angle=270, size=20, fontface = "bold")
#p2 <- add_sub(p2, label="Gen. 3", x=0, vjust=-92, hjust=24.2, angle=270, size=20, fontface = "bold")
#p2 <- add_sub(p2, label="Gen. 4", x=0, vjust=-92, hjust=24.2, angle=270, size=20, fontface = "bold")

output_name = "remove_fill_h5.png"
ggsave(output_name, p2, device="png", path=plot_path, width=20, height=14, unit="in")

