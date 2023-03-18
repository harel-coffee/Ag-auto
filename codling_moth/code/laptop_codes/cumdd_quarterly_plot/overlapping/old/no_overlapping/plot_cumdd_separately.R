.libPaths("/data/hydro/R_libs35")
.libPaths()
library(tidyverse)
library(lubridate)

library(ggpubr)
library(data.table)
library(ggplot2)
library(dplyr)

start_time <- Sys.time()

# input_dir = "/Users/hn/Documents/01_research_data/my_aeolus_2015/all_local/4_cumdd/"
# input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/cumdd_data/"
version = c("rcp45", "rcp85")

for (vers in version){
  print (vers)
  if (vers == "rcp45"){
    plot_title <- "RCP 4.5"
    } else{
      plot_title <- "RCP 8.5"
  }
  
  data = readRDS(paste0("./", "cumdd_CMPOP_", vers, ".rds"))
  data = subset(data, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumDDinF"))
  data$CountyGroup = as.character(data$CountyGroup)
  data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
  data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
  # add the new season column
  data[, season := as.character(0)]
  data[data[ , data$dayofyear <= 90]]$season = "QTR 1"
  data[data[ , data$dayofyear >= 91 & data$dayofyear <= 181]]$season = "QTR 2"
  data[data[ , data$dayofyear >= 182 & data$dayofyear <= 273]]$season = "QTR 3"
  data[data[ , data$dayofyear >= 274]]$season = "QTR 4"
  data = within(data, remove(dayofyear))
  data$season = factor(data$season, levels = c("QTR 1", "QTR 2", "QTR 3", "QTR 4"))

  #df <- data.frame(data)
  #df <- (df %>% group_by(CountyGroup, ClimateGroup, season))
  #medians <- (df %>% summarise(med = median(CumDDinF)))
  #medians <- medians$med
  #rm(df)

  data = melt(data, id = c("ClimateGroup", "CountyGroup", "season"))
  data = within(data, remove(variable))

  up_limt <- max(data$value)

  bplot <- ggplot(data = data, aes(x=season, y=value), group = season) + 
           geom_boxplot(outlier.size=-.15, notch=FALSE, width=.4, lwd=.25, aes(fill=ClimateGroup), 
                 position=position_dodge(width=0.5)) + 
           scale_y_continuous(breaks = seq(0, up_limt, by = 1000)) + # limits = c(0, 6000),
           facet_wrap(~CountyGroup, scales="free", ncol=6, dir="v") + 
           labs(x="", y="Cumulative degree day", color = "Climate Group") + 
           theme_bw() +
           theme(plot.title = element_text(size = 22),
                 legend.position="bottom", 
                 legend.margin=margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
                 legend.title = element_blank(),
                 legend.text = element_text(size=18, face="plain"),
                 legend.key.size = unit(1, "cm"), 
                 legend.spacing.x = unit(0.5, 'cm'),
                 panel.grid.major = element_line(size = 0.1),
                 panel.grid.minor = element_line(size = 0.1),
                 strip.text = element_text(size= 20, face = "plain"),
                 axis.text = element_text(face = "plain", size = 10, color="black"),
                 axis.title.x = element_text(face = "plain", size = 10, 
                                             margin = margin(t=10, r=0, b=0, l=0)),
                 axis.text.x = element_text(size = 18, color="black"), # tick text font size
                 axis.text.y = element_text(size = 18, color="black"),
                 axis.title.y = element_text(face = "plain", size = 22, 
                                             margin = margin(t=0, r=7, b=0, l=0)),
                 plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
                ) + 
           ggtitle(label = plot_title)
  
  assign(x = paste0("cumdd_plot_", vers), value ={bplot})
  rm(bplot)
}
rm(data)

cumdd_plot_qrt <- ggarrange(plotlist = list(cumdd_plot_rcp45, cumdd_plot_rcp85),
                            ncol = 2, nrow = 1,
                            common.legend = TRUE)

ggsave("cumdd_qrt_sep.png", cumdd_plot_qrt, width=15, height=7, unit="in", path="./", dpi=300, device="png")

print( Sys.time() - start_time)



