#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

#data_dir = getwd()
data_dir = "/home/kiran/giridhar/codmoth_pop"
filename <- paste0(data_dir, "/allData.rds")
data <- data.table(readRDS(filename))
data = data[, .(PercAdultGen1 = median(PercAdultGen1), PercAdultGen2 = median(PercAdultGen2), PercAdultGen3 = median(PercAdultGen3), PercAdultGen4 = median(PercAdultGen4), PercLarvaGen1 = median(PercLarvaGen1), PercLarvaGen2 = median(PercLarvaGen2), PercLarvaGen3 = median(PercLarvaGen3), PercLarvaGen4 = median(PercLarvaGen4), CumDDinC = median(CumDDinC), CumDDinF = median(CumDDinF)), by = c("ClimateGroup", "year", "month", "day", "dayofyear")]
#data <- subset(data, select = c("ClimateGroup", "month", 
#		"PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4", 
#		"PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"))

#data_melted = melt(data, c("ClimateGroup", "month"), 
#                   c("PercAdultGen1", "PercAdultGen2", "PercAdultGen3", "PercAdultGen4", 
#                     "PercLarvaGen1", "PercLarvaGen2", "PercLarvaGen3", "PercLarvaGen4"), 
#                   variable.name = "Generations")

#p = ggplot(data = data_melted, aes(x = Generations, y = value, fill = ClimateGroup)) +
#  geom_boxplot() + coord_flip() +
#  facet_wrap(~month)

#saveRDS(p, paste0(data_dir, "/", "popplot.rds"))
saveRDS(data, paste0(data_dir, "/", "subData.rds"))
