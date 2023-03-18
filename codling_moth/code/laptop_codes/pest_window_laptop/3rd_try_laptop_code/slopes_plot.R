##############################################################
###############################
############################### bar plots of slopes!
###############################
##############################################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
part_2 = "pest_control/3rd_try_data/safe_no_countyGroup/"
part_2 = "pest_control/3rd_try_data/"
data_dir = paste0(part_1, part_2)

part_1 <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/"
part_2 <- "pest_window/3rd_try_plots/"
plot_dir <- paste0(part_1, part_2)

name_pref = "slopes_"
models = c("45", "85")

climate_scenarios = c("historical", "bcc-csm1-1-m", 
                    "BNU-ESM", "CanESM2", "CNRM-CM5",
                    "GFDL-ESM2G", "GFDL-ESM2M")
for (model in models){
  curr_data <- data.table(readRDS(paste0(data_dir, name_pref, model, ".rds")))
  curr_data <- within(curr_data, remove(location))
  curr_data <- within(curr_data, remove(CountyGroup))
  curr_data[curr_data$month == "april"]$month = "April"
  curr_data[curr_data$month == "june"]$month = "June"
  curr_data[curr_data$month == "august"]$month = "August"

  curr_data <- melt(curr_data, id="month")
  #curr_data$month = factor(curr_data$month)
  curr_data$month = factor(curr_data$month, levels=c('April','June','August'))


  br_plt <- ggplot(data = curr_data, aes(x=variable, y=value, fill = variable)) +
            geom_bar(stat="identity", width = 0.1, position=position_dodge()) + 
            facet_grid(. ~ month, scales = "free") +
            labs(y="slope of regression line") +
            theme_bw() + 
            theme(panel.spacing=unit(.5, "cm"),
                  axis.text.y = element_text(size = 9, angle=90, color="black"),
                  axis.text.x = element_text(size = 9, color="black", angle=45, 
                                             margin=margin(t=10,r=5,b=-5,l=-5,"pt")),
                  axis.title.x=element_blank(),
                  legend.position="bottom",
                  legend.spacing.x = unit(.1, 'cm'),
                  legend.title=element_blank(),
                  legend.text=element_text(size=10),
                  legend.key.size = unit(.4, "cm"),
                  panel.grid.major = element_line(size = 0.05),
                  panel.grid.minor = element_line(size = 0.2)) +
            scale_x_discrete(labels=c("bcc-csm1-1-m" = "bcc", 
                                      "BNU-ESM" = "BNU",
                                      "CanESM2" = "Canada",
                                      "CNRM-CM5" = "CNRM",
                                      "GFDL-ESM2G" = "GF-G",
                                      "GFDL-ESM2M" = "GF-M")) +
            scale_y_continuous(limits = c(-10, 20), breaks=seq(-10, 20, by=5)) +
            scale_fill_manual(breaks=c("historical", "bcc-csm1-1-m", "BNU-ESM", "CanESM2",
                                       "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M"),
                              labels=c("historical", "bcc", "BNU", "Canada",
                                       "CNRM", "GF-G", "GF-M"),
                              values = c("yellow4", "red4", "steelblue4",
                                         "darkcyan", "grey36", "goldenrod4", "red") 
                              ) + 
            # the line below is supposed to make the legend to be printed in 1 line
            guides(color = guide_legend(nrow = 1))
  out_name <- paste0(model, "_bar_plot.png")
  ggsave(out_name, br_plt, path=plot_dir, 
         dpi=500, device="png", width=10.5, height=3.1, unit="in")
}
###################################
###################################
################################### Cloud of medians of slopes
###################################
###################################
rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
part_2 = "pest_control/3rd_try_data/safe_no_countyGroup/"
part_2 = "pest_control/3rd_try_data/"
data_dir = paste0(part_1, part_2)

part_1 <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/"
part_2 <- "pest_window/3rd_try_plots/"
plot_dir <- paste0(part_1, part_2)

name_pref = "slopes_"
models = c("45", "85")

climate_scenarios = c("historical", "bcc-csm1-1-m", 
                      "BNU-ESM", "CanESM2", "CNRM-CM5",
                      "GFDL-ESM2G", "GFDL-ESM2M")
model = models[1]
for (model in models){
  curr_data <- data.table(readRDS(paste0(data_dir, name_pref, model, ".rds")))
  curr_data <- within(curr_data, remove(CountyGroup))
  colnames(curr_data) <- c("location", "historical", "bcc", 
                           "BNU", "Canada", "CNRM", "GFDL_G", 
                           "GFDL_M", "month")
  #curr_data <- within(curr_data, remove(location))
  Ah <- curr_data[, .(CumDD_median = median(historical)), by = c("month", "location")]
  Ah$scenario = "historical"
  
  A1 <- curr_data[, .(CumDD_median = median(bcc)), by = c("month", "location")]
  A1$scenario = "bcc"
  
  A2 <- curr_data[, .(CumDD_median = median(BNU)), by = c("month", "location")]
  A2$scenario = "BNU"
  
  A3 <- curr_data[, .(CumDD_median = median(Canada)), by = c("month", "location")]
  A3$scenario = "Canada"
  
  A4 <- curr_data[, .(CumDD_median = median(CNRM)), by = c("month", "location")]
  A4$scenario = "CNRM"
  
  A5 <- curr_data[, .(CumDD_median = median(GFDL_G)), by = c("month", "location")]
  A5$scenario = "GFDL_G"
  
  A6 <- curr_data[, .(CumDD_median = median(GFDL_M)), by = c("month", "location")]
  A6$scenario = "GFDL_M"
  
  medians <- rbind(Ah, A1, A2, A3, A4, A5, A6)

  ggplot(medians, aes(x=, y=CumDD_median, fill=factor(scenario))) +
         labs(x = "Year", y = "Cumulative degree days (in F)", fill = "scenario") +
         guides(fill=guide_legend(title="")) + 
         facet_grid(. ~ month, scales="free") +

}





