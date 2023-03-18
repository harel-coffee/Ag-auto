
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(ggpubr)

source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)

#########
######### Running this is gonna take at least 40 minutes, 
######### IF it is successful! R memory allocation problem!
#########
# args = commandArgs(trailingOnly=TRUE)
# version = args[1]

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/"
plot_path = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/plots/"

adult_DoY_median_45 <- plot_adult_DoY_filling_median(input_dir=data_dir, 
                                                     file_name ="combined_CMPOP_", 
                                                     version="rcp45", 
                                                     output_dir=plot_path)

adult_DoY_median_85 <- plot_adult_DoY_filling_median(input_dir=data_dir, 
                                                     file_name ="combined_CMPOP_", 
                                                     version="rcp85", 
                                                     output_dir=plot_path)


adult_DoY_median <- ggpubr::ggarrange(plotlist = list(adult_DoY_mean_45, adult_DoY_mean_85),
                                    ncol = 2, nrow = 1,
                                    common.legend = TRUE, legend = "bottom")

ggsave("adult_DoY_median.png", adult_DoY_median, path=plot_path, dpi=400)




