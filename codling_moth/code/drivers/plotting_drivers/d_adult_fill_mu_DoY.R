
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(ggpubr)

source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)

options(digits=9)
options(digit=9)

start_time <- Sys.time()
#########
######### Running this is gonna take at least 80 minutes, 
######### IF it is successful! R memory allocation problem!
#########
# args = commandArgs(trailingOnly=TRUE)
# version = args[1]

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/"
plot_path = "/data/hydro/users/Hossein/codling_moth_new/local/processed/overlaping/plots/"

if (dir.exists(file.path(plot_path)) == F) {
  dir.create(path = plot_path, recursive = T)
}

adult_DoY_mean_45 <- plot_adult_DoY_filling_mean(input_dir=data_dir, 
                                                 file_name ="combined_CMPOP_", 
                                                 version="rcp45", 
                                                 output_dir=plot_path)

adult_DoY_mean_85 <- plot_adult_DoY_filling_mean(input_dir=data_dir, 
                                                 file_name ="combined_CMPOP_", 
                                                 version="rcp85", 
                                                 output_dir=plot_path)


adult_DoY_mean <- ggpubr::ggarrange(plotlist = list(adult_DoY_mean_45, adult_DoY_mean_85),
                                  ncol = 2, nrow = 1,
                                  common.legend = TRUE, legend = "bottom")

ggsave("adult_DoY_mean.png", adult_DoY_mean, width=15, height=7.5, path=plot_path, dpi=350)

print( Sys.time()- start_time)

