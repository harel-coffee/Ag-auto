#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)

input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"

files = list.files(input_dir, "*.rds")
meta_data = data.table(name=character(), n_rows=numeric(), n_cols=numeric())
for (file in files){
	file_name = paste0(input_dir, file)
	data <- data.table(readRDS(file_name))
	dimension = dim(data)
	meta_data = rbind(meta_data, list(file, dimension[1], dimension[2]))
	write.table(data.frame(meta_data), paste0(write_dir, "metadata" , ".txt"), sep="\t")
}