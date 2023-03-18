rm(list=ls())
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(ggplot2)


write_path = "/Users/hn/Documents/GitHub/Kirti/Codling_moth_Code/sensitivity/wider_intervals/"
read_path = "/Users/hn/Documents/GitHub/Kirti/Codling_moth_Code/sensitivity/wider_intervals/"

CodlingMothparameters_5_w <- read.csv(paste0(read_path, "CodlingMothparameters_5.csv"))
CodlingMothparameters_5_w = within(CodlingMothparameters_5_w, remove(X))
write.table(CodlingMothparameters_5_w, paste0(write_path, "CodlingMothparameters_5_w.txt"), 
            row.names = FALSE, col.names = TRUE, sep = ",")

rm(list=ls())
CodlingMothparameters_10_w <- read.csv(paste0(read_path, "CodlingMothparameters_10.csv"))
CodlingMothparameters_10_w = within(CodlingMothparameters_10_w, remove(X))
write.table(CodlingMothparameters_10_w, 
	        paste0(write_path, "CodlingMothparameters_10_w.txt"), 
            row.names = FALSE, col.names = TRUE, sep = ",")


rm(list=ls())
CodlingMothparameters_15_w <- read.csv(paste0(read_path, "wider_intervals/CodlingMothparameters_15.csv"))
CodlingMothparameters_15_w = within(CodlingMothparameters_15_w, remove(X))
write.table(CodlingMothparameters_15_w, paste0(write_path, "CodlingMothparameters_15_w.txt"), 
            row.names = FALSE, col.names = TRUE, sep = ",")

rm(list=ls())
CodlingMothparameters_20_w <- read.csv(paste0(read_path, "wider_intervals/CodlingMothparameters_20.csv"))
CodlingMothparameters_20_w = within(CodlingMothparameters_20_w, remove(X))
write.table(CodlingMothparameters_20_w, paste0(write_path, "CodlingMothparameters_20_w.txt"), 
            row.names = FALSE, col.names = TRUE, sep = ",")


