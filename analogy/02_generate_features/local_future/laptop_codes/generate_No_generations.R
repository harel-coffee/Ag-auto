rm(list=ls())
library(dplyr)
library(data.table)

in_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/"
out_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/my_aeolus_2015/all_local/analog/"
data = data.table(readRDS(paste0(in_dir, "generations_Aug_combined_CMPOP_rcp45.rds")))

stages = c("Larva")   # , "Adult"
dead_lines = c("Aug") # , "Nov"
versions = c("rcp45", "rcp85")

file_pref = "generations_" 
file_mid = "_combined_CMPOP_"
file_end = ".rds"

for (dead_line in dead_lines){
    for (version in versions){
        file_name = paste0(file_pref, dead_line, file_mid, version, file_end)
        for (stage in stages){
          assign(x = paste0(stage, "_", file_pref, dead_line, "_", version),
                 value ={generate_no_generations(in_dir, file_name, stage,
                                                 dead_line = dead_line,
                                                 version=version)}
                 )
        }
    }
}

# adult_45 <- merge(Adult_generations_Aug_rcp45, 
#                   Adult_generations_Nov_rcp45, 
#                   by=c("year", "location", "ClimateScenario"))
# rm(Adult_generations_Aug_rcp45, Adult_generations_Nov_rcp45)

larva_45 <- merge(Larva_generations_Aug_rcp45, 
                  Larva_generations_Nov_rcp45,
                  by=c("year", "location", "ClimateScenario"))

rm(Larva_generations_Aug_rcp45, Larva_generations_Nov_rcp45)

# generations_45 <- merge(adult_45, larva_45, by=c("year", "location", "ClimateScenario"))
generations_45 <- larva_45
saveRDS(generations_45, paste0(out_dir, "generations", "_rcp45.rds"))
rm(generations_45)

############################################################
############################################################

# adult_85 <- merge(Adult_generations_Aug_rcp85, 
#                   Adult_generations_Nov_rcp85, 
#                   by=c("year", "location", "ClimateScenario"))
# rm(Adult_generations_Aug_rcp85, Adult_generations_Nov_rcp85)

larva_85 <- merge(Larva_generations_Aug_rcp85, 
                  Larva_generations_Nov_rcp85,
                  by=c("year", "location", "ClimateScenario"))
rm(Larva_generations_Aug_rcp85, Larva_generations_Nov_rcp85)

# generations_85 <- merge(adult_85, larva_85, by=c("year", "location", "ClimateScenario"))
generations_85 <- larva_85
rm(adult_85, larva_85)
saveRDS(generations_85, paste0(out_dir, "generations", "_rcp85.rds"))


