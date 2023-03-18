############      packages      #############

library(data.table)
library(dplyr)


############      Directories      #############

data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/pre_microsoft_meeting/")

fip_dir <- "/Users/hn/Documents/01_research_data/large_4_GitHub/Min_DB/"

##############################################
############                     #############
############      read data      #############

table <- read.csv(paste0(data_dir, "table_hossein.csv"), 
                  as.is=TRUE) %>%
         data.table()

good_columns <- c("OBJECTID", "CropType", "LastSurvey", 
                  "County", "Notes")

table <- subset(table, select = good_columns)

table$LastSurvey_year <- as.character(table$LastSurvey_year)
table$LastSurvey_year <- tstrsplit(table$LastSurvey, 
                                   split="/",
                                   fixed=TRUE)[3]

table$LastSurvey_year <- as.numeric(table$LastSurvey_year)

setcolorder(table, c("OBJECTID", "CropType", 
                     "LastSurvey", "LastSurvey_year",
                     "County", "Notes"))

##############################################
############                     #############
############  filter 2018 data   #############

table <- table %>% 
         filter(LastSurvey_year >= 18)%>% 
         data.table()

##############################################
############                     #############
############ filter double crop  #############

double_cr_table <- table[grep("double", 
                              table$Notes, perl=TRUE), ]

##############################################
############                     #############
############ filter some crop    #############

some_crops <- table %>% 
              filter(CropType %in% c("Alfalfa Hay", "Alfalfa Seed",
                                     "Alfalfa/Grass Hay", 
                                     "Apple", "Blueberry", 
                                     "Cherry", 
                                     "Grape, Juice", "Grape, Table",
                                     "Grape, Wine", "Potato"))

##############################################
############                     #############
############   read Min data    #############

library(foreign)
Min <- read.dbf(paste0(fip_dir, "Min_DB/Min_VICID_CO.DBF"), as.is=T)
setnames(Min, old= colnames(Min), new= tolower(colnames(Min)))
Min <- within(Min, remove(vicid, state, vicclat, vicclon, vic_km2))
Min$county <- gsub(" County", "", Min$county)
setnames(Min, old=c("county"), new=c("County"))


double_cr_table <- merge(double_cr_table, Min, all.x=TRUE, by="County")
some_crops <- merge(some_crops, Min, all.x=TRUE, by="County")

##############################################
############                     #############
############      write data     #############

write.table(double_cr_table, 
            paste0(data_dir, "double_crop_2018.csv"),
            row.names=FALSE, col.names=TRUE, 
            sep=",")

write.table(some_crops, 
            paste0(data_dir, "some_crops_2018.csv"),
            row.names=FALSE, col.names=TRUE, 
            sep=",")

some_counties <- subset(double_cr_table, 
                        select=c("County", "state_fips", "fips"))
some_counties <- unique(some_counties)

write.table(some_counties, 
            paste0(data_dir, "counties_to_look_at.csv"),
            row.names=FALSE, col.names=TRUE, 
            sep=",")




