####
#### Q3. Crop intensities in each county, and for each (county, crop) pair, 
#### where NASS, LastSuveyDate, and perennials are filtered out in different combinations.
#### We do this for SOS = 0.3 and SG params = [7, 3]
####

#
# The script first answers Q3.
#
rm(list=ls())
library(data.table)
library(dplyr)
source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)


data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                   "05_01_allFields_SeasonCounts/")


param_dir <- paste0("/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/")
double_crop_potens <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))
cropTypes_categorizations <- read.csv(paste0(param_dir, "crop_types.csv"))


cropTypes_categorizations$dblCrop_potentials <- tolower(cropTypes_categorizations$dblCrop_potentials)
ignore_Crops <- cropTypes_categorizations %>% filter(dblCrop_potentials == "ignore") %>% data.table()

Mike_CI_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/MikeBrady/"
Mike_CI <- read.csv(paste0(Mike_CI_dir, "MikeBrady_CroppingIntensityCounty.csv"))
Mike_CI_WA <- Mike_CI %>% filter(STATE_NAME == "WASHINGTON") %>% data.table()
Mike_CI_WA <- within(Mike_CI_WA, remove("fips", "STATE_NAME"))
Mike_CI_WA$COUNTY_NAME <- stringr::str_to_title(Mike_CI_WA$COUNTY_NAME)

output_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/objectives_answer/Q3_MikeStyle/"
if (dir.exists(file.path(output_dir)) == F){dir.create(path=file.path(output_dir), recursive=T)}

################################################################################################################

fields_seasons <- read.csv(paste0(data_dir, "extended_all_fields_seasonCounts_noFilter_SEOS3.csv"))
fields_seasons <- pick_eastern_counties(fields_seasons) # only pick the eastern counties
fields_seasons <- fields_seasons %>%
                filter(SG_params == 73) %>%
                data.table()

fields_seasons$CropTyp <- tolower(fields_seasons$CropTyp)
fields_seasons$DataSrc <- tolower(fields_seasons$DataSrc)

################################################################################
#####
##### 1. Toss out ignore labels (for both numerator and denominator).  

##### 2. Toss irrigation type "None" (for both numerator and denominator).  Exactly  
#####    "none", i.e. keep the combinations of "none" with something else (for both numerator  
#####    and denominator). 

#####  3. USDA data does not include irrigated Pasture. Remove pasture category by CropGrp.  
#####  I.e. keep CropType Pasture whose cropGrp is Other (for both numerator and denominator). 

##### 4. Drop CropGrp "other" except the ones whose crop type includes the word fallow. 

################################################################################ 

# fields_seasons <- filter_out_non_irrigated_datatable(fields_seasons)

##### 1. Toss out ignore labels (for both numerator and denominator).  
fields_seasons <- fields_seasons %>% 
                filter((!CropTyp %in% ignore_Crops$CropTyp)) %>% 
                data.table()

##### 2. Toss irrigation type "None" (for both numerator and denominator).  Exactly  
#####    "none", i.e. keep the combinations of "none" with something else (for both numerator  
#####    and denominator). 
fields_seasons <- fields_seasons %>% 
                filter(Irrigtn != "none") %>% 
                data.table()


#####  3. USDA data does not include irrigated Pasture. Remove pasture category by CropGrp.  
#####  I.e. keep CropType Pasture whose cropGrp is Other (for both numerator and denominator). 

shapeFileDataPart <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"
WSDA_2016 <- read.csv(paste0(shapeFileDataPart, "WSDA_DataTable_2016.csv"))
WSDA_2017 <- read.csv(paste0(shapeFileDataPart, "WSDA_DataTable_2017.csv"))
WSDA_2018 <- read.csv(paste0(shapeFileDataPart, "WSDA_DataTable_2018.csv"))

WSDA_2016 <- subset(WSDA_2016, select=c("ID", "CropGrp"))
WSDA_2017 <- subset(WSDA_2017, select=c("ID", "CropGrp"))
WSDA_2018 <- subset(WSDA_2018, select=c("ID", "CropGrp"))

WSDA <- rbind(WSDA_2016, WSDA_2017, WSDA_2018)
WSDA$ID <- as.character(WSDA$ID)
WSDA$CropTyp <- tolower(WSDA$CropTyp)
WSDA$CropGrp <- tolower(WSDA$CropGrp)
rm(WSDA_2016, WSDA_2017, WSDA_2018)

fields_seasons <- dplyr::left_join(x = fields_seasons, y = WSDA, by = "ID")

fields_seasons <- fields_seasons %>% 
                filter(!(CropTyp == "pasture" & CropGrp == "pasture")) %>% 
                data.table()

#####
##### 4. Drop CropGrp "other" except the ones whose crop type includes the word fallow. 
#####    This is already done (only in our case, with this data set) by the column ignore and irrigation type.
#####    in more details, we dropped all of these crop types when we filtered by "ignore", except for "wheat fallow"
#####    in which case the "wheat fallow" is gone since their irrigation system is "none". at least this is the case for 2016.
#####    I did not check other years individually, but that is what is happening according to following steps. Lets just do it anyway.
#####

# toss the group other, except for those who have fallow in it.

# keep pasture whose group is other (keep this)
fields_seasons_pasture_other <- fields_seasons %>% filter(CropTyp == "pasture") %>% data.table()

# keep this.
fields_seasons_Otherthan_other <- fields_seasons %>% filter(CropGrp != "other") %>% data.table()


fields_seasons_other <- fields_seasons %>% filter(CropGrp == "other") %>% data.table()
fields_seasons_fallow_other <- fields_seasons_other[grepl('fallow', fields_seasons_other$CropTyp), ] # this is of dimension zero


fields_seasons <- rbind(fields_seasons_pasture_other, fields_seasons_fallow_other, fields_seasons_Otherthan_other)
fields_seasons$county <- as.character(fields_seasons$county)

year <- 2016
NASS <- FALSE
lastSurveyDate <- FALSE

for (year in c(2016, 2017, 2018)){
  curr_seasonCount_perField <- fields_seasons %>% filter(SF_year == year) %>% data.table()
  print (unique(curr_seasonCount_perField$SF_year))

  for (NASS in c(FALSE)){
    for (lastSurveyDate in c(FALSE)){
    	#
      # Perennials should be out for numerator and in for denominator.
      #
      # for (perennials in c(TRUE, FALSE)){

      if (NASS == TRUE){
        #  print (dim(curr_seasonCount_perField))
        curr_seasonCount_perField <- curr_seasonCount_perField %>% filter(DataSrc != "nass")
        nassName = "nassOut"
        #  print (dim(curr_seasonCount_perField))
        }else{
         nassName = "nassIn"
      }

      if (lastSurveyDate == TRUE){
        #  print (dim(curr_seasonCount_perField))
        curr_seasonCount_perField <- filter_lastSrvyDate_DataTable(curr_seasonCount_perField, year)
        #  print (dim(curr_seasonCount_perField))
        surveyDateName = "correctSurveyDate"
        }else{
        surveyDateName = "wrongSurveyDate"
      }

      #####
      ##### compute numerators
      #####
      curr_2cropped_fields <- curr_seasonCount_perField %>% filter(season_count >= 2) %>% data.table()
      curr_1cropped_fields <- curr_seasonCount_perField %>% filter(season_count < 2) %>% data.table()
      
      A2_per_county <- curr_2cropped_fields %>% 
                       group_by(county) %>% 
                       summarise(acrSum_A2=sum(ExctAcr)) %>%
                       data.table()

      A2_per_county_corp <- curr_2cropped_fields %>% 
                            group_by(county, CropTyp)%>% 
                            summarise(acrSum_A2=sum(ExctAcr)) %>%
                            data.table()

      A2_per_corp <- curr_2cropped_fields %>% 
                     group_by(CropTyp)%>% 
                     summarise(acrSum_A2=sum(ExctAcr)) %>%
                     data.table()


      A1_per_county <- curr_1cropped_fields %>% 
                       group_by(county) %>% 
                       summarise(acrSum_A1=sum(ExctAcr)) %>%
                       data.table()

      A1_per_county_corp <- curr_1cropped_fields %>% 
                            group_by(county, CropTyp)%>% 
                            summarise(acrSum_A1=sum(ExctAcr)) %>%
                            data.table()

      A1_per_corp <- curr_1cropped_fields %>% 
                     group_by(CropTyp)%>% 
                     summarise(acrSum_A1=sum(ExctAcr)) %>%
                     data.table()

        #####
        ##### Do the outer join so we keep everything
        ##### since, some plants/counties/etc may not be double-cropped and therefore 
        ##### are not present in the A1 or A2.
        #####
        intensity_perCounty <- merge(x = A1_per_county, y = A2_per_county, by = "county", all=TRUE)
        
        intensity_perCountyCrop <- merge(x=A1_per_county_corp, 
                                         y=A2_per_county_corp,
                                         by=c("county", "CropTyp"), all=TRUE)
        
        intensity_perCrop <- merge(x=A1_per_corp, 
                                   y=A2_per_corp,
                                   by=c("CropTyp"), all=TRUE)


        #
        # If a crop is not detected by algorithm as double-cropped
        # there will be NAs in the table. replace them with zero.
        #
        intensity_perCounty[is.na(intensity_perCounty)] <- 0
        intensity_perCountyCrop[is.na(intensity_perCountyCrop)] <- 0
        intensity_perCrop[is.na(intensity_perCrop)] <- 0


        intensity_perCounty$numerator <- intensity_perCounty$acrSum_A1 + (intensity_perCounty$acrSum_A2*2)
        intensity_perCounty$denominat <- intensity_perCounty$acrSum_A1 + (intensity_perCounty$acrSum_A2)

        intensity_perCountyCrop$numerator <- intensity_perCountyCrop$acrSum_A1 + (intensity_perCountyCrop$acrSum_A2*2)
        intensity_perCountyCrop$denominat <- intensity_perCountyCrop$acrSum_A1 + (intensity_perCountyCrop$acrSum_A2)

        intensity_perCrop$numerator <- intensity_perCrop$acrSum_A1 + (intensity_perCrop$acrSum_A2*2)
        intensity_perCrop$denominat <- intensity_perCrop$acrSum_A1 + (intensity_perCrop$acrSum_A2)

        intensity_perCounty$crp_intens <- intensity_perCounty$numerator / intensity_perCounty$denominat
        intensity_perCountyCrop$crp_intens <- intensity_perCountyCrop$numerator / intensity_perCountyCrop$denominat
        intensity_perCrop$crp_intens <- intensity_perCrop$numerator / intensity_perCrop$denominat


        a_county = paste("intens_perCounty", nassName, surveyDateName, year, sep = "_")
        a_countyCrop = paste("intens_perCountyCrop", nassName, surveyDateName, year, sep = "_")
        a_Crop = paste("intens_perCrop", nassName, surveyDateName, year, sep = "_")


        ####
        ####  round the decimals
        ####
        cols <- c("acrSum_A1", "acrSum_A2", "numerator", "denominat", "crp_intens")

        intensity_perCounty <- data.table(intensity_perCounty)
        intensity_perCountyCrop <- data.table(intensity_perCountyCrop)
        intensity_perCrop <- data.table(intensity_perCrop)

        intensity_perCounty[,(cols) := round(.SD,2), .SDcols=cols]
        intensity_perCountyCrop[,(cols) := round(.SD,2), .SDcols=cols]
        intensity_perCrop[,(cols) := round(.SD,2), .SDcols=cols]

        intensity_perCounty <- within(intensity_perCounty, remove("numerator", "denominat"))
        intensity_perCountyCrop <- within(intensity_perCountyCrop, remove("numerator", "denominat"))
        intensity_perCrop <- within(intensity_perCrop, remove("numerator", "denominat"))


        ####
        #### add Mike's CI to ours
        #### 
        intensity_perCounty <- dplyr::left_join(x=intensity_perCounty, 
                                                y=Mike_CI_WA,
                                                by=c("county" = "COUNTY_NAME"))
        intensity_perCounty <- data.table(intensity_perCounty)

        intensity_perCountyCrop <- data.table(intensity_perCountyCrop)
        intensity_perCrop <- data.table(intensity_perCrop)
        
        #####
        #####  Write the outputs on the disk
        #####
        curr_out <- paste0(output_dir, "/percounty/")
        if (dir.exists(file.path(curr_out)) == F){dir.create(path=file.path(curr_out), recursive=T)}

        write.csv(intensity_perCounty, 
                  paste0(curr_out, a_county, "_MikeStyle.csv"), row.names = F)

        
        curr_out <- paste0(output_dir, "/percountyCrop/")
        if (dir.exists(file.path(curr_out)) == F){dir.create(path=file.path(curr_out), recursive=T)}


        write.csv(intensity_perCountyCrop, 
                  paste0(curr_out, a_countyCrop, "_MikeStyle.csv"), row.names = F)


        curr_out <- paste0(output_dir, "/perCrop/")
        if (dir.exists(file.path(curr_out)) == F){dir.create(path=file.path(curr_out), recursive=T)}

        write.csv(intensity_perCrop, 
                  paste0(curr_out, a_Crop, "_MikeStyle.csv"), row.names = F)

        # rm(intensity_perCrop, intensity_perCountyCrop, intensity_perCounty)
        # rm(numerator_per_county, numerator_per_county_corp, numerator_per_corp)
        # rm(curr_denominator_perCounty, curr_denominator_perCountycrop, curr_denominator_perCrop)

     # }
    }
  }
}

