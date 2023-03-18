#
#     March 17th. Before meeting with Mike brady

#   In response to the following email of Kirti:
#
# Found this email, but it is not split by county. Can you just get a count based on 
# querying  the "notes" column for the words "double" or "dbl" 
# in a case insensitive way in the yr  data by county.  We can check how much of the 32 K comes from Grant county.

# And the following email which I cannot follow. There is no Note in 2017 data
# 
#
# For checking double cropped acres from WSDA data, we need to 
# restrict to data surveyed in 2017, but will just use the notes column. 
# Do you know where you have the summary data on the notes column you created for Mike? 

# The total for the whole state (area of plots coded as double cropped 
# some time in the last X years) was about 30,000 acres. I cannot seem 
# to find that email though and I dont know if you split it up by county. 
# If it was, we need the number for grant county from that.

# The whole point of the first cut, it to a sense of how much 
# we may be missing by just replying on the incomplete WSDA marking.
#
rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
# library(sp) # rgdal appears to load this already
library(foreign)

not_corrected_years <- data.table()
for (yr in c("2015", "2016", "2018")){
    not_year_corrected_dir <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
                                      "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                                      "01_true_shapefiles_separate_years/WSDACrop_", yr, "/")

    WSDA_yr_and_prior <- rgdal::readOGR(not_year_corrected_dir,
                                          layer = paste0("WSDACrop_", yr), 
                                           GDAL1_integer64_policy = TRUE)

    WSDA_yr_and_prior <- WSDA_yr_and_prior@data

    WSDA_yr_and_prior$Notes <- tolower(WSDA_yr_and_prior$Notes)
    ##
    ## Extract doubles
    ##
    WSDA_yr_and_prior_double <- WSDA_yr_and_prior[grepl('double',WSDA_yr_and_prior$Notes), ]
    WSDA_yr_and_prior_dbl <- WSDA_yr_and_prior[grepl('dbl',WSDA_yr_and_prior$Notes), ]

    WSDA_yr_and_prior_dbl_double <- rbind(WSDA_yr_and_prior_double, WSDA_yr_and_prior_dbl)
    WSDA_yr_and_prior_dbl_double <- data.table(WSDA_yr_and_prior_dbl_double)

    WSDA_yr_and_prior_dbl_double_summary <- WSDA_yr_and_prior_dbl_double %>% 
                                              group_by(county) %>% 
                                              summarise(acreage_per_county = sum(ExctAcr))%>% 
                                              data.table()

    WSDA_yr_and_prior_dbl_double_summary$shapeFileYear <- paste0(yr, "_not_corrected")
    not_corrected_years <- rbind(not_corrected_years, WSDA_yr_and_prior_dbl_double_summary)

    # write.csv(WSDA_yr_and_prior_dbl_double_summary, 
    #           file = paste0("/Users/hn/Desktop/acreage_summary/", 
    #                         "not_corrected_by_years/WSDA_", yr, "_and_prior_dbl_double_summary.csv"))

    ######
    ######   Does the same thing as above line:
    ######
    # WSDA_yr_and_prior_dbl_double[, acreage_sum := sum(ExctAcr), by=list(county)]
    # WSDA_yr_and_prior_dbl_double <- subset(WSDA_yr_and_prior_dbl_double, select=c("county", "acreage_sum"))
    # WSDA_yr_and_prior_dbl_double <- unique(WSDA_yr_and_prior_dbl_double)
}
write.csv(not_corrected_years,
          file = paste0("/Users/hn/Desktop/acreage_summary/", 
                         "/WSDA_yr_and_prior_dbl_double_summary.csv"),
          row.names = FALSE
          )

##
##  Corrected Years
##
rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
# library(sp) # rgdal appears to load this already
library(foreign)

corrected_years <- data.table()

for (yr in c("2015", "2016", "2018")){
    year_corrected_dir <- paste0("/Users/hn/Documents/01_research_data/Ag_check_point/", 
                                 "remote_sensing/00_shapeFiles/02_correct_years/", 
                                 "03_correct_years_separate/lat_long_projections/WSDACrop_", yr, "/")

    WSDA_only_yr <- rgdal::readOGR(year_corrected_dir,
                                     layer = paste0("WSDACrop_", yr), 
                                     GDAL1_integer64_policy = TRUE)

    WSDA_only_yr <- WSDA_only_yr@data
    print (yr)
    print(dim(WSDA_only_yr))

    WSDA_only_yr$Notes <- tolower(WSDA_only_yr$Notes)
    ##
    ## Extract doubles
    ##
    WSDA_only_yr_double <- WSDA_only_yr[grepl('double',WSDA_only_yr$Notes), ]
    WSDA_only_yr_dbl <- WSDA_only_yr[grepl('dbl',WSDA_only_yr$Notes), ]

    WSDA_only_yr_dbl_double <- rbind(WSDA_only_yr_double, WSDA_only_yr_dbl)
    WSDA_only_yr_dbl_double <- data.table(WSDA_only_yr_dbl_double)

    WSDA_only_yr_dbl_double_summary <- WSDA_only_yr_dbl_double %>% 
                                              group_by(county) %>% 
                                              summarise(acreage_per_county = sum(ExctAcr))%>% 
                                              data.table()

    WSDA_only_yr_dbl_double_summary$shapeFileYear <- paste0(yr, "_corrected")
    corrected_years <- rbind(corrected_years, WSDA_only_yr_dbl_double_summary)

    # write.csv(WSDA_only_yr_dbl_double_summary, 
    #           file = paste0("/Users/hn/Desktop/acreage_summary/", 
    #                         "corrected_by_years/WSDA_only_", yr, "_dbl_double_summary.csv"))

}

write.csv(corrected_years,
          file = paste0("/Users/hn/Desktop/acreage_summary/", 
                        "/WSDA_only_yr_dbl_double_summary.csv"),
          row.names = FALSE
          )

