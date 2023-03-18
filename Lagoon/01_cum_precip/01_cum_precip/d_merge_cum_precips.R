###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                      Define all paths                            ##
##                                                                  ##
######################################################################
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

data_dir <- "/data/hydro/users/Hossein/lagoon/01_storm_cumPrecip/cum_precip/"
annual_in <- paste0(data_dir, "annual/")
chunky_in <- paste0(data_dir, "chunky/")
monthly_in <- paste0(data_dir, "monthly/")
wtr_yr_in <- paste0(data_dir, "wtr_yr/")

out_dir <- data_dir
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}

######################################################################
##                                                                  ##
##                                                                  ##
##                                                                  ##
######################################################################
######################################################################
# monthly
#
month_cum_precip_last_day_modeled_hist <- readRDS(paste0(monthly_in, "month_cum_precip_last_day_modeled_hist.rds"))
month_cum_precip_last_day_observed <- readRDS(paste0(monthly_in, "month_cum_precip_last_day_observed.rds"))
month_cum_precip_last_day_RCP45 <- readRDS(paste0(monthly_in, "month_cum_precip_last_day_RCP45.rds"))
month_cum_precip_last_day_RCP85 <- readRDS(paste0(monthly_in, "month_cum_precip_last_day_RCP85.rds"))

month_cum_precip_last_day_modeled_hist_45 <- month_cum_precip_last_day_modeled_hist
month_cum_precip_last_day_modeled_hist_85 <- month_cum_precip_last_day_modeled_hist
month_cum_precip_last_day_modeled_hist_45$emission <- "RCP 4.5"
month_cum_precip_last_day_modeled_hist_85$emission <- "RCP 8.5"

month_cum_precip_last_day_observed_45 <- month_cum_precip_last_day_observed
month_cum_precip_last_day_observed_85 <- month_cum_precip_last_day_observed
month_cum_precip_last_day_observed_45$emission <- "RCP 4.5"
month_cum_precip_last_day_observed_85$emission <- "RCP 8.5"

month_cum_precip_last_day <- rbind(month_cum_precip_last_day_RCP45, 
                                   month_cum_precip_last_day_RCP85,
                                   month_cum_precip_last_day_modeled_hist_45,
                                   month_cum_precip_last_day_modeled_hist_85,
                                   month_cum_precip_last_day_observed_45,
                                   month_cum_precip_last_day_observed_85)
rm(month_cum_precip_last_day_RCP45, 
   month_cum_precip_last_day_RCP85,
   month_cum_precip_last_day_modeled_hist_45,
   month_cum_precip_last_day_modeled_hist_85,
   month_cum_precip_last_day_observed_45,
   month_cum_precip_last_day_observed_85)

month_cum_precip_RCP45 <- readRDS(paste0(monthly_in, "month_cum_precip_RCP45.rds"))
month_cum_precip_RCP85 <- readRDS(paste0(monthly_in, "month_cum_precip_RCP85.rds"))
month_cum_precip_modeled_hist <- readRDS(paste0(monthly_in, "month_cum_precip_modeled_hist.rds"))
month_cum_precip_observed <- readRDS(paste0(monthly_in, "month_cum_precip_observed.rds"))

month_cum_precip_modeled_hist_45 <- month_cum_precip_modeled_hist
month_cum_precip_modeled_hist_85 <- month_cum_precip_modeled_hist
month_cum_precip_modeled_hist_45$emission <- "RCP 4.5"
month_cum_precip_modeled_hist_85$emission <- "RCP 8.5"

month_cum_precip_observed_45 <- month_cum_precip_observed
month_cum_precip_observed_85 <- month_cum_precip_observed
month_cum_precip_observed_45$emission <- "RCP 4.5"
month_cum_precip_observed_85$emission <- "RCP 8.5"

monthly_cum_precip <- rbind(month_cum_precip_RCP45, 
                            month_cum_precip_RCP85,
                            month_cum_precip_modeled_hist_45,
                            month_cum_precip_modeled_hist_85,
                            month_cum_precip_observed_45,
                            month_cum_precip_observed_85)
rm(month_cum_precip_RCP45, 
   month_cum_precip_RCP85,
   month_cum_precip_modeled_hist_45,
   month_cum_precip_modeled_hist_85,
   month_cum_precip_observed_45,
   month_cum_precip_observed_85)

saveRDS(month_cum_precip_last_day, paste0(out_dir, "monthly_cum_precip_last_day.rds"))
saveRDS(monthly_cum_precip, paste0(out_dir, "monthly_cum_precip.rds"))

print ("Monthly is done")
#############################################
# chunky
#
Sept_March_cum_precip_last_day_modeled_hist <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_last_day_modeled_hist.rds"))
Sept_March_cum_precip_last_day_observed <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_last_day_observed.rds"))
Sept_March_cum_precip_last_day_RCP45 <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_last_day_RCP45.rds"))
Sept_March_cum_precip_last_day_RCP85 <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_last_day_RCP85.rds"))

Sept_March_cum_precip_last_day_modeled_hist_45 <- Sept_March_cum_precip_last_day_modeled_hist
Sept_March_cum_precip_last_day_modeled_hist_85 <- Sept_March_cum_precip_last_day_modeled_hist
Sept_March_cum_precip_last_day_modeled_hist_45$emission <- "RCP 4.5"
Sept_March_cum_precip_last_day_modeled_hist_85$emission <- "RCP 8.5"

Sept_March_cum_precip_last_day_observed_45 <- Sept_March_cum_precip_last_day_observed
Sept_March_cum_precip_last_day_observed_85 <- Sept_March_cum_precip_last_day_observed
Sept_March_cum_precip_last_day_observed_45$emission <- "RCP 4.5"
Sept_March_cum_precip_last_day_observed_85$emission <- "RCP 8.5"

Sept_March_cum_precip_last_day <- rbind(Sept_March_cum_precip_last_day_RCP45, 
                                        Sept_March_cum_precip_last_day_RCP85,
                                        Sept_March_cum_precip_last_day_modeled_hist_45,
                                        Sept_March_cum_precip_last_day_modeled_hist_85,
                                        Sept_March_cum_precip_last_day_observed_45,
                                        Sept_March_cum_precip_last_day_observed_85
                                        )

Sept_March_cum_precip_modeled_hist <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_modeled_hist.rds"))
Sept_March_cum_precip_observed <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_observed.rds"))
Sept_March_cum_precip_RCP45 <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_RCP45.rds"))
Sept_March_cum_precip_RCP85 <- readRDS(paste0(chunky_in, "Sept_March_cum_precip_RCP85.rds"))

Sept_March_cum_precip_modeled_hist_45 <- Sept_March_cum_precip_modeled_hist
Sept_March_cum_precip_modeled_hist_85 <- Sept_March_cum_precip_modeled_hist
Sept_March_cum_precip_modeled_hist_45$emission <- "RCP 4.5"
Sept_March_cum_precip_modeled_hist_85$emission <- "RCP 8.5"

Sept_March_cum_precip_observed_45 <- Sept_March_cum_precip_observed
Sept_March_cum_precip_observed_85 <- Sept_March_cum_precip_observed
Sept_March_cum_precip_observed_45$emission <- "RCP 4.5"
Sept_March_cum_precip_observed_85$emission <- "RCP 8.5"

Sept_March_cum_precip <- rbind(Sept_March_cum_precip_RCP45, 
                               Sept_March_cum_precip_RCP85,
                               Sept_March_cum_precip_modeled_hist_45,
                               Sept_March_cum_precip_modeled_hist_85,
                               Sept_March_cum_precip_observed_45,
                               Sept_March_cum_precip_observed_85
                               )

saveRDS(Sept_March_cum_precip_last_day, paste0(out_dir, "Sept_March_cum_precip_last_day.rds"))
saveRDS(Sept_March_cum_precip, paste0(out_dir, "Sept_March_cum_precip.rds"))


rm(Sept_March_cum_precip_RCP45, 
   Sept_March_cum_precip_RCP85,
   Sept_March_cum_precip_modeled_hist,
   Sept_March_cum_precip_modeled_hist_45,
   Sept_March_cum_precip_modeled_hist_85,
   Sept_March_cum_precip_observed,
   Sept_March_cum_precip_observed_45,
   Sept_March_cum_precip_observed_85,
   Sept_March_cum_precip_last_day_RCP45, 
   Sept_March_cum_precip_last_day_RCP85,
   Sept_March_cum_precip_last_day_modeled_hist,
   Sept_March_cum_precip_last_day_modeled_hist_45,
   Sept_March_cum_precip_last_day_modeled_hist_85,
   Sept_March_cum_precip_last_day_observed,
   Sept_March_cum_precip_last_day_observed_45,
   Sept_March_cum_precip_last_day_observed_85
   )
print ("Chunky is done")
#############################################
# annual
#
print ("line 174")
print (annual_in)
ann_cum_precip_last_day_modeled_hist <- readRDS(paste0(annual_in, "ann_cum_precip_last_day_modeled_hist.rds"))
ann_cum_precip_last_day_observed <- readRDS(paste0(annual_in, "ann_cum_precip_last_day_observed.rds"))
ann_cum_precip_last_day_RCP45 <- readRDS(paste0(annual_in, "ann_cum_precip_last_day_RCP45.rds"))
ann_cum_precip_last_day_RCP85 <- readRDS(paste0(annual_in, "ann_cum_precip_last_day_RCP85.rds"))

print (dim(ann_cum_precip_last_day_modeled_hist))
print (dim(ann_cum_precip_last_day_observed))
print (dim(ann_cum_precip_last_day_RCP45))
print (dim(ann_cum_precip_last_day_RCP85))

ann_cum_precip_last_day_modeled_hist_45 <- ann_cum_precip_last_day_modeled_hist
ann_cum_precip_last_day_modeled_hist_85 <- ann_cum_precip_last_day_modeled_hist

ann_cum_precip_last_day_modeled_hist_45$emission = "RCP 4.5"
ann_cum_precip_last_day_modeled_hist_85$emission = "RCP 8.5"

ann_cum_precip_last_day_observed_45 <- ann_cum_precip_last_day_observed
ann_cum_precip_last_day_observed_85 <- ann_cum_precip_last_day_observed
ann_cum_precip_last_day_observed_45$emission <- "RCP 4.5"
ann_cum_precip_last_day_observed_85$emission <- "RCP 8.5"

annual_cum_precip_last_day <- rbind(ann_cum_precip_last_day_RCP45, 
                                    ann_cum_precip_last_day_RCP85,
                                    ann_cum_precip_last_day_modeled_hist_45, 
                                    ann_cum_precip_last_day_modeled_hist_85,
                                    ann_cum_precip_last_day_observed_45, 
                                    ann_cum_precip_last_day_observed_85)
saveRDS(annual_cum_precip_last_day, paste0(out_dir, "annual_cum_precip_last_day.rds"))

print ("line 204 begining of annual, not last day")
ann_cum_precip_modeled_hist<-readRDS(paste0(annual_in, "ann_cum_precip_modeled_hist.rds"))
print ("line 207")
ann_cum_precip_observed<-readRDS(paste0(annual_in, "ann_cum_precip_observed.rds"))
print ("line 209")
ann_cum_precip_RCP45 <- readRDS(paste0(annual_in, "ann_cum_precip_RCP45.rds"))
print ("line 211")
ann_cum_precip_RCP85 <- readRDS(paste0(annual_in, "ann_cum_precip_RCP85.rds"))
print ("line 213")

ann_cum_precip_modeled_hist_45 <- ann_cum_precip_modeled_hist
ann_cum_precip_modeled_hist_85 <- ann_cum_precip_modeled_hist
ann_cum_precip_modeled_hist_45$emission <- "RCP 4.5"
ann_cum_precip_modeled_hist_85$emission <- "RCP 8.5"
ann_cum_precip_modeled_hist <- rbind(ann_cum_precip_modeled_hist_45, 
                                    ann_cum_precip_modeled_hist_85)
rm(ann_cum_precip_modeled_hist_45, 
   ann_cum_precip_modeled_hist_85)

print ("line 224")
ann_cum_precip_observed_45 <- ann_cum_precip_observed
ann_cum_precip_observed_85 <- ann_cum_precip_observed
ann_cum_precip_observed_45$emission <- "RCP 4.5"
ann_cum_precip_observed_85$emission <- "RCP 8.5"
ann_cum_precip_observed <- rbind(ann_cum_precip_observed_45,
                                 ann_cum_precip_observed_85)
rm(ann_cum_precip_observed_45,
   ann_cum_precip_observed_85)
print ("line 233")
ann_cum_precip <- rbind(ann_cum_precip_RCP45, 
                        ann_cum_precip_RCP45,
                        ann_cum_precip_modeled_hist,
                        ann_cum_precip_observed)

print ("line 228")
print (out_dir)
saveRDS(ann_cum_precip, paste0(out_dir, "ann_cum_precip.rds"))

rm(ann_cum_precip_RCP45, 
   ann_cum_precip_RCP45,
   ann_cum_precip_modeled_hist_45,
   ann_cum_precip_modeled_hist_85,
   ann_cum_precip_modeled_hist,
   ann_cum_precip_observed,
   ann_cum_precip_observed_45,
   ann_cum_precip_observed_85,
   ann_cum_precip_last_day_RCP45, 
   ann_cum_precip_last_day_RCP85,
   ann_cum_precip_last_day_modeled_hist,
   ann_cum_precip_last_day_modeled_hist_45, 
   ann_cum_precip_last_day_modeled_hist_85,
   ann_cum_precip_last_day_observed, 
   ann_cum_precip_last_day_observed_45, 
   ann_cum_precip_last_day_observed_85)

print ("Annual is done line 260")

#############################################
#
# wtr_yr
#
print ("line 265")
wtr_yr_sept_cum_precip_last_day_modeled_hist <- readRDS(paste0(wtr_yr_in, "wtr_yr_cum_precip_last_day_modeled_hist.rds"))
print ("line 267")
wtr_yr_sept_cum_precip_last_day_observed <- readRDS(paste0(wtr_yr_in, "wtr_yr_cum_precip_last_day_observed.rds"))
print ("line 269")
wtr_yr_sept_cum_precip_last_day_RCP45 <- readRDS(paste0(wtr_yr_in, "wtr_yr_cum_precip_last_day_RCP45.rds"))
print ("line 271")
wtr_yr_sept_cum_precip_last_day_RCP85 <- readRDS(paste0(wtr_yr_in, "wtr_yr_cum_precip_last_day_RCP85.rds"))
print ("line 273")

wtr_yr_sept_cum_precip_last_day_modeled_hist_45 <- wtr_yr_sept_cum_precip_last_day_modeled_hist
wtr_yr_sept_cum_precip_last_day_modeled_hist_85 <- wtr_yr_sept_cum_precip_last_day_modeled_hist
wtr_yr_sept_cum_precip_last_day_modeled_hist_45$emission <- "RCP 4.5"
wtr_yr_sept_cum_precip_last_day_modeled_hist_85$emission <- "RCP 8.5"
print ("line 279")

wtr_yr_sept_cum_precip_last_day_observed_45 <- wtr_yr_sept_cum_precip_last_day_observed
wtr_yr_sept_cum_precip_last_day_observed_85 <- wtr_yr_sept_cum_precip_last_day_observed
wtr_yr_sept_cum_precip_last_day_observed_45$emission <- "RCP 4.5"
wtr_yr_sept_cum_precip_last_day_observed_85$emission <- "RCP 8.5"

wtr_yr_sept_cum_precip_last_day <- rbind(wtr_yr_sept_cum_precip_last_day_RCP45, 
                                         wtr_yr_sept_cum_precip_last_day_RCP85,
                                         wtr_yr_sept_cum_precip_last_day_modeled_hist_45,
                                         wtr_yr_sept_cum_precip_last_day_modeled_hist_85,
                                         wtr_yr_sept_cum_precip_last_day_observed_45,
                                         wtr_yr_sept_cum_precip_last_day_observed_85)
rm(wtr_yr_sept_cum_precip_last_day_RCP45, 
   wtr_yr_sept_cum_precip_last_day_RCP85,
   wtr_yr_sept_cum_precip_last_day_modeled_hist_45,
   wtr_yr_sept_cum_precip_last_day_modeled_hist_85,
   wtr_yr_sept_cum_precip_last_day_observed_45,
   wtr_yr_sept_cum_precip_last_day_observed_85)
saveRDS(wtr_yr_sept_cum_precip_last_day, paste0(out_dir, "wtr_yr_sept_cum_precip_last_day.rds"))

wtr_yr_sept_cum_precip_observed <- readRDS(paste0(wtr_yr_in, "wtr_yr_sept_cum_precip_observed.rds"))
wtr_yr_sept_cum_precip_modeled_hist <- readRDS(paste0(wtr_yr_in, "wtr_yr_sept_cum_precip_modeled_hist.rds"))
wtr_yr_sept_cum_precip_RCP45 <- readRDS(paste0(wtr_yr_in, "wtr_yr_sept_cum_precip_RCP45.rds"))
wtr_yr_sept_cum_precip_RCP85 <- readRDS(paste0(wtr_yr_in, "wtr_yr_sept_cum_precip_RCP85.rds"))

wtr_yr_sept_cum_precip_observed_45 <- wtr_yr_sept_cum_precip_observed
wtr_yr_sept_cum_precip_observed_85 <- wtr_yr_sept_cum_precip_observed
wtr_yr_sept_cum_precip_observed_45$emission <- "RCP 4.5"
wtr_yr_sept_cum_precip_observed_85$emission <- "RCP 8.5"

wtr_yr_sept_cum_precip_modeled_hist_45 <- wtr_yr_sept_cum_precip_modeled_hist
wtr_yr_sept_cum_precip_modeled_hist_85 <- wtr_yr_sept_cum_precip_modeled_hist
wtr_yr_sept_cum_precip_modeled_hist_45$emission <- "RCP 4.5"
wtr_yr_sept_cum_precip_modeled_hist_85$emission <- "RCP 8.5"

wtr_yr_sept_cum_precip <- rbind(wtr_yr_sept_cum_precip_last_day_RCP45, 
                                wtr_yr_sept_cum_precip_last_day_RCP85,
                                wtr_yr_sept_cum_precip_modeled_hist_45,
                                wtr_yr_sept_cum_precip_modeled_hist_85,
                                wtr_yr_sept_cum_precip_observed_45,
                                wtr_yr_sept_cum_precip_observed_85)

rm(wtr_yr_sept_cum_precip_last_day_RCP45, 
   wtr_yr_sept_cum_precip_last_day_RCP85,
   wtr_yr_sept_cum_precip_modeled_hist_45,
   wtr_yr_sept_cum_precip_modeled_hist_85,
   wtr_yr_sept_cum_precip_observed_45,
   wtr_yr_sept_cum_precip_observed_85)

saveRDS(wtr_yr_sept_cum_precip, paste0(out_dir, "wtr_yr_sept_cum_precip.rds"))
print ("water year is done")

##################################################
end_time <- Sys.time()
print( end_time - start_time)



