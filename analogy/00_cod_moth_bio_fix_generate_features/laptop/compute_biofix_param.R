

library(data.table)
library(dplyr)
library(foreign)


param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

##############################################################################
loc_elev <- read.table(paste0(param_dir, "elevations.txt")) %>% data.table()
setnames(loc_elev, old=c("V1", "V2"), new=c("location", "elevation"))

##############################################################################
Min_counties <- read.dbf("/Users/hn/Documents/GitHub/Min_DB/VICID_CO.DBF")
setnames(Min_counties, old= colnames(Min_counties), new= tolower(colnames(Min_counties)))
setnames(Min_counties, old=c("vicclat", "vicclon"), new=c("lat", "long"))
Min_counties <- subset(Min_counties, select= c(state_fips, state, lat, long))
Min_counties$location <- paste0(Min_counties$lat, "_", Min_counties$long)


loc_elev <- merge(loc_elev, Min_counties)


# Filter the locations that are not in the ID, OR, WA
loc_elev <- loc_elev %>% 
            filter( !(state %in% (c("ID", "WA", "OR"))))

# compute bio fix
loc_elev_low_high <- mean_emerge_time_bio_low_hi(loc_elev)
write.table(x = loc_elev_low_high, 
            file = paste0(param_dir, "biofix_param_low_hi.csv"), 
            row.names=F, na="", col.names=T, sep=",")


loc_elev_high <- mean_emerge_time_bio_high(loc_elev)
write.table(x = loc_elev_high, 
            file = paste0(param_dir, "biofix_param_hi.csv"), 
            row.names=F, na="", col.names=T, sep=",")



