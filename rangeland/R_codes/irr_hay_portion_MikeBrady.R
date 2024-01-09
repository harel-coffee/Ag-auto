# 2017

d <- read_delim("2017_cdqt_data.txt.gz", "\t", escape_double = FALSE, col_types = cols(VALUE = col_character()), trim_ws = TRUE)

d <- d[d$AGG_LEVEL_DESC == "COUNTY",]
d <- d[d$STATE_NAME %in% c("ARIZONA","CALIFORNIA","IDAHO", "OREGON","WASHINGTON"),]
d <- d %>% unite(fips, STATE_FIPS_CODE, COUNTY_CODE, sep="", na.rm=FALSE)

 

#### 2017 ######

d <- d[d$SECTOR_DESC == "CROPS",]
d <- d[!is.na(d$VALUE),]

# Field 2017 #
field <- d[d$CENSUS_TABLE == 25,]
field <- field[field$SHORT_DESC %like% "ACRES HARVESTED",]
field <- field %>% filter(str_detect(SHORT_DESC, "IRRIGAT", negate=TRUE))

field <- field %>% filter(str_detect(SHORT_DESC, "SPRING", negate=TRUE)) %>%
                   filter(str_detect(SHORT_DESC, "WINTER", negate=TRUE)) %>%
                   filter(str_detect(SHORT_DESC, "PIMA", negate=TRUE)) %>%
                   filter(str_detect(SHORT_DESC, "UPLAND", negate=TRUE)) %>%
                   filter(str_detect(SHORT_DESC, "OIL TYPE", negate=TRUE)) %>%
                   filter(str_detect(SHORT_DESC, "NON-OIL TYPE", negate=TRUE))

field <- field %>% select(CENSUS_CHAPTER, CENSUS_TABLE, STATE_ALPHA, CENSUS_ROW, fips, SHORT_DESC, VALUE)

 
field_i <- d[d$CENSUS_TABLE == 25,]
field_i <- field_i[field_i$SHORT_DESC %like% "ACRES HARVESTED",]
field_i <- field_i %>% filter(str_detect(SHORT_DESC, "IRRIGAT"))
field_i <- field_i %>% filter(str_detect(SHORT_DESC, "SPRING", negate=TRUE)) %>%
                       filter(str_detect(SHORT_DESC, "WINTER", negate=TRUE)) %>%
                       filter(str_detect(SHORT_DESC, "PIMA", negate=TRUE)) %>%
                       filter(str_detect(SHORT_DESC, "UPLAND", negate=TRUE)) %>%
                       filter(str_detect(SHORT_DESC, "OIL TYPE", negate=TRUE)) %>%
                       filter(str_detect(SHORT_DESC, "NON-OIL TYPE", negate=TRUE))

field_i <- field_i %>% rename(VALUE_I = VALUE)
field_i <- field_i %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE_I)

field_merge <- full_join(field, field_i, by = c("fips", "CENSUS_CHAPTER","CENSUS_TABLE","CENSUS_ROW"))
field_merge2 <- field_merge %>% filter(str_detect(VALUE, "(D)", negate=TRUE))
field_merge2 <- field_merge2 %>% replace_na(list(VALUE_I = 0, y=c(NA)))
field_merge2$VALUE <- as.numeric(gsub(",", "", field_merge2$VALUE))

field_merge2$harv <- field_merge2$VALUE_I # why a new column? why not just rename?
field_merge2$harv <- as.numeric(gsub(",","", field_merge2$harv))
field_merge2$harv <- as.numeric(field_merge2$harv)

field_merge2 <- full_join(field_merge2, fieldMeanImp, by = c("fips","SHORT_DESC.x"))
field_merge2 <- field_merge2 %>% mutate(harv = ifelse(VALUE_I == "(D)", VALUE*meanImp, harv))
fieldSum <- field_merge2 %>% group_by(fips) %>% summarise(fieldSum = sum(harv, na.rm=TRUE)) 

## Grasses and legumes ##

gl <- d[d$CENSUS_TABLE == 26,]

gl_i <- subset(gl, SHORT_DESC %in% c("HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED",
                                     "GRASSES & LEGUMES TOTALS, IRRIGATED, SEED - ACRES HARVESTED",
                                     "CORN, SILAGE, IRRIGATED - ACRES HARVESTED", 
                                     "SORGHUM, SILAGE, IRRIGATED - ACRES HARVESTED"))

gl <- subset(gl, SHORT_DESC %in% c("HAY & HAYLAGE - ACRES HARVESTED",
                                   "GRASSES & LEGUMES TOTALS, SEED - ACRES HARVESTED",
                                   "CORN, SILAGE - ACRES HARVESTED", 
                                   "SORGHUM, SILAGE - ACRES HARVESTED"))

gl_i <- gl_i %>% rename(VALUE_I = VALUE)
gl_i <- gl_i %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE_I)
gl <- gl     %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE, STATE_ALPHA)
gl_merge <- full_join(gl, gl_i, by = c("CENSUS_CHAPTER","CENSUS_TABLE","CENSUS_ROW", "fips"))
gl_merge$VALUE <- as.numeric(gsub(",","", gl_merge$VALUE))
gl_merge$harv <- gl_merge$VALUE_I
gl_merge$harv <- as.numeric(gsub(",","", gl_merge$harv))
gl_merge$harv <- as.numeric(gl_merge$harv)
gl_merge$VALUE_Inum <- as.numeric(gsub(",","", gl_merge$VALUE_I))
gl_merge <- full_join(gl_merge, glMeanImp, by = c("fips","SHORT_DESC.x"))
gl_merge <- gl_merge %>% mutate(harv = ifelse(VALUE_I == "(D)", VALUE*meanImp, harv))
glSum <- gl_merge %>% group_by(fips) %>% summarise(glSum = sum(harv,na.rm=TRUE))

## end - Grasses and legumes ##

# Other
other <- d[d$CENSUS_TABLE == 27,]
other <- other[other$SHORT_DESC %like% "ACRES HARVESTED",]
other_i <- other %>% filter(str_detect(SHORT_DESC, "IRRIGAT"))
other <- other %>% filter(str_detect(SHORT_DESC, "IRRIGAT", negate=TRUE))
other_i <- other_i %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE)
other_i <- other_i %>% rename(VALUE_I = VALUE)
other <- other %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE)
other_merge <- full_join(other, other_i, by = c("fips", "CENSUS_CHAPTER","CENSUS_TABLE","CENSUS_ROW"))
other_merge <- other_merge %>% mutate(harv = ifelse(VALUE_I == "(D)", VALUE, VALUE_I))
other_merge$harv <- as.numeric(gsub(",","", other_merge$harv))
otherSum <- other_merge %>% group_by(fips) %>% summarise(otherSum = sum(harv, na.rm=TRUE))

 

# Veg
veg <- d[d$CENSUS_TABLE == 29,]
veg <- veg[veg$SHORT_DESC == "VEGETABLE TOTALS, IN THE OPEN - ACRES HARVESTED",]
veg <- veg %>% filter(VALUE != "(D)")
veg$harv <- as.numeric(gsub(",","", veg$VALUE))
vegSum <- veg %>% group_by(fips) %>% summarise(vegSum = sum(harv, na.rm=TRUE))

 

# Tree fruit and nuts
tfn <- d[d$CENSUS_TABLE == 31,]

noncitrus <- tfn[tfn$SHORT_DESC == "NON-CITRUS TOTALS, (EXCL BERRIES) - ACRES BEARING & NON-BEARING",]
noncitrus <- noncitrus %>% filter(VALUE != "(D)")
noncitrus$harv <- as.numeric(gsub(",","", noncitrus$VALUE))
noncitrusSum <- noncitrus %>% group_by(fips) %>% summarise(noncitrusSum = sum(harv, na.rm=TRUE))
 
citrus <- tfn[tfn$SHORT_DESC == "CITRUS TOTALS - ACRES BEARING & NON-BEARING",]
citrus <- citrus %>% filter(VALUE != "(D)")
citrus$harv <- as.numeric(gsub(",","", citrus$VALUE))
citrusSum <- citrus %>% group_by(fips) %>% summarise(citrusSum = sum(harv, na.rm=TRUE))
 
nut <- tfn[tfn$SHORT_DESC == "TREE NUT TOTALS - ACRES BEARING & NON-BEARING",]
nut <- nut %>% filter(VALUE != "(D)")
nut$harv <- as.numeric(gsub(",","", nut$VALUE))
nutSum <- nut %>% group_by(fips) %>% summarise(nutSum = sum(harv, na.rm=TRUE))


# Berries
berries <- d[d$CENSUS_TABLE == 32,]
berries_i <- berries[berries$SHORT_DESC == "BERRY TOTALS, IRRIGATED - ACRES GROWN",]
berries_i <- berries_i %>% rename(VALUE_I = VALUE)
berries <- berries[berries$SHORT_DESC == "BERRY TOTALS - ACRES GROWN",]
berries_i <- berries_i %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE_I)
berries <- berries %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE)
berries_merge <- full_join(berries, berries_i, by = c("fips", "CENSUS_CHAPTER","CENSUS_TABLE","CENSUS_ROW"))
berries_merge <- berries_merge %>% mutate(harv = ifelse(VALUE_I == "(D)", VALUE, VALUE_I))
berries_merge <- berries_merge %>% filter(harv != "(D)") %>% filter(harv!= "(Z)")
berries_merge$harv <- as.numeric(gsub(",","", berries_merge$harv))
berriesSum <- berries_merge %>% group_by(fips) %>% summarise(berriesSum = sum(harv, na.rm=TRUE))

# Flor
flor <- d[d$CENSUS_TABLE == 34,]
flor <- flor[flor$SHORT_DESC == "FLORICULTURE TOTALS, IN THE OPEN - ACRES IN PRODUCTION",]
flor <- flor %>% filter(VALUE != "(D)")
flor$harv <- as.numeric(gsub(",","", flor$VALUE))
florSum <- flor %>% group_by(fips) %>% summarise(florSum = sum(harv, na.rm=TRUE))

# Xmas
xmas <- d[d$CENSUS_TABLE == 35,]
xmas <- xmas[xmas$SHORT_DESC == "CUT CHRISTMAS TREES - ACRES IN PRODUCTION",]

# Sum
irrHarv <- merge(fieldSum, glSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, otherSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, vegSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, noncitrusSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, citrusSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, nutSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, berriesSum, by = "fips", all = TRUE)
irrHarv <- merge(irrHarv, florSum, by = "fips", all = TRUE)

 

irrHarv$totHarv <- rowSums(cbind(irrHarv$fieldSum,irrHarv$glSum,irrHarv$otherSum,irrHarv$vegSum,irrHarv$noncitrusSum,
                                 irrHarv$citrusSum,irrHarv$nutSum,irrHarv$berriesSum,irrHarv$florSum),na.rm=TRUE)

irrHarv <- merge(irrHarv, extent, by = "fips", all = TRUE)

irrHarv$CI_17 <- irrHarv$totHarv/irrHarv$irrigExtent
irrHarv17comp <- irrHarv
irrHarvAll <- irrHarv %>% select(fips, CI_17)
