
diapause_map_prep <- function(input_dir, file_name,
                             param_dir, location_group_name="LocationGroups.csv"){
    file_name = paste0(input_dir, file_name, ".rds")
    data <- data.table(readRDS(file_name))

    loc_grp = data.table(read.csv(paste0(param_dir, location_group_name)))
    #loc_grp = loc_grp[1:15,]
    loc_grp$latitude = as.numeric(loc_grp$latitude)
    loc_grp$longitude = as.numeric(loc_grp$longitude)

    data <- data[latitude %in% loc_grp$latitude & longitude %in% loc_grp$longitude]

    theta = 0.2163108 + (2 * atan(0.9671396 * tan(0.00860 * (data$dayofyear - 186))))
    phi = asin(0.39795 * cos(theta))
    D = 24 - ((24/pi) * acos((sin(6 * pi / 180) + 
                        (sin(data$latitude * pi / 180) * sin(phi)))/(cos(data$latitude * pi / 180) * cos(phi))))
    data$daylength = D

    data$diapause = 102.6077 * exp(-exp(-(-1.306483) * (data$daylength - 16.95815)))
    data$diapause1 = data$diapause
    data[diapause1 > 100, diapause1 := 100]
    data$enterDiap = (data$diapause1/100) * data$SumLarva
    data$escapeDiap = data$SumLarva - data$enterDiap

    sub = data
    rm(data)
    startingpopulationfortheyear<-1000
    #generation1
    sub[,LarvaGen1RelFraction := LarvaGen1/sum(LarvaGen1), 
         by =list(year,ClimateScenario, 
         latitude,longitude,ClimateGroup, CountyGroup) ]
    sub$AbsPopLarvaGen1<-sub$LarvaGen1RelFraction*startingpopulationfortheyear
    sub$AbsPopLarvaGen1Diap<-sub$AbsPopLarvaGen1*sub$diapause1/100
    sub$AbsPopLarvaGen1NonDiap<-sub$AbsPopLarvaGen1- sub$AbsPopLarvaGen1Diap

    #generation2
    sub[,LarvaGen2RelFraction := LarvaGen2/sum(LarvaGen2), 
         by = list(year, ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]
    sub[,AbsPopLarvaGen2 := LarvaGen2RelFraction * sum(AbsPopLarvaGen1NonDiap)*3.9, 
         by = list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]
    sub$AbsPopLarvaGen2Diap <- sub$AbsPopLarvaGen2 * sub$diapause1/100
    sub$AbsPopLarvaGen2NonDiap <- sub$AbsPopLarvaGen2 - sub$AbsPopLarvaGen2Diap

    #generation3
    sub[,LarvaGen3RelFraction := LarvaGen3/sum(LarvaGen3), 
         by =list(year,ClimateScenario, latitude, longitude, ClimateGroup, CountyGroup)]

    sub[,AbsPopLarvaGen3 := LarvaGen3RelFraction*sum(AbsPopLarvaGen2NonDiap)*3.9, 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup) ]
    sub$AbsPopLarvaGen3Diap <- sub$AbsPopLarvaGen3*sub$diapause1/100
    sub$AbsPopLarvaGen3NonDiap <- sub$AbsPopLarvaGen3- sub$AbsPopLarvaGen3Diap

    #generation4
    sub[,LarvaGen4RelFraction := LarvaGen4/sum(LarvaGen4), 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

    sub[,AbsPopLarvaGen4 := LarvaGen4RelFraction*sum(AbsPopLarvaGen3NonDiap)*3.9, 
         by =list(year,ClimateScenario, latitude,longitude,ClimateGroup, CountyGroup)]

    sub$AbsPopLarvaGen4Diap <- sub$AbsPopLarvaGen4*sub$diapause1/100
    sub$AbsPopLarvaGen4NonDiap <- sub$AbsPopLarvaGen4- sub$AbsPopLarvaGen4Diap

    ### get totals similar to Sum Larva column, but abs numbers
    sub$AbsPopTotal <- sub$AbsPopLarvaGen1 + sub$AbsPopLarvaGen2 + sub$AbsPopLarvaGen3 + sub$AbsPopLarvaGen4
    sub$AbsPopDiap <- sub$AbsPopLarvaGen1Diap + sub$AbsPopLarvaGen2Diap + sub$AbsPopLarvaGen3Diap + sub$AbsPopLarvaGen4Diap
    sub$AbsPopNonDiap <- sub$AbsPopLarvaGen1NonDiap + sub$AbsPopLarvaGen2NonDiap + 
                        sub$AbsPopLarvaGen3NonDiap + sub$AbsPopLarvaGen4NonDiap

    sub1 = subset(sub, 
                select = c("latitude", "longitude", 
                           "County", "CountyGroup", 
                           "ClimateScenario", "ClimateGroup", 
                           "year", "dayofyear", "CumDDinF", 
                           "SumLarva", "enterDiap", 
                           "escapeDiap", "AbsPopTotal",
                           "AbsPopNonDiap","AbsPopDiap"))
    rm (sub)
    sub2 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), 
                                                  RelNonDiap = mean(escapeDiap), 
                                                  AbsLarvaPop = mean(AbsPopTotal), 
                                                  AbsDiap = mean(AbsPopDiap), 
                                                  AbsNonDiap = mean(AbsPopNonDiap), 
                                                  CumulativeDDF = mean(CumDDinF)), 
                                                  by = c("ClimateGroup", "CountyGroup", 
                                                          "latitude", "longitude", 
                                                          "dayofyear")]

    sub3 = sub1[, .(RelLarvaPop = mean(SumLarva), RelDiap = mean(enterDiap), 
                    RelNonDiap = mean(escapeDiap), AbsLarvaPop = mean(AbsPopTotal), 
                    AbsDiap = mean(AbsPopDiap), AbsNonDiap = mean(AbsPopNonDiap), 
                    CumulativeDDF = mean(CumDDinF)), 
                    by = c("ClimateGroup", "CountyGroup", "dayofyear")]

    return (list(sub1, sub2, sub3))
}