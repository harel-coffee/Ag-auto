# .libPaths("/data/hydro/R_libs35")
# .libPaths()
library(data.table)
library(dplyr)
source_path = "/weka/data/project/agaid/fabio_scarpare/F_and_V/SidFabio_core.R"
source(source_path)
options(digit=9)
options(digits=9)
args <- commandArgs(trailingOnly = TRUE)
v1 = as.numeric(args[1])
v2 = as.numeric(args[2])
print(v1)
print(v2)

#vegetation_list <- c("tomato","carrot","spinach","bean")
#veg_type <- vegetation_list[1]
veg_type = "spinach"
#print(veg_type)

planting_list <- c(1,8,15,22,29,36,43,50,57,64,71,78,85,92,99,106,113,120,127,
                   134,141,148,155,162,169,176,183,190,197,204,211,218,225,232,239,
                   246,253,260,267,274,281,288,295,302,309,316,323,330,337,344,351,358,365)
start_doy <- planting_list[v1]

print("start_DOY")
print(start_doy)

#model_list <- c("BNUESM","CanESM2","CNRMCM5","CSIROMk360","GFDLESM2G",
#                "GFDLESM2M","HadGEM2CC365","HadGEM2ES365","inmcm4","IPSLCM5ALR",
#                "IPSLCM5AMR","IPSLCM5BLR","MIROC5","MIROCESM","MIROCESMCHEM","observed","bcccsm11")
# model_type <- model_list[1]

model_type = "bcccsm11"
RCP = "rcp85"
#print(model_type)

region_part_list <- c("P1","P2","P3","P4","P5","P6","P7","P8","P9","P10")
region_part <- region_part_list[v2]
print(region_part)
#region_part = "P1"

######################################################################
database <- "/weka/data/project/agaid/fabio_scarpare/F_and_V/drivers/"


if (model_type == "observed") {
data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", veg_type, "/", 
                   model_type, "_ALL", "_Jan25_2023_allUS/")
} else {
data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", veg_type, "/", 
                   model_type, "_ALL", "_Jan25_2023_allUS_",RCP,"/")
}
#data_dir <- paste0(database, "00_cumGDD_separateLocationsModels/", veg_type, "/", 
#                   model_type, "_ALL", "_Jan25_2023_allUS/")

out_database = database
param_dir = paste0(database, "000_parameters/")

veg_params <- data.table(read.csv(paste0(param_dir, "veg_params_Jan25_2023.csv"),  as.is=T))
veg_params=veg_params[veg_params$veg==veg_type, ]

if (region_part=="P1"){
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part1.csv")))
  VIC_grids = filter(VIC_grids, file_name != "data_38.71875_-106.84375")
#  VIC_grids = filter(VIC_grids, St_county %in% c("CA_Fresno ", "CA_Kings "))
#  VIC_grids = filter(VIC_grids, St_county %in% c("CA_Fresno ", "CA_Kings ", "CA_Merced ", "CA_San Joaquin ", "CA_Yolo ")) #tomato
#  VIC_grids = filter(VIC_grids, St_county %in% c("CA_Ventura ", "CA_Monterey ", "CA_Santa Barbara ", 
#                      "CA_San Benito ", "CA_Stanislaus ","CA_Tulare ","CA_Riverside ","CA_Imperial ")) #spinach
} else if (region_part=="P2") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part2.csv"))) 
  #VIC_grids = filter(VIC_grids, file_name == "data_45.90625_-116.28125")
} else if (region_part=="P3") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part3.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_41.59375_-85.09375")
} else if (region_part=="P4") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part4.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_38.03125_-84.21875")
} else if (region_part=="P5") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part5.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_44.34375_-69.09375")
} else if (region_part=="P6") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part6.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_46.15625_-112.90625")
} else if (region_part=="P7") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part7.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_47.71875_-98.28125")
} else if (region_part=="P8") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part8.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_44.84375_-71.46875")
} else if (region_part=="P9") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part9.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_43.90625_-116.96875")
} else if (region_part=="P10") {
  VIC_grids = data.table(read.csv(paste0(param_dir, "VIC_noPasture_CRD_ID_unique_Part10.csv")))
  #VIC_grids = filter(VIC_grids, file_name == "data_46.09375_-116.96875")
  VIC_grids = filter(VIC_grids, file_name != "data_43.46875_-110.65625")
  VIC_grids = filter(VIC_grids, file_name != "data_42.90625_-110.15625")
  VIC_grids = filter(VIC_grids, file_name != "data_42.90625_-110.34375")
  VIC_grids = filter(VIC_grids, file_name != "data_42.59375_-109.46875")
  VIC_grids = filter(VIC_grids, file_name != "data_43.21875_-110.34375")
  VIC_grids = filter(VIC_grids, file_name != "data_42.84375_-109.96875")
  VIC_grids = filter(VIC_grids, file_name != "data_42.84375_-110.03125")
  VIC_grids = filter(VIC_grids, file_name != "data_42.90625_-109.90625")
} else {
  #VIC_grids = data.table(read.csv(paste0(param_dir, "bean_grids_GDD_calibration.csv"))) #518
  VIC_grids = data.table(read.csv(paste0(param_dir, "carrot_grids_GDD_calibration.csv"))) %>% filter(VICID %in% c(113372,113371))       #2172
  #VIC_grids = data.table(read.csv(paste0(param_dir, "spinach_grids_GDD_calibration.csv"))) #1035
  #VIC_grids = data.table(read.csv(paste0(param_dir, "tomato_grids_GDD_calibration.csv"))) #1396
}

VIC_grids$location= paste0(VIC_grids$lat, "_" , VIC_grids$long)

grid_count = dim(VIC_grids)[1]

filename_county_state = data.table(read.csv(paste0(param_dir, "filename_county_state.csv")))

# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

col_names <- c("location","year","days_to_maturity","no_of_extreme_cold",
               "no_of_extreme_heat","cum_solar","cum_CDD","cum_HDD") # no_days_in_opt_interval

# 36 below comes from 1980-2015.
if (model_type=="observed"){
  start_yr=1980
  end_yr=2013
 } else{
  start_yr=2041 #2041
  end_yr=2095   #2070
}

year_count = end_yr-start_yr+1
days_to_maturity_tb <- setNames(data.table(matrix(nrow = grid_count*year_count,ncol = length(col_names))), col_names)
days_to_maturity_tb$location <- rep.int(VIC_grids$file_name, year_count)
setorderv(days_to_maturity_tb,  ("location"))
days_to_maturity_tb$year <- rep.int(c(start_yr:end_yr), grid_count)

counter=1
if (dir.exists(data_dir)){
  for(a_file_name in VIC_grids$file_name){
    # if (counter==1){
    #   print (paste0(data_dir, ", ", a_file_name))
    #   counter+=1
    # }
    # Look into the right directory
    if (file.exists(paste0(data_dir, a_file_name, ".csv"))){
      data_tb <- data.table(read.csv(paste0(data_dir, a_file_name, ".csv")))
      data_tb$row_num <- seq.int(nrow(data_tb))

      # add day of year
      data_tb$doy = 1
      data_tb[, doy := cumsum(doy), by=list(year)]
      #
      # Here is where we have filtered 2041 to 2070: unique(days_to_maturity_tb$year)
      #
      for (a_year in sort(unique(days_to_maturity_tb$year))){
        curr_row_num <- data_tb[data_tb$year==a_year & data_tb$doy==start_doy, ]$row_num
        curr_data <- data_tb[data_tb$row_num>=curr_row_num, ]
        curr_data <- data.table(curr_data)
        curr_data$cum_GDD <- 0
        curr_data[, cum_GDD := cumsum(WEDD)] # , by=list(year)

        curr_data$cum_SRAD <- 0
        curr_data[, cum_SRAD := cumsum(SRAD)] # , by=list(year)
		
		curr_data$cum_CDD <- 0
        curr_data[, cum_CDD := cumsum(CDD_Fabio)] # , by=list(year)
		
		curr_data$cum_HDD <- 0
        curr_data[, cum_HDD := cumsum(HDD_Fabio)] # , by=list(year)
		
	#	curr_data$cum_HDDs <- 0
    #   curr_data[, cum_HDDs := cumsum(HDD_Sid)] # , by=list(year)

        day_of_maturity=curr_data[curr_data$cum_GDD >= veg_params[veg_params$veg==veg_type]$maturity_gdd]
        dayCount = day_of_maturity$row_num[1]-curr_data$row_num[1]

        # Record days_to_maturity
        days_to_maturity_tb$days_to_maturity[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dayCount

        # Subset the table between start day and day of maturity to count
        # the number of days in optimum interval and extreme events
        start_to_maturity_tb = curr_data %>%
                               filter(row_num>=curr_data$row_num[1]) %>%
                               filter(row_num<=day_of_maturity$row_num[1]) %>%
                               data.table()
print(a_file_name)
#print(days_to_maturity_tb)
print(a_year)
#print(start_to_maturity_tb)

        days_to_maturity_tb$cum_solar[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_SRAD
		
		days_to_maturity_tb$cum_CDD[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_CDD
		
		days_to_maturity_tb$cum_HDD[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_HDD
		
		#days_to_maturity_tb$cum_HDDs[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year]=tail(start_to_maturity_tb, 1)$cum_HDDs

        # optimum_table = start_to_maturity_tb %>%
        #                 filter(Tavg>=veg_params$optimum_low) %>%
        #                 filter(Tavg<=veg_params$optimum_hi) %>%
        #                 data.table()

        # days_to_maturity_tb$no_days_in_opt_interval[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(optimum_table)[1]

        extreme_cold_tb = start_to_maturity_tb %>%
                          filter(tmin<=veg_params$cold_stress) %>%
                          data.table()

        days_to_maturity_tb$no_of_extreme_cold[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(extreme_cold_tb)[1]


        extreme_heat_tb = start_to_maturity_tb %>%
                          filter(tmax>=veg_params$heat_stress) %>%
                          data.table()

        days_to_maturity_tb$no_of_extreme_heat[days_to_maturity_tb$location==a_file_name & days_to_maturity_tb$year==a_year] = dim(extreme_heat_tb)[1]
		
		#Merging with another file to get the county name
		#days_to_maturity_tb = merge(days_to_maturity_tb,filename_county_state,by.x="location",by.y="filename",all.x=TRUE,all.y=FALSE)
		
      }
    }
  }
#} #deletar depois
#  current_out = paste0(out_database, "/01_countDays_toReachMaturity/",  
#                                       veg_type, "_NL_", "_Oct21_2022_allUS_", start_yr, "_", end_yr, "/") 
  
  
if (model_type=="observed"){
 current_out = paste0(out_database, "01_countDays_toReachMaturity/",veg_type, 
                      "_NL_", "GDD_paper_results_", start_yr, "_", end_yr, "_", region_part,"/") 
 } else{
current_out = paste0(out_database, "01_countDays_toReachMaturity/",veg_type, "_NL_", 
                     "GDD_paper_results_", start_yr, "_", end_yr, "_", region_part,"_",RCP,"_",model_type,"/") 
}

  if (dir.exists(current_out) == F) {
      dir.create(path = current_out, recursive = T)
  }

  write.csv(days_to_maturity_tb, 
            file = paste0(current_out, model_type, "_start_DoY_", start_doy,"_days2maturity.csv"), 
            row.names=FALSE)
}
# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)
