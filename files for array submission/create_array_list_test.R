library(readr)
count=1
df = data.frame(matrix(data=NA,ncol=2))
for (start_doy in 1:53){
   for (region_part in 1:10){
			df[count,1:2] <-c(start_doy,region_part)
			count = count + 1
				
		}
	}

setwd("/weka/data/project/agaid/fabio_scarpare/F_and_V/drivers/01_countDays_toReachMaturity")
write_delim(df,'input_for_fabio_test.txt',delim = " ")