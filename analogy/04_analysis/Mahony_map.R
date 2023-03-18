
## import data

# the digital elevation model (DEM) used to generate the 
# input data. This is used as a template raster for mapping the results. 
dem <- raster("dem8.tif") 

# DEM cells that make up the B matrix. this list 
# includes a coastal buffer that is excluded 
# from the analog pool (A matrix).  
land <- read.csv("land.NAnaec8.csv")[,1] 

# read in if not already in memory:
# NN.sigma <- read.csv("NN.sigma.RCP45.GlobalMean.2085.csv")[,1]  

## map (exported to working directory via the png() and dev.off() calls)
png(filename = paste("NoveltyMap.png",sep="_"), type="cairo", 
  units = "in", width=9, height=8, pointsize=16, res=800)
par(mar=c(0,0,0,0))
xl <- 2550000; yb <- -3000000; xr <- 2850000; yt <- -200000
breakseq <- c(0,2,4)
breakpoints <- c(seq(breakseq[1], breakseq[3], 0.01), 9); length(breakpoints)

ColScheme <- colorRampPalette(c("light gray", "black"))(length(breakpoints)-1)  

# alternate color scheme:
# ColScheme <- colorRampPalette(c("#0000CD", 
#                               blue2red(length(breakpoints)-1), "#CD0000"))(length(breakpoints)-1) 

X <- dem
values(X) <- NA
values(X)[land] <- NN.sigma
plot(X, xaxt="n", yaxt="n", col=ColScheme, breaks=breakpoints, 
      legend=FALSE, legend.mar=0, maxpixels=ncell(X)) 
rect(xl, head(seq(yb,yt,(yt-yb)/length(ColScheme)), -1), 
   xr, tail(seq(yb,yt,(yt-yb)/length(ColScheme)),-1), 
   col=ColScheme, border=NA)

rect(xl, yb, xr, yt, col=NA)
text(rep(xr,3), c(yb,mean(c(yt,yb)),yt-100000), 
   sapply(c(bquote(.(breakseq[1])*sigma), 
   bquote(.(breakseq[2])*sigma), 
   bquote(.(breakseq[3])*sigma)),as.expression), pos=4, cex=1, font=1)  

text(xl-180000,mean(c(yb,yt)), "Dissimilarity of best analog", font=2, cex=1, srt=90) 
rect(xl,  yt+100000,  xr,  yt+300000,  col=ColScheme[length(ColScheme)])
text(xr,  yt+200000,  bquote(">"*.(breakseq[3])*sigma),pos=4,cex=1,font=1)
box(col="white", lwd=1.5)
dev.off()


#####################
####### Code for preparing ClimateNA 
####### files for use as an alternate B matrix for 
####### other time periods, other model projections, other climate variables etc.
#####################

# ClimateNA can be obtained here: 
# http://cfcg.forestry.ubc.ca/projects/climate-data/climatebcwna/#ClimateNA
# use the NAnaec8.csv file provided in the working directory to 
# download projected normals for North America ("NA") in a North American 
# equidistant conic ("naec") projection at 8-km resolution. 

# Specify the ClimateNA file attributes
models <-  c("ACCESS1-0","CanESM2","CCSM4","CESM1-CAM5",
           "CNRM-CM5","CSIRO-Mk3-6-0", "GFDL-CM3","GISS-E2R", 
           "GlobalMean","HadGEM2-ES", "INM-CM4", "IPSL-CM5A-MR", 
           "MIROC-ESM", "MIROC5", "MPI-ESM-LR","MRI-CGCM3")    

grid <- "NAnaec8"
model <- models[9]
RCP <- "RCP45"

# central year of the normal period, e.g., 
# 2085 indicates the 2071-2100 normal period. 
proj.year <- 2085 

# type of variables selected from ClimateNA. 
# Y is annual, S is seasonal, M is monthly.
VarCode <- "S" 

# read in ClimateNA output file
file <- paste(grid,"_", model,"_", RCP,"_", proj.year, VarCode, ".csv", sep="")
grid.proj <- read.csv(file, strip.white = TRUE, na.strings = c("NA","",-9999) )
nonCNA <- which(is.na(grid.ref[,6]))  # dem cells outside climateWNA extent

# select predictor variables
predictors <- names(grid.proj)[grep("Tmin|Tmax|PPT", names(grid.proj))]

# removes NA cells and selects analysis variables. 
X.grid.proj <- grid.proj[-nonCNA, which(names(grid.proj) %in% predictors)]

## log-transform precipitation
# set zero values to one, to facilitate log-transformation
for(i in grep("PPT", names(X.grid.proj))){X.grid.proj[which(X.grid.proj[,i]==0),i] <- 1} 

# log-transform the precipitation variables 
X.grid.proj[, grep("PPT", names(X.grid.proj))] <- log(X.grid.proj[, grep("PPT", names(X.grid.proj))])

# This file can be used as the B matrix 
# to calculate novelty for other 
# time periods and/or other model projections. 

write.csv(X.grid.proj, 
        paste("X.",grid,".proj_", model,"_", RCP,"_", proj.year, ".csv", sep=""), 
        row.names=FALSE)  

