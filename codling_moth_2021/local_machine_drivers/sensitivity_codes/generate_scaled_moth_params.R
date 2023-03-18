generate_new_param_scale <- function(param_dir, param_name, scale_shift_percent){
  ## This function computes the low_gdd and high_gdd
  ## for the new modified scale of Weibull function.
  ## Keep in mind, qweibull is inverse of pweibull. 
  ## pinvweibull is NOT inverse of pweibull !!!
  params = read.table(paste0(param_dir, param_name), header=TRUE, sep=",")
  params_new = params
  params_new$scale = params$scale * (1 + scale_shift_percent)
  for (ii in seq(1:16)){
    x_gddhigh = params$gddhigh[ii]
    y_gddhigh = pweibull(x_gddhigh, shape=params$shape[ii], scale=params$scale[ii])
    params_new$gddhigh[ii] = round(qweibull(y_gddhigh, shape = params_new$shape[ii], scale= params_new$scale[ii]))
    
    x_gddlow = params$gddlow[ii]
    y_gddlow = pweibull(x_gddlow, shape=params$shape[ii], scale=params$scale[ii])
    params_new$gddlow[ii] = round(qweibull(y_gddlow, shape = params_new$shape[ii], scale= params_new$scale[ii]))
  }
  
  sensitivity_param_dir <- paste0(param_dir, "sensitivity_params/" )
  dir.create(file.path(sensitivity_param_dir), recursive = TRUE)

  out_name = paste0(sensitivity_param_dir, "/CodlingMothparameters_", scale_shift_percent, ".txt")
  write.table(params_new, out_name, sep=",", row.names = F, quote = FALSE)
}

param_dir = "/Users/hn/Documents/00_GitHub/Ag/codling_moth_2021/parameters/"
param_name = "CodlingMothparameters.txt"
scale_shift = seq(0, 20, 1)/100
for (scale_shift_percent in scale_shift){
  generate_new_param_scale(param_dir, param_name, scale_shift_percent)
}


