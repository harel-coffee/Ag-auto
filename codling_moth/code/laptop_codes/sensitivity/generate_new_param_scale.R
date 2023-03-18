
param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
param_name = "CodlingMothparameters.txt"
shift_list = c(.01, .02, .03, .04, .05, .1, .15, .2)
for (shift in shift_list){
  generate_new_param_scale(param_dir, param_name, shift)
}
