diapause <- function(x){
  102.6077 * exp(-exp(1.306483 * (x - 16.95815)))
}

sigmoid <- function(x, scaler=-102, h_shift=10, v_shift = 102){
  term = exp(x - h_shift)
  scaler * (term / (1+term)) + v_shift
}

sigmoid_1 <- function(x, scaler=-110, h_shift=12, v_shift = 200){
  term = exp(x - h_shift)
  scaler * (term / (1+term)) + v_shift
}

ggplot(data.frame(x=c(-10, 35)), aes(x=x)) + 
stat_function(fun = diapause, color="black") +
stat_function(fun = sigmoid, color="red") +
stat_function(fun = sigmoid_1, color="blue", linetype=1)
