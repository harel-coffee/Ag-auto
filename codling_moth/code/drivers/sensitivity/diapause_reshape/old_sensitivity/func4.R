diapause <- function(x){
  102.6077 * exp(-exp(1.306483 * (x - 16.95815)))
}

diapause_1 <- function(x){
  103 * exp(-exp(.8 * (x - 15.95815 )))
}

diapause_2 <- function(x){
  104 * exp(-exp(.6 * (x - 14.95815 )))
}

diapause_3 <- function(x){
  104 * exp(-exp(.5 * (x - 13.95815 )))
}

ggplot(data = data.frame(x = 0), mapping = aes(x = x)) + 
xlim(5, 20) + 
stat_function(fun = diapause) +
stat_function(fun = diapause_1, color="red") + 
stat_function(fun = diapause_2, color="blue") +
stat_function(fun = diapause_3, color="green")
