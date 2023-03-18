tanh_1<- function(x, scaler=-100, shift=14){
  scaler * tanh(x + shift)
}

ggplot(data.frame(x=c(-40, 10)), aes(x=x)) + 

geom_path(stat="function", fun=tanh_1, aes(colour="grey70"), linetype=1)