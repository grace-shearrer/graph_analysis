# these are functions used in the HCP analysis
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")
violin <- function(d, x, y){
  plot1<-ggplot(d, aes(factor(x), y)) + 
    geom_violin(aes(fill = x),stat = "ydensity",draw_quantiles = c(0.25, 0.5, 0.75))+scale_fill_manual(values=cbPalette)+theme_classic()
  return(plot1)
}

fit_function<-function(y,data){
  fit<-lm(y~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=data)
  # sumfit<-summary(fit)
  lsq_c1 = lsmeans(fit, "group")
  contrast(lsq_c1, cons, adjust='sidak')
  
}
