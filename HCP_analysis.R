library(tidyverse)
library(magrittr)
library(purrr)
library(plyr)
library(ggplot2)
library(data.table)
library(dplyr)
# library(formattable)
library(gplots)
library(lm.beta)
library(mgcv)
library(lsmeans)
library(psych)
library(multcomp)
setwd("~/Google Drive/HCP/HCP_graph/1200/datasets/tmp")
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

# Reading in all the data from python 
tbl <- 
  list.files(pattern = "*_per_sub.csv") %>% 
  map_df(~read_csv(.))

dim(tbl)

fc_tbl <- 
  list.files(pattern = "*FC.csv") %>% 
  map_df(~read_csv(.))

dim(fc_tbl)

tbl<-join(tbl,fc_tbl)

demos<-read.table("~/Google Drive/HCP/HCP_graph/1200/datasets/demos.csv", header=T, sep=",")
head(demos)
demos$Age_c<-scale(demos$Age_in_Yrs, scale = FALSE)
demos$HBA1c_c<-scale(demos$HbA1C, scale = FALSE)
demos$group <- ordered(demos$group, levels = c("no", "ov", "ob"))

demos$race_eth[demos$Race == "White" & demos$Ethnicity == "Hispanic/Latino" ] <- "White and Hispanic"
demos$race_eth[demos$Race == "Black or African Am." & demos$Ethnicity =="Hispanic/Latino" ] <- "Black or African American and Hispanic"
demos$race_eth[demos$Race == "Am. Indian/Alaskan Nat." & demos$Ethnicity =="Hispanic/Latino" ] <- "Native American and Hispanic"
demos$race_eth[demos$Race == "Asian/Nat. Hawaiian/Othr Pacific Is." & demos$Ethnicity =="Hispanic/Latino" ] <- "Asian and Hispanic"
demos$race_eth[demos$Race == "More than one" & demos$Ethnicity =="Hispanic/Latino" ] <- "Multi-racial and Hispanic"
demos$race_eth[demos$Race == "Unknown or Not Reported" & demos$Ethnicity =="Hispanic/Latino" ] <- "Unknown or not reported and Hispanic"

demos$race_eth[demos$Race == "White" &demos$Ethnicity == "Not Hispanic/Latino"] <- "White"
demos$race_eth[demos$Race == "Black or African Am." & demos$Ethnicity =="Not Hispanic/Latino" ] <- "Black or African American"
demos$race_eth[demos$Race == "Am. Indian/Alaskan Nat." & demos$Ethnicity =="Not Hispanic/Latino" ] <- "Native American"
demos$race_eth[demos$Race == "Asian/Nat. Hawaiian/Othr Pacific Is." & demos$Ethnicity == "Not Hispanic/Latino"] <- "Asian"
demos$race_eth[demos$Race == "More than one" & demos$Ethnicity =="Not Hispanic/Latino" ] <- "Multi-racial"
demos$race_eth[demos$Race == "Unknown or Not Reported" & demos$Ethnicity =="Not Hispanic/Latino" ] <- "Unknown or not reported"

demos$race_eth[demos$Race == "White" & demos$Ethnicity =="Unknown or Not Reported"] <- "White"
demos$race_eth[demos$Race == "Black or African Am." & demos$Ethnicity =="Unknown or Not Reported"] <- "Black or African American"
demos$race_eth[demos$Race == "Am. Indian/Alaskan Nat." & demos$Ethnicity =="Unknown or Not Reported" ] <- "Native American"
demos$race_eth[demos$Race == "Asian/Nat. Hawaiian/Othr Pacific Is." & demos$Ethnicity =="Unknown or Not Reported"] <- "Asian"
demos$race_eth[demos$Race == "More than one" & demos$Ethnicity =="Unknown or Not Reported" ] <- "Multi-racial"
demos$race_eth[demos$Race == "Unknown or Not Reported" & demos$Ethnicity =="Unknown or Not Reported" ] <- "Unknown or not reported"

demos$race_eth <- ordered(demos$race_eth, levels = c("White", "Black or African American"  , "Asian" ,
                                             "Multi-racial" , "Unknown or not reported and Hispanic","Multi-racial and Hispanic",
                                             "Black or African American and Hispanic", "White and Hispanic"))

unres<-read.table("~/Google Drive/HCP/HCP_graph/1200/datasets/unrestricted_gshearrer_4_19_2018_11_31_37.csv", header=T, sep=",")
myvars<-c("Subject","Gender")
unres<-unres[myvars]
unres$sub<-unres$Subject
head(unres)

data<-join(tbl,demos)
dim(data)
names(data)
data<-join(data, unres)
dim(data)


#Cleanup, center, ect
cols <- c("sub","Gender" ,"ZygosityGT", "Ethnicity", "Mother_ID","Father_ID","group","race_eth", "measure")

data %<>% mutate_at(cols, funs(factor(.)))
str(data[100:114])

data$group <- ordered(data$group, levels = c("no", "ov", "ob"))


## demographics
dat<-data[!duplicated(data[,c('sub')]),]
levels(dat$group)    
levels(dat$race_eth)

#          no  ov  ob
cons<- list(
  noVov = c(1, -1,  0),
  noVob = c(1,  0, -1),
  ovVob = c(0,  1, -1)
) 

#Age
dm1<-lm(Age_c~group, data=dat)
lsq_dm1 = lsmeans(dm1, "group")
contrast(lsq_dm1, cons, adjust='sidak')

violin(dat,dat$group,dat$Age_c)
describeBy(dat$Age_in_Yrs, dat$group)
## No difference in age 

#BMI
dm2<-lm(BMI~group, data=dat)
lsq_dm2 = lsmeans(dm2, "group")
contrast(lsq_dm2, cons, adjust='sidak')
## Sanity check difference between the groups
violin(dat,dat$group,dat$BMI)
describeBy(dat$BMI, dat$group)

#HBA1c
dm3<-lm(HBA1c_c~group, data=dat)
lsq_dm3 = lsmeans(dm3, "group")
contrast(lsq_dm3, cons, adjust='sidak')

violin(dat,dat$group,dat$HbA1C)
describeBy(dat$HbA1C, dat$group)

#sex
chisq.test(table(dat$group, dat$Gender))
tapply(dat$group, dat$Gender, summary)

balloonplot(table(dat$group, dat$Gender), main ="Gender Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, colmar=5)


#Race
chisq.test(table(dat$group, dat$race_eth))
tapply(dat$group, dat$race_eth, summary)

balloonplot(table(dat$group, dat$race_eth), main ="Racial/Ethnic Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[3], colmar=5)

# Models
fit_function<-function(y,data){
  fit<-lm(y~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=data)
  # sumfit<-summary(fit)
  lsq_c1 = lsmeans(fit, "group")
  contrast(lsq_c1, cons, adjust='sidak')

}


## clustering
cluster<-subset(data, data$measure == "cluster")
head(cluster$FC)
fit_function(y = cluster$IC_29, data = cluster)
apply(cluster[2:101], FUN=fit_function, MARGIN = 2 ,data=cluster)

## clustering
cluster<-subset(data, data$measure == "cluster")
fit_function(y = cluster$IC_0, data = cluster)
apply(cluster[2:101], FUN=fit_function, MARGIN = 2 ,data=cluster)
violin(cluster,cluster$group,cluster$IC_16)
violin(cluster,cluster$group,cluster$IC_29)

## centrality
centrality<-subset(data, data$measure == "centrality")
apply(centrality[2:101], FUN=fit_function, MARGIN = 2 ,data=centrality)
#IC_58 noVov 0.0098
fit<-lm(IC_58~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=centrality)
summary(fit)
violin(centrality,centrality$group,centrality$IC_58)
#IC_27 noVob 0.0065
violin(centrality,centrality$group,centrality$IC_27)
fit<-lm(IC_27~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=centrality)
summary(fit)

## participation coeffiecent 
PC<-subset(data, data$measure == "PC")
apply(PC[2:101], FUN=fit_function, MARGIN = 2 ,data=PC)

### To examine associations between SES and development within putative functional networks
data1<-data.frame(data)
labels<-read.table("~/Google Drive/HCP/HCP_graph/1200/datasets/tmp/mod_labels.csv", header=T, sep=",")
nam <- as.vector(labels$area)
old_nam<-as.vector(names(data1[2:101]))

setnames(data1, old=old_nam, new=nam)

head(data1)
head(data)
mdata <- melt(data1, id=c("sub","group","measure","FC","Age_in_Yrs","ZygosityGT","Race","Ethnicity","BMI","HbA1C","Hypothyroidism", "Hyperthyroidism","OtherEndocrn_Prob","Mother_ID","Father_ID","Age_c","HBA1c_c","race_eth","Gender"))
head(mdata)

#  "no.Visual","ov.Visual","ob.Visual", "no.Default","ov.Default", "ob.Default", "no.CinguloOperc" ,"ov.CinguloOperc","ob.CinguloOperc","no.FrontoParietal","ov.FrontoParietal","ob.FrontoParietal","no.DorsalAttn","ov.DorsalAttn","ob.DorsalAttn","no.MedialParietal","ov.MedialParietal","ob.MedialParietal","no.Smmouth","ov.Smmouth","ob.Smmouth",            
# "no.Smhand","ov.Smhand","ob.Smhand", "no.None","ov.None" ,"ob.None","no.VentralAttn","ov.VentralAttn","ob.VentralAttn", "no.Cerebellum","ov.Cerebellum", "ob.Cerebellum","no.Auditory", "ov.Auditory","ob.Auditory", "no.","ov.", "ob.","no.Amygdala" , "ov.Amygdala","ob.Amygdala","no.Caudate","ov.Caudate", "ob.Caudate", "no.Putamen","ov.Putamen","ob.Putamen", "no.Thalamus",           
# "ov.Thalamus","ob.Thalamus", "no.BrainStem","ov.BrainStem","ob.BrainStem","no. Cerebellum","ov. Cerebellum","ob. Cerebellum","no.VentralDiencephalon", "ov.VentralDiencephalon", "ob.VentralDiencephalon"
zeros<-rep(0, each=57)
visual<-c(1,0,-1,zeros)
length(visual)
##############################
prezeros<-rep(0, each=3)
zeros<-rep(0, each=54)
default <-c(prezeros,1,0,-1,zeros)
length(default)
##############################
prezeros<-rep(0, each=6)
zeros<-rep(0, each=51)
cinguloOperc<-c(prezeros,1,0,-1,zeros)
length(cinguloOperc)
##############################
##############################
prezeros<-rep(0, each=9)
zeros<-rep(0, each=48)
frontoParietal<-c(prezeros,1,0,-1,zeros)
length(frontoParietal)
##############################
##############################
prezeros<-rep(0, each=12)
zeros<-rep(0, each=45)
dorsalAttn<-c(prezeros,1,0,-1,zeros)
length(dorsalAttn)
##############################
##############################
prezeros<-rep(0, each=15)
zeros<-rep(0, each=42)
medialParietal<-c(prezeros,1,0,-1,zeros)
length(medialParietal)
##############################
##############################
prezeros<-rep(0, each=18)
zeros<-rep(0, each=39)
smmouth<-c(prezeros,1,0,-1,zeros)
length(smmouth)
##############################
##############################
prezeros<-rep(0, each=21)
zeros<-rep(0, each=36)
smhand<-c(prezeros,1,0,-1,zeros)
length(smhand)
##############################
##############################
prezeros<-rep(0, each=24)
zeros<-rep(0, each=33)
none<-c(prezeros,0,0,0,zeros)
length(none)
##############################
##############################
prezeros<-rep(0, each=27)
zeros<-rep(0, each=30)
ventralAttn<-c(prezeros,1,0,-1,zeros)
length(ventralAttn)
##############################
##############################
prezeros<-rep(0, each=30)
zeros<-rep(0, each=27)
cbellum<-c(prezeros,1,0,-1,zeros)
length(cbellum)
##############################
##############################
prezeros<-rep(0, each=33)
zeros<-rep(0, each=24)
auditory<-c(prezeros,1,0,-1,zeros)
length(auditory)
##############################
##############################
prezeros<-rep(0, each=36)
zeros<-rep(0, each=21)
group<-c(prezeros,1,0,-1,zeros)
length(group)
##############################
##############################
prezeros<-rep(0, each=39)
zeros<-rep(0, each=18)
amygdala<-c(prezeros,1,0,-1,zeros)
length(amygdala)
##############################
##############################
prezeros<-rep(0, each=42)
zeros<-rep(0, each=15)
caudate<-c(prezeros,1,0,-1,zeros)
length(caudate)
##############################
##############################
prezeros<-rep(0, each=45)
zeros<-rep(0, each=12)
putamen<-c(prezeros,1,0,-1,zeros)
length(putamen)
##############################
##############################
prezeros<-rep(0, each=48)
zeros<-rep(0, each=9)
thalamus<-c(prezeros,1,0,-1,zeros)
length(thalamus)
##############################
##############################
prezeros<-rep(0, each=51)
zeros<-rep(0, each=6)
brainstem<-c(prezeros,1,0,-1,zeros)
length(brainstem)
##############################
##############################
prezeros<-rep(0, each=54)
zeros<-rep(0, each=3)
cbellum2<-c(prezeros,1,0,-1,zeros)
length(cbellum2)
##############################
##############################
prezeros<-rep(0, each=57)
ventralDiencephalon<-c(prezeros,1,0,-1)
length(ventralDiencephalon)
##############################

con_air<- list(
  visual,
  default ,
  cinguloOperc,
  frontoParietal,
  dorsalAttn,
  medialParietal,
  smmouth,
  smhand,
  ventralAttn,
  auditory,
  amygdala,
  caudate,
  putamen,
  thalamus,
  brainstem,
  ventralDiencephalon,
  cbellum,
  cbellum2,
  group,
  none
) 
length(con_air)

## clustering
mcluster<-subset(mdata, mdata$measure == "cluster")
head(mcluster)
mcluster$GroupArea <- interaction(mcluster$group, mcluster$variable)

m1<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=mcluster)
summary(m1)
lsq_m1 = lsmeans(m1, "GroupArea")
contrast(lsq_m1, con_air, adjust='sidak')

## centrality
mcentrality<-subset(mdata, mdata$measure == "centrality")
head(mcentrality)
mcentrality$GroupArea <- interaction(mcentrality$group,mcentrality$variable)

m2<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=mcentrality)
summary(m2)
lsq_m2 = lsmeans(m2, "GroupArea")
contrast(lsq_m2, con_air, adjust='sidak')

## participation coeffiecent 
mPC<-subset(mdata, mdata$measure == "PC")
head(mPC)
mPC$GroupArea <- interaction(mPC$group,mPC$variable)

m3<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=mPC)
summary(m3)
lsq_m3 = lsmeans(m3, "GroupArea")
contrast(lsq_m3, con_air, adjust='sidak')


### To examine associations between SES and development within modules
data2<-data.frame(data)

mods<-read.table("~/Google Drive/HCP/HCP_graph/1200/datasets/tmp/mod_data.csv", header=T, sep=",")
head(mods)
names(data2)
data3 <- melt(data2, id=c("sub","group","measure","FC","Age_in_Yrs","ZygosityGT","Race","Ethnicity","BMI","HbA1C","Hypothyroidism", "Hyperthyroidism","OtherEndocrn_Prob","Mother_ID","Father_ID","Age_c","HBA1c_c","race_eth","Gender"))
data3$Index<-data3$variable
head(data3, 25)

names(mods)
vars<-c("Index","area","group","modules")
just_mods<-mods[vars]
head(just_mods)

data4<-join(data3,just_mods)
head(data4)

#normal
normal<-data.frame(data2)
normal<-subset(data2,data2$group == 'no')
no_mod<-subset(mods, mods$group == 'no')
no_nam <- as.character(as.vector(no_mod$modules))
no_old_nam<-as.vector(names(normal[2:101]))
setnames(normal, old=no_old_nam, new=no_nam)
head(normal)
mNo <- melt(normal, id=c("sub","group","measure","FC","Age_in_Yrs","ZygosityGT","Race","Ethnicity","BMI","HbA1C","Hypothyroidism", "Hyperthyroidism","OtherEndocrn_Prob","Mother_ID","Father_ID","Age_c","HBA1c_c","race_eth","Gender"))
head(mNo)

#over
over<-data.frame(data2)
over<-subset(over,data2$group == 'ov')
ov_mod<-subset(mods, mods$group == 'ov')
ov_nam <- as.character(as.vector(ov_mod$modules))
ov_old_nam<-as.vector(names(over[2:101]))
setnames(over, old=ov_old_nam, new=ov_nam)
head(over)
mOv <- melt(over, id=c("sub","group","measure","FC","Age_in_Yrs","ZygosityGT","Race","Ethnicity","BMI","HbA1C","Hypothyroidism", "Hyperthyroidism","OtherEndocrn_Prob","Mother_ID","Father_ID","Age_c","HBA1c_c","race_eth","Gender"))
head(mOv)

#obese
obese<-data.frame(data2)
obese<-subset(obese,data2$group == 'ob')
ob_mod<-subset(mods, mods$group == 'ob')
ob_nam <- as.character(as.vector(ob_mod$modules))
ob_old_nam<-as.vector(names(obese[2:101]))
setnames(obese, old=ob_old_nam, new=ob_nam)
head(obese)
mOb <- melt(obese, id=c("sub","group","measure","FC","Age_in_Yrs","ZygosityGT","Race","Ethnicity","BMI","HbA1C","Hypothyroidism", "Hyperthyroidism","OtherEndocrn_Prob","Mother_ID","Father_ID","Age_c","HBA1c_c","race_eth","Gender"))
head(mOb)
dim(mOb)

#Back together
longmodData<-rbind(mNo, mOv, mOb)
head(longmodData)
summary(longmodData)

longmodData$GroupMod <- interaction(longmodData$group, longmodData$variable)
levels(longmodData$GroupMod)
# [1] "no.0" "ov.0" "ob.0" "no.1" "ov.1" "ob.1" "no.2" "ov.2" "ob.2" "no.3" "ov.3" "ob.3" "no.4" "ov.4" "ob.4" "no.5" "ov.5" "ob.5" "no.6" "ov.6" "ob.6"
# [22] "no.7" "ov.7" "ob.7" "ov.8" "ob.8" "ov.9"
zeros<-rep(0, each=24)
mod0<-c(1,0,-1,zeros)
length(mod0)
##############################
prezeros<-rep(0, each=3)
zeros<-rep(0, each=21)
mod1 <-c(prezeros,1,0,-1,zeros)
length(mod1)
##############################
prezeros<-rep(0, each=6)
zeros<-rep(0, each=18)
mod2<-c(prezeros,1,0,-1,zeros)
length(mod2)
##############################
##############################
prezeros<-rep(0, each=9)
zeros<-rep(0, each=15)
mod3<-c(prezeros,1,0,-1,zeros)
length(mod3)
##############################
##############################
prezeros<-rep(0, each=12)
zeros<-rep(0, each=12)
mod4<-c(prezeros,1,0,-1,zeros)
length(mod4)
##############################
##############################
prezeros<-rep(0, each=15)
zeros<-rep(0, each=9)
mod5<-c(prezeros,1,0,-1,zeros)
length(mod5)
##############################
##############################
prezeros<-rep(0, each=18)
zeros<-rep(0, each=6)
mod6<-c(prezeros,1,0,-1,zeros)
length(mod6)
##############################
##############################
prezeros<-rep(0, each=21)
zeros<-rep(0, each=3)
mod7<-c(prezeros,1,0,-1,zeros)
length(mod7)
##############################
##############################
prezeros<-rep(0, each=24)
zeros<-rep(0, each=0)
mod8<-c(prezeros,1,0,-1,zeros)
length(mod8)
##############################
##############################
prezeros<-rep(0, each=27)
mod9<-c(prezeros)
length(mod9)


con_mod<- list(
  mod0,mod1,mod2,mod3,mod4,mod5,mod6,mod7,mod8,mod9
) 
con_mod

length(con_mod[[4]])

## clustering
mod_cluster<-subset(longmodData, longmodData$measure == "cluster")

mod_cluster$GroupArea <- interaction(mod_cluster$group, mod_cluster$variable)
mod1<-lm(value~GroupMod+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_cluster)
summary(mod1)
lsq_mod1 = lsmeans(mod1, "GroupMod")
contrast(lsq_mod1, con_mod, adjust='sidak')
# mod 4 is significant 
# estimate         SE     df    t.ratio p.value
# 0.039612646 0.01092282 1968   3.627  0.0029
## Drop non-sig to plot
mod_cluster4 <-mod_cluster[mod_cluster$variable == "4",]
mod1.4<-lm(value~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_cluster4)
summary(mod1.4)
violin(mod_cluster4, mod_cluster4$group, mod_cluster4$value)

## centrality 
mod_centrality<-subset(longmodData, longmodData$measure == "centrality")
mod_centrality$GroupArea <- interaction(mod_centrality$group, mod_centrality$variable)
mod2<-lm(value~GroupMod+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_centrality)
summary(mod2)
lsq_mod2 = lsmeans(mod2, "GroupMod")
contrast(lsq_mod2, con_mod, adjust='sidak')

# mod   estimate           SE      df   t.ratio p.value
# mod 4 0.0017035409 0.0003452200 1968   4.935  <.0001
# mod 8 0.0011282821 0.0003202495 1968   3.523  0.0044
## Drop non-sig to plot
mod_centrality4 <-mod_centrality[mod_centrality$variable == "4",]
mod2.4<-lm(value~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_centrality4)
summary(mod2.4)
head(mod_centrality4)
violin(mod_centrality4, mod_centrality4$group, mod_centrality4$value)
## Drop non-sig to plot
mod_centrality8 <-mod_centrality[mod_centrality$variable == "8",]
mod2.8<-lm(value~group+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_centrality8)
summary(mod2.8)
violin(mod_centrality8, mod_centrality8$group, mod_centrality8$value)

## PC 
mod_PC<-subset(longmodData, longmodData$measure == "PC")
mod_PC$GroupArea <- interaction(mod_PC$group, mod_PC$variable)
mod3<-lm(value~GroupMod+Age_c+HBA1c_c+Gender+race_eth+FC, data=mod_PC)
summary(mod3)
lsq_mod3 = lsmeans(mod3, "GroupMod")
contrast(lsq_mod3, con_mod, adjust='sidak')

### Testing frequencies
if(!require(psych)){install.packages("psych")}
if(!require(vcd)){install.packages("vcd")}
if(!require(DescTools)){install.packages("DescTools")}
if(!require(rcompanion)){install.packages("rcompanion")}
head(mods)
freq<-xtabs(~ modules+area+group, data=mods)
ftable(freq)
woolf_test(freq) # not significant use CMH test
mantelhaen.test(freq, exact=T) # significant
mods$AreaMods <- interaction(mods$modules, mods$area)
lilfreq<-xtabs(~ AreaMods+group, data=mods)
lilfreq
library(rcompanion)
library(RVAideMemoire)
module0<-subset(mods, mods$modules == "0")
freq0<-xtabs(~ area+group, data=module0)
freq0
f0<-fisher.test(freq0, simulate.p.value = TRUE, B = 1e6) #error

module1<-subset(mods, mods$modules == "1")
freq1<-xtabs(~ area+group, data=module1)
freq1
f1<-fisher.test(freq1, simulate.p.value = TRUE, B = 1e6)#no diff

module2<-subset(mods, mods$modules == "2")
freq2<-xtabs(~ area+group, data=module2)
freq2
f2<-fisher.test(freq2, simulate.p.value = TRUE, B = 1e6)#no diff

module3<-subset(mods, mods$modules == "3")
freq3<-xtabs(~ area+group, data=module3)
freq3
f3<-fisher.test(freq3, simulate.p.value = TRUE, B = 1e6)#no diff

module4<-subset(mods, mods$modules == "4")
freq4<-xtabs(~ area+group, data=module4)
freq4
f4<-fisher.test(freq4, simulate.p.value = TRUE, B = 1e6)#significant 
inner4<-subset(data4, data4$modules == "4")
incluster_4<-subset(inner4, data4$measure == "cluster")
inner4_con<-list(
  amygdala=  c(1,0,0,0,0,0,0,0),
  Cerebellum=c(0,1,-1,0,0,0,0,0),
  default=   c(0,0,0,1,0,0,0,0),
  dorsalAttn=c(0,0,0,0,1,0,0,0),
  frontoParl=c(0,0,0,0,0,1,0,0),
  medialPa=  c(0,0,0,0,0,0,1,0),
  None=      c(0,0,0,0,0,0,0,1)
)
incluster_4$GroupArea <- interaction(incluster_4$group, incluster_4$area)
inclu4.1<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=incluster_4)
summary(inclu4.1)
lsq_mod2 = lsmeans(inclu4.1, "GroupArea")
contrast(lsq_mod2, inner4_con, adjust='sidak')

module5<-subset(mods, mods$modules == "5")
freq5<-xtabs(~ area+group, data=module5)
freq5
f5<-fisher.test(freq5, simulate.p.value = TRUE, B = 1e6)#significant 
inner5<-subset(data4, data4$modules == "5")
incluster_5<-subset(inner5, inner5$measure == "cluster")
inner5_con<-list(
  caudate=   c(1,-1,0,0,0,0,0,0,0,0,0,0),
  Cerebellum=c(0,0,1,0,0,0,0,0,0,0,0,0),
  dorsalAttn=c(0,0,0,1,0,0,0,0,0,0,0,0),
  frontoParl=c(0,0,0,0,1,0,0,0,0,0,0,0),
  medialParl=c(0,0,0,0,0,1,0,0,0,0,0,0),
  putamen=   c(0,0,0,0,0,0,1,-1,0,0,0,0),
  Thalamus=  c(0,0,0,0,0,0,0,0,1,-1,0,0),
  ventralA=  c(0,0,0,0,0,0,0,0,0,0,1,0),
  ventralDia=c(0,0,0,0,0,0,0,0,0,0,0,1)
)
incluster_5$GroupArea <- interaction(incluster_5$group, incluster_5$area)
inclu5.1<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=incluster_5)
summary(inclu5.1)
lsq_mod5.1 = lsmeans(inclu5.1, "GroupArea")
contrast(lsq_mod5.1, inner5_con, adjust='sidak')

module6<-subset(mods, mods$modules == "6")
freq6<-xtabs(~ area+group, data=module6)
freq6
f6<-fisher.test(freq6, simulate.p.value = TRUE, B = 1e6)#significant 
inner6<-subset(data4, data4$modules == "6")
incluster_6<-subset(inner6, inner6$measure == "cluster")
inner6_con<-list(
  none=      c(1,0,0,0,0,0,0),
  brainstem= c(0,1,0,0,0,0,0),
  caudate=   c(0,0,1,0,0,0,0),
  cerbel=    c(0,0,0,1,0,0,0),
  frontoParl=c(0,0,0,0,1,0,0),
  putamen=   c(0,0,0,0,0,1,0),
  Thalamus=  c(0,0,0,0,0,0,1)
)
incluster_6$GroupArea <- interaction(incluster_6$group, incluster_6$area)
inclu6.1<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=incluster_6)
summary(inclu6.1)
lsq_mod6.1 = lsmeans(inclu6.1, "GroupArea")
contrast(lsq_mod6.1, inner6_con, adjust='sidak')

module7<-subset(mods, mods$modules == "7")
freq7<-xtabs(~ area+group, data=module7)
freq7
f7<-fisher.test(freq7, simulate.p.value = TRUE, B = 1e6)#significant 

module8<-subset(mods, mods$modules == "8")
freq8<-xtabs(~ area+group, data=module8)
freq8
f8<-fisher.test(freq8, simulate.p.value = TRUE, B = 1e6)#significant 
inner8<-subset(data4, data4$modules == "8")
incluster_8<-subset(inner8, inner8$measure == "cluster")
inner8_con<-list(
  VD=        c(0,0,0,1),
  brainstem= c(1,-1,0,0),
  caudate=   c(0,0,1,0),
  cerbel=    c(0,0,1,0)
)
incluster_8$GroupArea <- interaction(incluster_8$group, incluster_8$area)
inclu8.1<-lm(value~GroupArea+Age_c+HBA1c_c+Gender+race_eth+FC, data=incluster_8)
summary(inclu8.1)
lsq_mod8.1 = lsmeans(inclu8.1, "GroupArea")
contrast(lsq_mod8.1, inner8_con, adjust='sidak')
# brainstem is significantly different

module9<-subset(mods, mods$modules == "9")
freq9<-xtabs(~ area+group, data=module9)
freq9
f9<-fisher.test(freq9, simulate.p.value = TRUE, B = 1e6)#error

fp<-c(f1$p.value, f2$p.value,f3$p.value, f4$p.value, f5$p.value, f6$p.value, f7$p.value, f8$p.value)
corr_fp<-p.adjust(fp,method = "BH")<0.05
corr_fp
#different 3,4,5,6,7,8
##################################################################################################################################

#group data
data <- read.table("~/Google Drive/HCP_graph/1200/datasets/tmp/mod_data.csv", header=T, sep=",")
head(data)

# Data preparation
No<-subset(data, data$group == "no")
t1<-xtabs(~area+modules, data=No)
head(t1)
# Visualization
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

balloonplot(t1, main ="Healthy Weight Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[1], colmar=5)

Ov<-subset(data, data$group == "ov")
t2<-xtabs(~area+modules, data=Ov)
head(t2)
# Visualization
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

balloonplot(t2, main ="Overweight Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[2], colmar=5)

Ob<-subset(data, data$group == "ob")
t3<-xtabs(~area+modules, data=Ob)
head(t3)
# Visualization
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

balloonplot(t3, main ="Obese Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[3], colmar=5)


modFP<-subset(data, data$area == 'FrontoParietal')
mytable <- table(modFP$group, modFP$modules)
chisq.test(mytable) 
mytable

modFP$modules<-as.factor(modFP$modules)
m1<-lm(PC~group, data=modFP)
summary(m1)

m2<-lm(centrality~group, data=modFP)
summary(m2)

m3<-lm(clustering~group, data=modFP)
summary(m3)

modCO<-subset(data, data$area == 'CinguloOperc')
mytable <- table(modCO$group, modCO$modules)
chisq.test(mytable) 

modDA<-subset(data, data$area == 'DorsalAttn')
mytable <- table(modDA$group, modDA$modules)
chisq.test(mytable) 
mytable


modVA<-subset(data, data$area == 'VentralAttn')
mytable <- table(modVA$group, modVA$modules)
chisq.test(mytable) 
mytable

