library(plyr)
library(ggplot2)
library(data.table)
library(dplyr)
library(formattable)
library(tidyr)
library(gplots)
library(lm.beta)
data<-read.table("~/Google Drive/ABCD/ABCD_puberty/bak/_data/demos copy.csv", header=T, sep=",")
head(data)
describe(data$BMItile)
describe(data$age)
summary(data$PCS)
summary(data$sex)
summary(data$race)
summary(data$ethnicity)


describeBy(data$BMItile, data$OVOB)
describeBy(data$age, data$OVOB)
tapply(data$PCS, data$OVOB, summary)
tapply(data$sex, data$OVOB, summary)
tapply(data$race, data$OVOB, summary)
tapply(data$ethnicity, data$OVOB, summary)

mytable <- xtabs(~PCS+OVOB, data=data)
summary(mytable) # chi-square test of indepedence

mytable <- xtabs(~sex+OVOB, data=data)
summary(mytable) # chi-square test of indepedence

mytable <- xtabs(~race+OVOB, data=data)
summary(mytable) # chi-square test of indepedence

mytable <- xtabs(~ethnicity+OVOB, data=data)
summary(mytable) # chi-square test of indepedence

m1<-lm(BMItile~OVOB, data=data)
summary(m1)

m1<-lm(age~OVOB, data=data)
summary(m1)


df<-read.table("~/Google Drive/ABCD/ABCD_puberty/bak/_data/df.csv", sep=",", header=T)
head(df)
df$OVOB
ov_ob<-c(0,1,-1)
contrasts(df$OVOB)<-cbind(ov_ob)

m0<-lm(CC~OVOB, data=df)
summary(m0)

df$OVOB <- ordered(df$OVOB, levels = c("normal", "overweight", "obese"))
df$OVOB

df$OVOB<-revalue(df$OVOB, c("normal"="5≤BMI%<85", "overweight"="85≤BMI%<95","obese"="BMI%≤95"))


# Basic violin plot
n=25
cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

p <- ggplot(df, aes(x=OVOB, y=CC, fill=OVOB)) + 
  geom_violin()+ geom_boxplot(
    width=0.1) + theme_classic()+ ylab(
      "Clustering Coefficient") + xlab(
        " ") + theme(
          axis.text.x  = element_text(size=n),
          axis.text.y = element_text(size=n), 
          axis.title.y = element_text(size=n))+scale_fill_manual(
            values=cbPalette)+ guides(fill=FALSE)



p

ov_ob<-c(0,1,-1)
contrasts(df$OVOB)<-cbind(ov_ob)

m2<-lm(PC~OVOB, data=df)
summary(m2)

q <- ggplot(df, aes(x=OVOB, y=PC, fill=OVOB)) + 
  geom_violin()+ geom_boxplot(
    width=0.1) + theme_classic()+ ylab(
      "Participation Coefficient") + xlab(
        " ") + theme(
          axis.text.x  = element_text(size=n),
          axis.text.y = element_text(size=n), 
          axis.title.y = element_text(size=n))+scale_fill_manual(
            values=cbPalette)+ guides(fill=FALSE)



q

locs<-read.table("~/Google Drive/ABCD/ABCD_puberty/bak/_data/locations.csv",header=TRUE, sep=",")
head(locs)
locs$unique<-locs$unique-1
locs$unique
df<-join(df,locs)
head(df)

t0<-xtabs(~Suggested.System+color+OVOB, data=df)
summary(t0)

No<-subset(df, df$OVOB=='5≤BMI%<85')
t1<-xtabs(~Suggested.System+color, data=No)
head(t1)

Ov<-subset(df, df$OVOB=='85≤BMI%<95')
t2<-xtabs(~Suggested.System+color, data=Ov)
head(t2)

Ob<-subset(df, df$OVOB=='BMI%≤95')
t3<-xtabs(~Suggested.System+color, data=Ob)
head(t3)

cbPalette <- c( "#E69F00", "#56B4E9",  "#CC79A7")

balloonplot(t1, main ="Healthy Weight Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[1], colmar=5)

balloonplot(t2, main ="Overweight Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[2], colmar=5)

balloonplot(t3, main ="Obese Frequencies", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt = 90,text.size=1.5 ,dotsize = 10, dotcolor=cbPalette[3], colmar=5)


btw<-read.table("~/Google Drive/ABCD/ABCD_puberty/_data/btw.csv",sep=",", header=T)
btw<-join(btw,locs)
head(btw)
b0<-lm(betweenness ~ OVOB*color*Suggested.System, data=btw)
summary(b0)
lm.beta(b0)

btw_soma<-subset(btw, btw$Suggested.System == 'Sensory/somatomotor Mouth')
head(btw_soma)
btw_soma$OVOB <- ordered(btw_soma$OVOB, levels = c("normal weight", "overweight", "obese"))
btw_soma$OVOB

btw_soma$OVOB<-revalue(btw_soma$OVOB, c("normal weight"="5≤BMI%<85", "overweight"="85≤BMI%<95","obese"="BMI%≤95"))

p <- ggplot(btw_soma, aes(x=OVOB, y=betweenness, fill=OVOB)) + 
  geom_violin()+ geom_boxplot(
    width=0.1) + theme_classic()+ ylab(
      "Betweenness in the \n oral somatosensory regions") + xlab(
        " ") + theme(
          axis.text.x  = element_text(size=n),
          axis.text.y = element_text(size=n), 
          axis.title.y = element_text(size=n))+scale_fill_manual(
            values=cbPalette)+ guides(fill=FALSE)



p
