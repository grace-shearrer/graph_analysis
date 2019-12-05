library(ggplot2)
library(nlme)
library(lme4)
library(psych)

demos<-read.table("~/Google Drive/HCP_graph/1200/datasets/demos.csv", sep=",", header=T)


head(demos)
describeBy(demos$BMI,demos$ov_ob )
describeBy(demos$Age_in_Yrs,demos$ov_ob )
describeBy(demos$HbA1C,demos$ov_ob )

df<-read.table("~/Google Drive/HCP_graph/1200/datasets/tmp/mod_data.csv", sep=",", header=T)
head(df)



df$modules<-as.factor(df$modules)
df$group<-as.factor(df$group)
names(df)
myvars<-c("Index","centrality","clustering", "PC","modules","group","re_gordon","label")
df<-df[myvars]
head(df)

df$group <- factor(df$group, levels = c("no", "ov", "ob"))
library(plyr)
df$group<-revalue(df$group, c("no"="18-25", "ov"="25-30", "ob"=">30"))
summary(df$PC)
df_sub<-subset(df, df$PC > -30)
plot1<-ggplot(df_sub, aes(group, PC)) + 
  geom_violin(aes(fill = group),stat = "ydensity",draw_quantiles = c(0.25, 0.5, 0.75))+
  labs(x = "BMI group", y="Participation coefficent") + theme_bw()
plot1


df0<-subset(df, df$modules == "0")
head(df0)

mytable <- table(df0$group, df0$label)
mytable

df1<-subset(df, df$modules == "1")
head(df1)
mytable1 <- table(df1$group, df1$label)
mytable1

mytable <- table(df0$group, df0$modules, df0$label)
ftable(mytable)
summary(mytable)



par(mfrow=c(1,2)) # put the next two plots side by side
plot(PC ~ group, data=df, main="By Group")
plot(PC ~ modules, data=df, main="By Module")
par(mfrow=c(1,1))


# specify contrast weights
levels(df$modules)

c1 <- c(0,0,0,0,0,0,0,0,0,0) 
c2 <- c(0,0,0,0,0,0,0,0,0,0)
c3 <- c(0,0,0,0,0,0,0,0,0,0)
c4 <- c(0,0,0,0,0,0,0,0,0,0)
c5 <- c(0,0,0,0,0,0,0,0,0,0)
c6 <- c(0,0,0,0,0,0,0,0,0,0)
c7 <- c(0,0,0,0,0,0,0,0,0,0)

# create temporary matrix
mat.temp <- rbind(constant=1/4, c1, c2, c3)
mat.temp


m0<-lm(PC~group, data=df)
summary(m0)
names(df)
m1<-lm(centrality~group, data=df)
summary(m1)
m2<-lm(clustering~group, data=df)
summary(m2)


m3<-lm(PC~modules*group, data=df)
summary(m3)

m4<-lm(centrality~modules*group, data=df)
summary(m4)

m5<-lm(clustering~modules*group, data=df)
summary(m5)
