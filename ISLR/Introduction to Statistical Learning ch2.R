install.packages("ISLR")
library(ISLR)
library(MASS)

###Chapter 2: Statistical Learning

Col <- College
fix(Col) #Really interesing function!
pairs(Col[,1:5])

plot(x = Col$Private, y = Col$Outstate)

a <- ifelse(Col$Top10perc > 50, "Yes", "No")
a <- as.factor(a)
Col <- data.frame(Col, a)

plot(x = Col$a, y = Col$Outstate)

par(mfrow = c(2,2))
hist(Col$Apps)
hist(Col$perc.alumni, col=2)
hist(Col$S.F.Ratio, col=3, breaks=10)
hist(Col$Expend, breaks=100)

par(mfrow=c(1,1))
plot(Col$Outstate, Col$Grad.Rate)
# High tuition correlates to high graduation rate.
plot(Col$Accept / Col$Apps, Col$S.F.Ratio)
# Colleges with low acceptance rate tend to have low S:F ratio.

#-------------

auto <- Auto
auto <- na.omit(auto)

sapply(auto[,1:7], range)

sapply(auto[,1:7], mean)

auto <- auto[-(10:85),]
sapply(auto[,1:7], range)
sapply(auto[,1:7], mean)
pairs(auto) #to predict best to use all but name



