library(ISLR)
library(MASS)
library(car)

###Chapter 3: Linear Regression

#------------------SIMPLE LINEAR REGRESSION
fix(Boston)
names(Boston)
lm.fit=lm(medv~lstat,data=Boston)
attach(Boston)
lm.fit=lm(medv~lstat)
lm.fit
summary(lm.fit)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

#Prediction and Confidence intervals for prediction of Y for a given value of X ex. (c(5,10,15)).
predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="confidence")
predict(lm.fit,data.frame(lstat=(c(5,10,15))), interval="prediction")
plot(lstat,medv) 
abline(lm.fit)

par(mfrow = c(2,2))
plot(lm.fit)

par(mfrow = c(1,1))
plot(predict (lm.fit), residuals (lm.fit)) # evidence of non linearity
plot(predict (lm.fit), rstudent (lm.fit))
plot(hatvalues(lm.fit)) #leverage values(outliers to get rid of) see pics
which.max(hatvalues(lm.fit)) #This one has the largest leverage statistic

#------------------------- MULTIPLE LINEAR REGRESSION
attach(Boston)
lm.fit <- lm(medv ~ lstat+age,  data = Boston)
summary(lm.fit)

#To have all variables
lm.fit <- lm(medv~., data = Boston)
summary(lm.fit)

#Check multicollinearity (VIF < 5 or VIF < 10)
vif(lm.fit)

#age has a high p-value so remove it etc.
lm.fit <- lm(medv~.-age, data = Boston)
summary(lm.fit)


#Interaction terms and base terms of lstat and age 
summary (lm(medv~lstat*age ,data=Boston))

#Polynomial fit
lm.fit2 <- lm(medv~lstat+I(lstat^2))
summary(lm.fit2)
#The near zero p-values suggests a good fit. We can further quantify how much supperior this is to the linear
lm.fit <- lm(medv~lstat, data = Boston)
anova(lm.fit , lm.fit2) #high F-stat and low p-value suggest that Model2 is better

par(mfrow=c(2,2))
plot(lm.fit2)

#5 Poly model
lm.fit5=lm(medv~poly(lstat ,5))
summary(lm.fit5) # model looks even better

#Log model
summary(lm(medv~log(rm),data=Boston))


#------------------QUALITATIVE PREDICTORS
fix(Carseats)

#R generates dummy variables automatically
lm.fit <- lm(Sales~.+ Income:Advertising +Price:Age ,data=Carseats) #interaction terms but not base!
summary(lm.fit) #Positive coef for ShelveGood is good shelving location is associated with high sales(compared to bad loc)
#ShelvMed shows that better than bad but not as good for sales as good.

#To see dummy variable coding
contrasts(Carseats$ShelveLoc)



#------------------8
auto <- Auto
par(mfrow = c(1,1))
plot(y = auto$mpg, x = auto$horsepower) # negative Non-linear relationship
model <- lm(mpg ~ horsepower, data = auto)
summary(model)#large abs t-stat and low p-value. Stat sig relationship
#R-Squared is 0.6 which is good. 
#Large F-stat and small associated p-value: we can reject the null hypothesis and state there is a statistically significant relationship between horsepower and mpg.
par(mfrow = c(2,2))
plot(model)#Based on the residuals plots, there is some evidence of non-linearity.

predict(model,data.frame(horsepower=98), interval="confidence")
predict(model,data.frame(horsepower=98), interval="prediction")

#-----------------9
pairs(auto)
cor(subset(auto, select = -name))#There are some very high correlations

model <- lm(mpg ~.-name, auto)
vif(model)
summary(model) #there is a relatioship between the predictors and the response by testing the null hypothesis of whether all the regression coefficients are zero. 
#The F -statistic is far from 1 (with a small p-value), indicating evidence against the null hypothesis.
par(mfrow = c(2,2))
plot(model)#Pattern in residual plot
#From the leverage plot, point 14 appears to have high leverage, although not a high magnitude residual.

par(mfrow = c(1,1))
plot(predict(model), rstudent(model))
abline(h = 3)#The studentized residuals displays potential outliers (>3)
plot(hatvalues(model)) #outliers
which.max(hatvalues(model))

#From the correlation matrix pick the ones with the highest correlations
#and use them as interaction effects.
lm.model <- lm(mpg ~ cylinders*displacement+displacement*weight,auto)
summary(lm.model)#interaction between displacment and weight is stat sig

plot(auto$mpg, auto$weight)
plot(auto$mpg, log(auto$weight))#maybe better

plot(auto$mpg, auto$horsepower)
plot(auto$mpg, log(auto$horsepower))#maybe better

plot(auto$mpg, auto$acceleration)
plot(auto$mpg, (auto$acceleration^2))#better

lm.model2 <-lm(mpg~log(weight)+log(horsepower)+acceleration+I(acceleration^2), data = auto)
summary(lm.model2)
par(mfrow = c(2,2))
plot(lm.model2)#The residuals plot has less of a discernible pattern than the plot of all linear regression terms.
#The residuals vs fitted plot indicates heteroskedasticity 
#The Q-Q plot indicates somewhat unnormality of the residuals.
#The leverage plot indicates more than three points with high leverage.

par(mfrow = c(1,1))
plot(predict(lm.model2), rstudent(lm.model2))#The studentized residuals displays potential outliers (>3)
abline(h = 3)

#So, a better transformation need to be applied to our model.

par(mfrow = c(1,1))
plot(auto$mpg, auto$displacement)
plot(auto$mpg, log(auto$displacement))#better

lm.model2 <-lm(mpg~log(weight)+log(horsepower)+acceleration+I(acceleration^2) + log(auto$displacement) + year + origin + cylinders, data = auto)
summary(lm.model2)

par(mfrow = c(2,2))
plot(lm.model2)
par(mfrow = c(1,1))
plot(predict(lm.model2), rstudent(lm.model2))#The studentized residuals displays potential outliers (>3)
abline(h = 3)

#Still doesn't look good

lm.fit2<-lm(log(mpg)~cylinders+displacement+horsepower+weight+acceleration+year+origin,data=Auto)
par(mfrow=c(2,2)) 
plot(lm.fit2)
par(mfrow = c(1,1))
plot(predict(lm.fit2),rstudent(lm.fit2))
abline(h = 3)

#Much better


#-----------------------------10

carseats <- Carseats
lm.fit = lm(Sales~Price+Urban+US, carseats)
summary(lm.fit)
#Price high abs t-value and low p-value. Neg relationship
#UrbanYes does not have a stat sig effect on Sales. low t and p
#USYes has a pos signif effect when stores in US

#High F-stat with a low corresponding p-value

lm.fit2 = lm(Sales~Price+US, carseats)
summary(lm.fit2)
#they both fit the data similarly, with linear regression from (e) fitting the data slightly better.

confint(lm.fit2)

plot(predict(lm.fit2), rstudent(lm.fit2))
abline(h=3)
abline(h=-3)
#no outliers

par(mfrow=c(2,2))
plot(lm.fit2)
#some high leverage points


#----------------------12

Bos <- Boston
lm.zn = lm(crim~zn)
summary(lm.zn) # yes
lm.indus = lm(crim~indus)
summary(lm.indus) # yes
lm.chas = lm(crim~chas) 
summary(lm.chas) # no
lm.nox = lm(crim~nox)
summary(lm.nox) # yes
lm.rm = lm(crim~rm)
summary(lm.rm) # yes
lm.age = lm(crim~age)
summary(lm.age) # yes
lm.dis = lm(crim~dis)
summary(lm.dis) # yes
lm.rad = lm(crim~rad)
summary(lm.rad) # yes
lm.tax = lm(crim~tax)
summary(lm.tax) # yes
lm.ptratio = lm(crim~ptratio)
summary(lm.ptratio) # yes
lm.black = lm(crim~black)
summary(lm.black) # yes
lm.lstat = lm(crim~lstat)
summary(lm.lstat) # yes
lm.medv = lm(crim~medv)
summary(lm.medv) # yes

lm.all = lm(crim~., data=Boston)
summary(lm.all)
#we can reject zn, dis, rad, black, medv

#check if we need to other functional forms etc
