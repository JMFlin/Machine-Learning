library(ISLR)
library(MASS)#functions
library(class)#functions
library(boot)#cross-validation
library(plotrix)#for std error function

###Chapter 5: Resampling Methods

#See Hungs script for other type of cross-validation. Here we test poly fits with CV

#----------------------------LOOCV
glm.fit <- glm(mpg~horsepower, data = Auto)#if you don't specify a family, glm uses linear regression same as lm()
coef(glm.fit)
plot(Auto$mpg,Auto$horsepower)#Non-linear so which poly form is best? Which minimizes error?

cv.err <- cv.glm(Auto, glm.fit)
cv.err$delta # the two numbers in the delta vector contain the cross-validation results
#Our cross-validation estimate for the test error is approximately 24.23.

cv.error <- rep(0,5)#We begin by initializing the vector.
for(i in 1:5){#calculate fits for polynomials from 1 to 5
  glm.fit <- glm(mpg~poly(horsepower, i), data = Auto)
  cv.error[i] <- cv.glm(Auto ,glm.fit)$delta[1]#automated calculation of LOOCV
}
cv.error
plot(cv.error, type = "o")# we see a sharp drop in the estimated test MSE between
#the linear and quadratic fits, but then no clear improvement from using
#higher-order polynomials. Poly 2 seems the best.


#------------------------k-Fold CV
plot(Auto$mpg,Auto$horsepower)#Non-linear so which poly form is best? Which minimizes error?
cv.error.10 <- rep(0,10)#We begin by initializing the vector.
for(i in 1:10){
  glm.fit <- glm(mpg~poly(horsepower,i), data = Auto)
  cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
} #computational time is much shorter than LOOCV
cv.error.10#lowest is best
plot(cv.error.10, type = "o")#Linear regression is the worst of the poly forms

#-----------------------The Bootstrap 208
fix(Portfolio)
#Suppose that we wish to invest a fixed sum of money in two financial
#assets that yield returns of X and Y , respectively, where X and Y are
#random quantities. We will invest a fraction alpha of our money in X, and will
#invest the remaining 1 - alpha in Y . Since there is variability associated with
#the returns on these two assets, we wish to choose alpha to minimize the total
#risk, or variance, of our investment.


#To illustrate the use of the bootstrap on this data, we must first create
#a function, alpha.fn(), which takes as input the (X, Y) data as well as
#a vector indicating which observations should be used to estimate alpha. The
#function then outputs the estimate for alpha based on the selected observations.


alpha.fn <- function(data, index){
  X <- data$X[index]
  Y <- data$Y[index]
  return ((var(Y)-cov(X,Y))/(var(X)+var(Y) -2*cov(X,Y)))
}
#This function returns, or outputs, an estimate for alpha based on applying
#(5.7)to the observations indexed by the argument index. For instance, the
#following command tells R to estimate alpha using all 100 observations nrow(Portfolio).

alpha.fn(Portfolio, 1:100)

#Below we produce R = 1, 000 bootstrap estimates for alpha
boot(Portfolio, alpha.fn,R=1000)
boot.out <- boot(Portfolio, alpha.fn,R=1000)
plot(boot.out)#Gaussian distribution, if it lines up on the straight line it is a gaussian distribution.
#Bootstrap is a good way to get a standard error for nasty statistics.
#The final output shows that using the original data, alpha hat = 0.5758, and that
#the bootstrap estimate for SE(alpha hat) is 0.0886
#so invest 58% into X and 1-0.58 to Y. Risk is minimized.

#-------------------------------Estimating the Accuracy of a Linear Regression Model

#The bootstrap approach can be used to assess the variability of the coef-
#ficient estimates and predictions from a statistical learning method. Here
#we use the bootstrap approach in order to assess the variability of the
#estimates for B0 and B1, the intercept and slope terms for the linear regression
#model

#We first create a simple function, boot.fn(), which takes in the Auto data
#set as well as a set of indices for the observations, and returns the intercept
#and slope estimates for the linear regression model. We then apply this
#function to the full set of 392 (nrow(Auto)) observations in order to compute the estimates
#of B0 and B1 on the entire data set using the usual linear regression
#coefficient estimate formulas

boot.fn <- function(data, index){
  glm.fit <- glm(mpg~horsepower, data = data, subset = index)
  glm.coef <- coef(glm.fit)
  return (glm.coef)
}

boot.fn(Auto ,nrow(Auto))

#Next, we use the boot() function to compute the standard errors of 1,000
#bootstrap estimates for the intercept and slope terms.
boot(Auto, boot.fn, 1000)

summary(glm(mpg~horsepower ,data=Auto))$coef
#The standard errors are different. The ones given by the bootstrap are better because they don't rely on 
#as many assumptions as the what is given in the summary() command
#In this case you should try to see if a better functional form of predictors is better.

boot.fn <- function(data, index){
  glm.fit <- glm(mpg~horsepower+I(horsepower^2), data = data, subset = index)
  glm.coef <- coef(glm.fit)
  return (glm.coef)
}

boot(Auto, boot.fn, 1000)
summary(glm(mpg~horsepower +I(horsepower^2),data=Auto))$coef
#there is now a better correspondence between the bootstrap estimates and
#the standard estimates of SE(B0hat), SE(B1hat) and SE(B2hat).


#---------------------VALIDATION SET APPROACH (NOT AS GOOD AS LOOCV OR CROSS-VALIDATION)


#Fit a logistic regression model that uses income and balance to
#predict default.
glm.fit <- glm(default~income+balance, data = Default, family = binomial)
summary(glm.fit)
exp(cbind(OR = coef(glm.fit), confint(glm.fit))) #confint shoulnd't include 1.

#Split the sample set into a training set and a validation set.
train <- sample(nrow(Default), nrow(Default)/2)
validation <- Default[-train,]#test set

#Fit a multiple logistic regression model using only the training
#observations.
glm.fit <- glm(default~income+balance, data=Default, family=binomial,
              subset=train)
#Obtain a prediction of default status for each individual in
#the validation set by computing the posterior probability of
#default for that individual, and classifying the individual to
#the default category if the posterior probability is greater
#than 0.5.
glm.probs <- predict(glm.fit, validation, type="response")
glm.pred <- ifelse(glm.probs > 0.5, "Yes", "No")

#Compute the validation set error, which is the fraction of
#the observations in the validation set that are misclassified.
mean(glm.pred != Default[-train,]$default)
pred.table <- table(glm.pred, validation$default)
1-(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#Miscalssification

#Repeat the process three times, using three different splits
#of the observations into a training set and a validation set. Comment
#on the results obtained.
#Run the train and validation through 3 times or build it into a function
#and call it 3 times
#mean looks to be 2.6%

glm.fit <- glm(default~income+balance+student, data = Default, family = binomial)
summary(glm.fit)
exp(cbind(OR = coef(glm.fit), confint(glm.fit))) #confint shoulnd't include 1.

train <- sample(nrow(Default), nrow(Default)/2)
test <- Default[-train,]

glm.fit <- glm(default~income+balance+student, data=Default, family=binomial,
               subset=train)
glm.probs <- predict(glm.fit, test, type="response")
glm.pred <- ifelse(glm.probs > 0.5, "Yes", "No")
pred.table <- table(glm.pred, test$default)
1-(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                       + pred.table[1,1] + pred.table[1,2])#Miscalssification
#2.64% test error rate, with student dummy variable. Using the validation set approach, 
#it doesn't appear adding the student dummy variable leads to a reduction in the test error rate.

boot.fn <- function(data = Default, index = 100){
  return (coef(glm(default~income+balance, data=data, family=binomial,
                   subset=index)))
}
glm.fit <- glm(default~income+balance, data=Default, family=binomial)
summary(glm.fit)
boot(Default, boot.fn, 50)
#standard errors look the same!

#------------------Cross-Validation
y=rnorm(100)
x=rnorm(100)
y=x-2*x^2+rnorm(100)

plot(x, y)#we need some poly model
?cv.glm
#compute the LOOCV errors that
#result from fitting the following four models using least squares:
#i. Y = B0 + B1X + e
#ii. Y = B0 + B1X + B2X^2 + e
#iii. Y = B0 + B1X + B2X^2 + B3X^3 + e
#iv. Y = B0 + B1X + B2X^2 + B3X^3 + B4X^4 + e

Data <- data.frame(x,y)

glm.fit <- glm(y~x)
cv.glm(Data, glm.fit)$delta

glm.fit <- glm(y~poly(x,2))
cv.glm(Data, glm.fit)$delta #lowest error rate, the best
summary(glm.fit)#p-values show statistical significance of linear and quadratic terms, which agrees with the CV results.

glm.fit <- glm(y~poly(x,3))
cv.glm(Data, glm.fit)$delta

glm.fit <- glm(y~poly(x,4))
cv.glm(Data, glm.fit)$delta


#------------------------BOOTSTRAP
attach(Boston)
#Based on this data set, provide an estimate for the population
#mean of medv.
medv.mean = mean(medv)
medv.mean
#std error by the normal way
std.error(medv)
#std error by bootstrap
boot.fn <- function(data = medv, index = 1000){
  return (mean(data[index]))
}
bstrap <- boot(medv, boot.fn, 1000)
bstrap #close to the same
#Based on your bootstrap estimate, provide a 95 % con-
#fidence interval for the mean of medv. Compare it to the results
#obtained using t.test(Boston$medv).
t.test(medv)
c(bstrap$t0 - 2*0.4119, bstrap$t0 + 2*0.4119)#both are close together
#calculate median
medv.median <- median(medv)
medv.median
#calculate median with bootstrap
boot.fn <- function(data = medv, index = 1000){
  return (median(data[index]))
}
bstrap <- boot(medv, boot.fn, 1000)
bstrap #the same, Small standard error relative to median value.
#calculate quantiles(percentiles) (10s)
medv.tenth <- quantile(medv, c(0.1))
medv.tenth
#do same with bootstrap
boot.fn = function(data, index) return(quantile(data[index], c(0.1)))
boot(medv, boot.fn, 1000)#the same with Small standard error relative to tenth-percentile value.

