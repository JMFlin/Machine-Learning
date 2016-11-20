library(ISLR)
library(MASS)#functions
library(splines)
library(gam)
library(akima)
library(boot)
library(leaps)

###Chapter 7: Moving Beyond Linearity

#------------------------------------Poly, Splines and Local Regression
fit <- lm(wage~poly(age,4), data = Wage)
coef(summary(fit))

#The function returns a matrix whose columns are a basis of orthogonal
#polynomials, which essentially means that each column is a linear orthogonal
#combination of the variables age, age^2, age^3 and age^4.

#However, we can also use poly() to obtain age, age^2, age^3 and age^4
#directly, if we prefer. We can do this by using the raw=TRUE argument to
#the poly() function. Later we see that this does not affect the model in a
#meaningful way-though the choice of basis clearly affects the coefficient
#estimates, it does not affect the fitted values obtained.

fit2 <- lm(wage~poly(age,4, raw = T),data = Wage)
coef(summary(fit2))

fit2a <- lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
coef(fit2a)

#We now create a grid of values for age at which we want predictions, and
#then call the generic predict() function, specifying that we want standard
#errors as well.

agelims <- range(Wage$age)
age.grid <- seq(from=agelims[1], to=agelims[2])
preds <- predict(fit, newdata =list(age=age.grid), se=TRUE)
se.bands <- cbind(preds$fit +2* preds$se.fit ,preds$fit -2*preds$se.fit)

par(mfrow=c(1,2),mar=c(4.5,4.5,1,1) ,oma=c(0,0,4,0))
plot(Wage$age, Wage$wage ,xlim=agelims ,cex =.5,col="darkgrey")
title("Degree 4 Polynomial",outer=T)
lines(age.grid ,preds$fit ,lwd=2,col="blue")
matlines(age.grid ,se.bands ,lwd=1, col="blue",lty=3)

#Here the mar and oma arguments to par() allow us to control the margins
#of the plot, and the title() function creates a figure title that spans both subplots.

#We mentioned earlier that whether or not an orthogonal set of basis functions
#is produced in the poly() function will not affect the model obtained
#in a meaningful way. What do we mean by this? The fitted values obtained
#in either case are identical:
preds2 <- predict(fit2 ,newdata =list(age=age.grid),se=TRUE)
max(abs(preds$fit -preds2$fit ))

#In performing a polynomial regression we must decide on the degree of
#the polynomial to use. One way to do this is by using hypothesis tests. We
#now fit models ranging from linear to a degree-5 polynomial and seek to
#determine the simplest model which is sufficient to explain the relationship
#between wage and age. We use the anova() function, which performs an
#analysis of variance (ANOVA, using an F-test) in order to test the null
#hypothesis that a model M1 is sufficient to explain the data against the
#alternative hypothesis that a more complex model M2 is required. In order
#to use the anova() function, M1 and M2 must be nested models: the
#predictors in M1 must be a subset of the predictors in M2. In this case,
#we fit five different models and sequentially compare the simpler model to
#the more complex model.

fit.1 <- lm(wage~age ,data=Wage)
fit.2 <- lm(wage~poly(age ,2),data=Wage)
fit.3 <- lm(wage~poly(age ,3),data=Wage)
fit.4 <- lm(wage~poly(age ,4),data=Wage)
fit.5 <- lm(wage~poly(age ,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)

#The p-value comparing the linear Model 1 to the quadratic Model 2 is
#essentially zero (<10???15), indicating that a linear fit is not sufficient. Similarly
#the p-value comparing the quadratic Model 2 to the cubic Model 3
#is very low (0.0017), so the quadratic fit is also insufficient. The p-value
#comparing the cubic and degree-4 polynomials, Model 3 and Model 4, is approximately
#5 % while the degree-5 polynomial Model 5 seems unnecessary
#because its p-value is 0.37. Hence, either a cubic or a quartic polynomial
#appear to provide a reasonable fit to the data, but lower- or higher-order
#models are not justified.

#In this case, instead of using the anova() function, we could have obtained
#these p-values more succinctly by exploiting the fact that poly() creates
#orthogonal polynomials.

coef(summary (fit.5))

#Notice that the p-values are the same, and in fact the square of the
#t-statistics are equal to the F-statistics from the anova() function

#As an alternative to using hypothesis tests and ANOVA, we could choose
#the polynomial degree using cross-validation, as discussed in Chapter 5.

#Next we consider the task of predicting whether an individual earns more
#than $250,000 per year. We proceed much as before, except that first we
#create the appropriate response vector, and then apply the glm() function
#using family="binomial" in order to fit a polynomial logistic regression model.

fit <- glm(I(wage >250)~poly(age ,4), data=Wage, family=binomial)
preds <- predict(fit ,newdata =list(age=age.grid),se=T)


#However, calculating the confidence intervals is slightly more involved than
#in the linear regression case. The default prediction type for a glm() model
#is type="link", which is what we use here. This means we get predictions
#for the logit

pfit <- exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit <- cbind(preds$fit +2* preds$se.fit , preds$fit -2*
                           preds$se.fit)
se.bands <- exp(se.bands.logit)/(1+exp(se.bands.logit))

plot(Wage$age ,I(Wage$wage >250),xlim=agelims ,type="n",ylim=c(0,.2))
points(jitter(Wage$age), I((Wage$wage >250)/5),cex=.5,pch ="|",col="darkgrey")
lines(age.grid ,pfit ,lwd=2, col ="blue")
matlines (age.grid ,se.bands ,lwd=1, col=" blue",lty=3)

#We have drawn the age values corresponding to the observations with wage
#values above 250 as gray marks on the top of the plot, and those with wage
#values below 250 are shown as gray marks on the bottom of the plot. We
#used the jitter() function to jitter the age values a bit so that observations
#jitter() with the same age value do not cover each other up.


#In order to fit a step function, as discussed in Section 7.2, we use the cut() function.

table(cut(Wage$age ,4))
fit <- lm(wage~cut(age ,4),data=Wage)
coef(summary (fit))

#Here cut() automatically picked the cutpoints at 33.5, 49, and 64.5 years
#of age. We could also have specified our own cutpoints directly using the
#breaks option. The function cut() returns an ordered categorical variable;
#the lm() function then creates a set of dummy variables for use in the regression.
#The age<33.5 category is left out, so the intercept coefficient of
#$94,160 can be interpreted as the average salary for those under 33.5 years
#of age, and the other coefficients can be interpreted as the average additional
#salary for those in the other age groups. We can produce predictions
#and plots just as we did in the case of the polynomial fit.

#----------------------------SPLINES

#Fitting wage to age using a regression spline is simple:
par(mfrow = c(1,1))
fit <- lm(wage~bs(age, knots=c(25,40,60)), data=Wage)
pred <- predict (fit, newdata =list(age=age.grid),se=T)
plot(Wage$age ,Wage$wage ,col="gray")
lines(age.grid ,pred$fit ,lwd=2)
lines(age.grid ,pred$fit+2*pred$se ,lty="dashed")
lines(age.grid ,pred$fit-2*pred$se ,lty="dashed")
abline(v=c(25,40,60), lty = 2, col = "darkgreen")#Splin places
#You can also use CV to chose splines by making multiple models and comparing.
#Here we have prespecified knots at ages 25, 40, and 60. This produces a
#spline with six basis functions.

#We could also use the df option to
#produce a spline with knots at uniform quantiles of the data.

dim(bs(Wage$age, knots=c(25,40,60)))
dim(bs(Wage$age, df=6))
attr(bs(Wage$age, df=6), "knots")

#In this case R chooses knots at ages 33.8, 42.0, and 51.0, which correspond
#to the 25th, 50th, and 75th percentiles of age. The function bs() also has
#a degree argument, so we can fit splines of any degree, rather than the
#default degree of 3 (which yields a cubic spline).

#In order to instead fit a natural spline, we use the ns() function. 
#Here ns() we fit a natural spline with four degrees of freedom.

fit2 <- lm(wage~ns(age ,df=4),data=Wage)
pred2 <- predict(fit2 ,newdata=list(age=age.grid),se=T)
lines(age.grid, pred2$fit, col="red", lwd=2)

#In order to fit a smoothing spline, we use the smooth.spline() function.

plot(Wage$age, Wage$wage, xlim=agelims, cex =.5, col="darkgrey")
title("Smoothing Spline ")
fit <- smooth.spline(Wage$age, Wage$wage, df=16)
fit2 <- smooth.spline(Wage$age, Wage$wage, cv=TRUE)
fit2$df
lines(fit, col="red", lwd=2)
lines(fit2 ,col="blue",lwd=2)
legend("topright",legend=c("16 DF" ,"6.8 DF"),
          col=c("red","blue"),lty=1,lwd=2, cex =.8)

#Notice that in the first call to smooth.spline(), we specified df=16. The
#function then determines which value of ?? leads to 16 degrees of freedom. In
#the second call to smooth.spline(), we select the smoothness level by crossvalidation;
#this results in a value of ?? that yields 6.8 degrees of freedom.

#In order to perform local regression, we use the loess() function.

par(mfrow = c(1,1))
plot(Wage$age, Wage$wage ,xlim=agelims ,cex =.5,col="darkgrey")
title("Local Regression")
fit <- loess(wage~age ,span=.2,data=Wage)
fit2 <- loess(wage~age ,span=.5,data=Wage)
lines(age.grid ,predict (fit ,data.frame(age=age.grid)),
        col="red",lwd=2)
lines(age.grid ,predict (fit2 ,data.frame(age=age.grid)),
        col="blue",lwd=2)
legend ("topright",legend=c("Span=0.2","Span=0.5"),
          col=c("red","blue"),lty=1,lwd=2, cex =.8)

#Here we have performed local linear regression using spans of 0.2 and 0.5:
#that is, each neighborhood consists of 20 % or 50 % of the observations. The
#larger the span, the smoother the fit. The locfit library can also be used
#for fitting local regression models in R. You can use CV to chose best fit.

#----------------------GAMs

gam1 <- lm(wage~ns(year ,4)+ns(age ,5)+education ,data=Wage)

#We now fit a GAM to predict wage using natural spline functions of year
#and age, treating education as a qualitative predictor. Since
#this is just a big linear regression model using an appropriate choice of
#basis functions, we can simply do this using the lm() function.

#We now fit the model (7.16) using smoothing splines rather than natural
#splines. In order to fit more general sorts of GAMs, using smoothing splines
#or other components that cannot be expressed in terms of basis functions
#and then fit using least squares regression, we will need to use the gam
#library in R

#The s() function, which is part of the gam library, is used to indicate that we would like to use a smoothing spline. 
#We specify that the function of year should have 4 degrees of freedom, and that the function of age will
#have 5 degrees of freedom. Since education is qualitative, we leave it as is,
#and it is converted into four dummy variables. We use the gam() function in order to fit a GAM using these components. 
#All of the terms in (7.16) are fit simultaneously, taking each other into account to explain the response.

gam.m3 <- gam(wage~s(year ,4)+s(age ,5)+education ,data=Wage)
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE ,col =" blue")
plot.gam(gam1 , se=TRUE , col="red")

#In these plots, the function of year looks rather linear. We can perform a
#series of ANOVA tests in order to determine which of these three models is
#best: a GAM that excludes year (M1), a GAM that uses a linear function
#of year (M2), or a GAM that uses a spline function of year (M3).

gam.m1 <- gam(wage~s(age ,5)+education ,data=Wage)
gam.m2 <- gam(wage~year+s(age ,5)+education ,data=Wage)
anova(gam.m1,gam.m2,gam.m3,test="F")

#We find that there is compelling evidence that a GAM with a linear function
#of year is better than a GAM that does not include year at all
#(p-value = 0.00014). However, there is no evidence that a non-linear function
#of year is needed (p-value = 0.349). In other words, based on the results
#of this ANOVA, M2 is preferred.

summary(gam.m3)

#The p-values for year and age correspond to a null hypothesis of a linear
#relationship versus the alternative of a non-linear relationship. The large
#p-value for year reinforces our conclusion from the ANOVA test that a linear
#function is adequate for this term. However, there is very clear evidence
#that a non-linear term is required for age

preds <- predict(gam.m2,newdata =Wage)

#We can also use local regression fits as building blocks in a GAM, using the lo() function.

gam.lo <- gam(wage~s(year ,df=4)+lo(age ,span =0.7)+education, data=Wage)
plot.gam(gam.lo, se=TRUE , col =" green")

#Here we have used local regression for the age term, with a span of 0.7.
#We can also use the lo() function to create interactions before calling the
#gam() function. For example,
gam.lo.i <- gam(wage~lo(year ,age , span=0.5)+education, data=Wage)
plot(gam.lo.i)
#fits a two-term model, in which the first term is an interaction between
#year and age, fit by a local regression surface. We can plot the resulting
#two-dimensional surface

#In order to fit a logistic regression GAM, we once again use the I() function
#in constructing the binary response variable, and set family=binomial.

gam.lr <- gam(I(wage >250)~year+s(age ,df=5)+education ,
           family=binomial ,data=Wage)
par(mfrow=c(1,3))
plot(gam.lr,se=T,col="green ")

#It is easy to see that there are no high earners in the <HS category:

table(Wage$education ,I(Wage$wage >250))

#Hence, we fit a logistic regression GAM using all but this category. This
#provides more sensible results.

gam.lr.s <- gam(I(wage >250)~year+s(age ,df=5)+education ,family=binomial, 
             data=Wage, subset =(education !="1. < HS Grad"))
plot(gam.lr.s,se=T,col="green")

#-----------------------------------APPLIED

#Keep an array of all cross-validation errors. We are performing K-fold cross validation with K=10.

#Perform polynomial regression to predict wage using age. Use
#cross-validation to select the optimal degree d for the polynomial.
#What degree was chosen, and how does this compare to
#the results of hypothesis testing using ANOVA? Make a plot of
#the resulting polynomial fit to the data.

all.deltas <- rep(NA, 10)
for (i in 1:10) {
  glm.fit <- glm(wage~poly(age, i), data = Wage)
  all.deltas[i] = cv.glm(Wage, glm.fit, K=10)$delta[2]
}
plot(1:10, all.deltas, xlab="Degree", ylab="CV error", type="l", pch=20, lwd=2, ylim=c(1590, 1700))
min.point = min(all.deltas)
sd.points = sd(all.deltas)
abline(h=min.point + 0.2 * sd.points, col="red", lty="dashed")
abline(h=min.point - 0.2 * sd.points, col="red", lty="dashed")
legend("topright", "0.2-standard deviation lines", lty="dashed", col="red")
which.min(all.deltas) # the min is poly 9 but poly 4 looks just as good in picture.

#We now find best degree using Anova.

fit.1 <- lm(wage~poly(age, 1), data=Wage)
fit.2 <- lm(wage~poly(age, 2), data=Wage)
fit.3 <- lm(wage~poly(age, 3), data=Wage)
fit.4 <- lm(wage~poly(age, 4), data=Wage)
fit.5 <- lm(wage~poly(age, 5), data=Wage)
fit.6 <- lm(wage~poly(age, 6), data=Wage)
fit.7 <- lm(wage~poly(age, 7), data=Wage)
fit.8 <- lm(wage~poly(age, 8), data=Wage)
fit.9 <- lm(wage~poly(age, 9), data=Wage)
fit.10 <- lm(wage~poly(age, 10), data=Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5, fit.6, fit.7, fit.8, fit.9, fit.10)

#Anova shows that all polynomials above degree 4 are insignificant at 1% significance level (not 9).

#We now plot the polynomial prediction on the data

plot(wage~age, data=Wage, col="darkgrey")
agelims <- range(Wage$age)
age.grid <- seq(from=agelims[1], to=agelims[2])
lm.fit <- lm(wage~poly(age, which.min(all.deltas)), data=Wage) # You could just manually put the poly number which you
#thought was best via picture. a 9 poly might be a bit too much
lm.pred <- predict(lm.fit, data.frame(age=age.grid))
lines(age.grid, lm.pred, col="blue", lwd=2)

#Fit a step function to predict wage using age, and perform crossvalidation
#to choose the optimal number of cuts. Make a plot of
#the fit obtained.

all.cvs = rep(NA, 10)
for (i in 2:10){
  Wage$age.cut = cut(Wage$age, i)
  lm.fit = glm(wage~age.cut, data=Wage)
  all.cvs[i] = cv.glm(Wage, lm.fit, K=10)$delta[2]
}
plot(2:10, all.cvs[-1], xlab="Number of cuts", ylab="CV error", type="l", pch=20, lwd=2)

#The cross validation shows that test error is minimum for k=8 cuts.
#We now train the entire data with step function using 8 cuts and plot it.

lm.fit <- glm(wage~cut(age, 8), data=Wage)
agelims <- range(Wage$age)
age.grid <- seq(from=agelims[1], to=agelims[2])
lm.pred <- predict(lm.fit, data.frame(age=age.grid))
plot(wage~age, data=Wage, col="darkgrey")
lines(age.grid, lm.pred, col="red", lwd=2)

#--------------------------------------------

#Explore the relationships between some of these other
#predictors and wage, and use non-linear fitting techniques in order to
#fit flexible models to the data. Create plots of the results obtained,
#and write a summary of your findings.

par(mfrow = c(1, 2))
plot(Wage$maritl, Wage$wage)
plot(Wage$jobclass, Wage$wage)

#It appears a married couple makes more money on average than other groups. 
#It also appears that Informational jobs are higher-wage than Industrial jobs on average.

fit <- lm(wage ~ maritl, data = Wage)
deviance(fit)
fit = lm(wage ~ jobclass, data = Wage)
deviance(fit)
fit = lm(wage ~ maritl + jobclass, data = Wage)
deviance(fit)
fit = gam(wage ~ maritl + jobclass + s(age, 4), data = Wage)
deviance(fit)

#Without more advanced techniques, we cannot fit splines to categorical variables (factors). 
#maritl and jobclass do add statistically significant improvements to the previously discussed models.

#--------------------------------------

#Fit some of the non-linear models investigated in this chapter to the
#Auto data set. Is there evidence for non-linear relationships in this
#data set? Create some informative plots to justify your answer.

#mpg appears inversely proportional to cylinders, displacement, horsepower, weight.
table(mtcars$mpg)

#Poly

rss <- rep(NA, 10)
fits <- list()
for (d in 1:10) {
  fits[[d]] <- lm(mpg ~ poly(displacement, d), data = Auto)
  rss[d] <- deviance(fits[[d]])
}
rss
anova(fits[[1]], fits[[2]], fits[[3]], fits[[4]])#Training RSS decreases over time. Quadratic polynomic sufficient from ANOVA-perspective.

cv.errs <- rep(NA, 15)
for (d in 1:15) {
  fit <- glm(mpg ~ poly(displacement, d), data = Auto)
  cv.errs[d] <- cv.glm(Auto, fit, K = 10)$delta[2]
}
cv.errs
which.min(cv.errs)#CV chose 10th degree polynomial

#Step Function

cv.errs <- rep(NA, 10)
for (c in 2:10) {
  Auto$dis.cut <- cut(Auto$displacement, c)
  fit <- glm(mpg ~ dis.cut, data = Auto)
  cv.errs[c] <- cv.glm(Auto, fit, K = 10)$delta[2]
}
cv.errs
which.min(cv.errs)

#Splines

cv.errs <- rep(NA,10)
for (df in 3:10) {
  fit <- glm(mpg~ns(displacement, df=df), data=Auto)
  cv.errs[df] = cv.glm(Auto, fit, K=10)$delta[2]
}
which.min(cv.errs)
cv.errs

#GAMS

fit <- gam(mpg~s(displacement, 4) + s(horsepower, 4), data=Auto)
summary(fit)

#------------------------------

#This question uses the variables dis (the weighted mean of distances
#to five Boston employment centers) and nox (nitrogen oxides concentration
#in parts per 10 million) from the Boston data. We will treat
#dis as the predictor and nox as the response.

#Use the poly() function to fit a cubic polynomial regression to
#predict nox using dis. Report the regression output, and plot
#the resulting data and polynomial fits.

lm.fit <- lm(nox ~ poly(dis, 3), data = Boston)
summary(lm.fit)

dislim <- range(Boston$dis)
dis.grid <- seq(from = dislim[1], to = dislim[2], by = 0.1)
lm.pred <- predict(lm.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "darkgrey")
lines(dis.grid, lm.pred, col = "red", lwd = 2)
#Summary shows that all polynomial terms are significant while predicting nox using dis. 
#Plot shows a smooth curve fitting the data fairly well.


#Plot the polynomial fits for a range of different polynomial
#degrees (say, from 1 to 10), and report the associated residual sum of squares.

#We plot polynomials of degrees 1 to 10 and save train RSS.

all.rss <- rep(NA, 10)
for (i in 1:10) {
  lm.fit <- lm(nox ~ poly(dis, i), data = Boston)
  all.rss[i] <- sum(lm.fit$residuals^2)
}
all.rss

#As expected, train RSS monotonically decreases with degree of polynomial.

#Perform cross-validation or another approach to select the optimal
#degree for the polynomial, and explain your results.

#We use a 10-fold cross validation to pick the best polynomial degree.
all.deltas <- rep(NA, 10)
for (i in 1:10) {
  glm.fit <- glm(nox ~ poly(dis, i), data = Boston)
  all.deltas[i] <- cv.glm(Boston, glm.fit, K = 10)$delta[2]
}
plot(1:10, all.deltas, xlab = "Degree", ylab = "CV error", type = "l", pch = 20, lwd = 2)
which.min(all.deltas)

#A 10-fold CV shows that the CV error reduces as we increase degree from 1 to 3, 
#stay almost constant till degree 5, and the starts increasing for higher degrees. 
#We pick 4 as the best polynomial degree.

#Use the bs() function to fit a regression spline to predict nox
#using dis. Report the output for the fit using four degrees of
#freedom. How did you choose the knots? Plot the resulting fit.

#We see that dis has limits of about 1 and 13 respectively. 
#We split this range in roughly equal 4 intervals and establish knots at [4, 7, 11]. 
#Note: bs function in R expects either df or knots argument. If both are specified, knots are ignored.

sp.fit <- lm(nox ~ bs(dis, df = 4, knots = c(4, 7, 11)), data = Boston)
summary(sp.fit)

sp.pred <- predict(sp.fit, list(dis = dis.grid))
plot(nox ~ dis, data = Boston, col = "darkgrey")
lines(dis.grid, sp.pred, col = "red", lwd = 2)

#The summary shows that all terms in spline fit are significant. 
#Plot shows that the spline fits data well except at the extreme values of dis, (especially dis > 10$).

#Now fit a regression spline for a range of degrees of freedom, and
#plot the resulting fits and report the resulting RSS. Describe the
#results obtained.

#We fit regression splines with dfs between 3 and 16.

all.cv <- rep(NA, 16)
for (i in 3:16) {
  lm.fit <- lm(nox ~ bs(dis, df = i), data = Boston)
  all.cv[i] <- sum(lm.fit$residuals^2)
}
all.cv[-c(1, 2)] #Train RSS monotonically decreases till df=14 and then slightly increases for df=15 and df=16.

#Perform cross-validation or another approach in order to select
#the best degrees of freedom for a regression spline on this data.
#Describe your results.

#Finally, we use a 10-fold cross validation to find best df. We try all integer values of df between 3 and 16.

all.cv <- rep(NA, 16)
for (i in 3:16) {
  lm.fit <- glm(nox ~ bs(dis, df = i), data = Boston)
  all.cv[i] <- cv.glm(Boston, lm.fit, K = 10)$delta[2]
}
plot(3:16, all.cv[-c(1, 2)], lwd = 2, type = "l", xlab = "df", ylab = "CV error")
#CV error is more jumpy in this case, but attains minimum at df=10. We pick 10 as the optimal degrees of freedom.
which.min(all.cv)

#----------------------------------------

#Split the data into a training set and a test set. Using out-of-state
#tuition as the response and the other variables as the predictors,
#perform forward stepwise selection on the training set in order
#to identify a satisfactory model that uses just a subset of the predictors.

train <- sample(length(College$Outstate), length(College$Outstate)/2)
test <- -train
College.train <- College[train, ]
College.test <- College[test, ]
reg.fit <- regsubsets(Outstate ~ ., data = College.train, nvmax = 17, method = "forward")
reg.summary <- summary(reg.fit)
par(mfrow = c(1, 3))
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
min.cp <- min(reg.summary$cp)
std.cp <- sd(reg.summary$cp)
abline(h = min.cp + 0.2 * std.cp, col = "red", lty = 2)
abline(h = min.cp - 0.2 * std.cp, col = "red", lty = 2)
plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
min.bic <- min(reg.summary$bic)
std.bic <- sd(reg.summary$bic)
abline(h = min.bic + 0.2 * std.bic, col = "red", lty = 2)
abline(h = min.bic - 0.2 * std.bic, col = "red", lty = 2)
plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R2", 
     type = "l", ylim = c(0.4, 0.84))
max.adjr2 <- max(reg.summary$adjr2)
std.adjr2 <- sd(reg.summary$adjr2)
abline(h = max.adjr2 + 0.2 * std.adjr2, col = "red", lty = 2)
abline(h = max.adjr2 - 0.2 * std.adjr2, col = "red", lty = 2)

#All cp, BIC and adjr2 scores show that size 6 is the minimum size for the subset for which the scores are withing 0.2 standard deviations of optimum. 
#We pick 6 as the best subset size and find best 6 variables using entire data.

reg.fit <- regsubsets(Outstate ~ ., data = College, method = "forward")
coefi <- coef(reg.fit, id = 6)
names(coefi)

#Fit a GAM on the training data, using out-of-state tuition as
#the response and the features selected in the previous step as
#the predictors. Plot the results, and explain your findings.

gam.fit <- gam(Outstate ~ Private + s(Room.Board, df = 2) + s(PhD, df = 2) + 
                s(perc.alumni, df = 2) + s(Expend, df = 5) + s(Grad.Rate, df = 2), data = College.train)
par(mfrow = c(2, 3))
plot(gam.fit, se = T, col = "blue")

#Evaluate the model obtained on the test set, and explain the
#results obtained.

gam.pred <- predict(gam.fit, College.test)
gam.err <- mean((College.test$Outstate - gam.pred)^2)
gam.err

gam.tss <- mean((College.test$Outstate - mean(College.test$Outstate))^2)
test.rss <-  1 - gam.err/gam.tss
test.rss
#We obtain a test R-squared of 0.77 using GAM with 6 predictors. 
#This is a slight improvement over a test RSS of 0.74 obtained using OLS.

#For which variables, if any, is there evidence of a non-linear
#relationship with the response?
summary(gam.fit)

#Non-parametric Anova test shows a strong evidence of non-linear relationship between response and Expend, 
#and a moderately strong non-linear relationship (using p value of 0.05) between response and Grad.Rate or PhD.

#-------------------------------------------------

#In Section 7.7, it was mentioned that GAMs are generally fit using
#a backfitting approach. The idea behind backfitting is actually quite
#simple. We will now explore backfitting in the context of multiple
#linear regression.
#Suppose that we would like to perform multiple linear regression, but
#we do not have software to do so. Instead, we only have software
#to perform simple linear regression. Therefore, we take the following
#iterative approach: we repeatedly hold all but one coefficient estimate
#fixed at its current value, and update only that coefficient
#estimate using a simple linear regression. The process is continued until
#convergence-that is, until the coefficient estimates stop changing.

#We create variables according to the equation Y = -2.1 + 1.3X + 0.54X^2.
X1 <- rnorm(100)
X2 <- rnorm(100)
eps <- rnorm(100, sd=0.1)
Y <- -2.1 + 1.3 * X1 + 0.54 * X2 + eps

#Create a list of 1000 hat{beta}_0, hat{beta}_1 and hat{beta}_2. Initialize first of the hat{\beta}_1 to 10.

beta0 <- rep(NA, 1000)
beta1 <- rep(NA, 1000)
beta2 <- rep(NA, 1000)
beta1[1] <- 10

#Accumulate results of 1000 iterations in the beta arrays.

for (i in 1:1000) {
  a <- Y - beta1[i] * X1
  beta2[i] <- lm(a~X2)$coef[2]
  a <- Y - beta2[i] * X2
  lm.fit <- lm(a~X1)
  if (i < 1000) {
    beta1[i+1] <- lm.fit$coef[2]
  }
  beta0[i] <- lm.fit$coef[1]
}
par(mfrow = c(1,1))
plot(1:1000, beta0, type="l", xlab="iteration", ylab="betas", ylim=c(-2.2, 1.6), col="green")
lines(1:1000, beta1, col="red")
lines(1:1000, beta2, col="blue")
legend('center', c("beta0","beta1","beta2"), lty=1, col=c("green","red","blue"))
#The coefficients quickly attain their least square values.

it <- lm(Y~X1+X2)
plot(1:1000, beta0, type="l", xlab="iteration", ylab="betas", ylim=c(-2.2, 1.6), col="green")
lines(1:1000, beta1, col="red")
lines(1:1000, beta2, col="blue")
abline(h=lm.fit$coef[1], lty="dashed", lwd=3, col=rgb(0, 0, 0, alpha=0.4))
abline(h=lm.fit$coef[2], lty="dashed", lwd=3, col=rgb(0, 0, 0, alpha=0.4))
abline(h=lm.fit$coef[3], lty="dashed", lwd=3, col=rgb(0, 0, 0, alpha=0.4))
legend('center', c("beta0","beta1","beta2", "multiple regression"), lty=c(1, 1, 1, 2), col=c("green","red","blue", "black"))
#Dotted lines show that the estimated multiple regression coefficients match exactly with the coefficients obtained using backfitting.

#When the relationship between Y and X's is linear, 
#one iteration is sufficient to attain a good approximation of true regression coefficients.

#----------------------------------------

#This problem is a continuation of the previous exercise. In a toy
#example with p = 100, show that one can approximate the multiple
#linear regression coefficient estimates by repeatedly performing simple
#linear regression in a backfitting procedure. How many backfitting
#iterations are required in order to obtain a "good" approximation to
#the multiple regression coefficient estimates? Create a plot to justify
#your answer.

p <- 100
n <- 1000
x <- matrix(ncol = p, nrow = n)
coefi <- rep(NA, p)
for (i in 1:p) {
  x[, i] <- rnorm(n)
  coefi[i] <- rnorm(1) * 100
}
y <- x %*% coefi + rnorm(n)
beta <- rep(0, p)
max_iterations <- 1000
errors <- rep(NA, max_iterations + 1)
iter <- 2
errors[1] <- Inf
errors[2] <- sum((y - x %*% beta)^2)
threshold <- 1e-04
while (iter < max_iterations && errors[iter - 1] - errors[iter] > threshold) {
  for (i in 1:p) {
    a <- y - x %*% beta + beta[i] * x[, i]
    beta[i] <- lm(a ~ x[, i])$coef[2]
  }
  iter <- iter + 1
  errors[iter] <- sum((y - x %*% beta)^2)
  print(c(iter - 2, errors[iter - 1], errors[iter]))
}

#10 iterations to get to a "good" approximation defined by the threshold on sum of squared errors between subsequent iterations. The error increases on the 11th iteration.

plot(1:11, errors[3:13], type = "b")
