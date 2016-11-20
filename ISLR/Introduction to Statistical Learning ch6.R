library(ISLR)
library(MASS)#functions
library(class)#functions
library(leaps)#functions for subset selection etc.
library(glmnet)#Ride and Lasso
library(pls) #for pcr and pls

###Chapter 6: Linear Model Selection and Regularization

#Cross-Validation is the best validation method, R2, BIC, AIC can't be calculated for all models.
#With financial time series you could chose the trainig and testing at random, not <2010 for train
#and >2010 for test?

#-------------BEST SUBSET REGRESSION

fix(Hitters)
sum(is.na(Hitters$Salary))

Hitters <- na.omit(Hitters)

#The regsubsets() function performs best sub- regsubsets() set selection by identifying the best model that contains a given number
#of predictors, where best is quantified using RSS.

regfit.full <- regsubsets(Salary~., data = Hitters)
summary(regfit.full)

#An asterisk indicates that a given variable is included in the corresponding
#model. For instance, this output indicates that the best two-variable model
#contains only Hits and CRBI. By default, regsubsets() only reports results
#up to the best eight-variable model. But the nvmax option can be used
#in order to return as many variables as are desired. Here we fit up to a
#19-variable model.

regfit.full <- regsubsets(Salary~.,data=Hitters ,nvmax=ncol(Hitters)-1)
reg.summary <- summary(regfit.full)
names(reg.summary)

#The summary() function also returns R2, RSS, adjusted R2, Cp, and BIC.
#We can examine these to try to select the best overall model.

reg.summary$rsq
#As expected, the R^2 statistic increases monotonically as more
#variables are included.

#Plotting RSS, adjusted R2, Cp, and BIC for all of the models at once will
#help us decide which model to select.

par(mfrow = c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables",ylab="RSS", type="b")
plot(reg.summary$adjr2 ,xlab="Number of Variables",
     ylab="Adjusted RSq",type="b")

#The points() command works like the plot() command, except that it points() puts points on a plot that has already been created, instead of creating a
#new plot. The which.max() function can be used to identify the location of
#the maximum point of a vector. We will now plot a red dot to indicate the
#model with the largest adjusted R2 statistic.
which.max(reg.summary$adjr2)
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col="red",cex=2,pch =20)

#In a similar fashion we can plot the Cp and BIC statistics, and indicate the
#models with the smallest statistic using which.min().

plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp",type="b")
which.min(reg.summary$cp)
points (which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col ="red",cex=2,pch =20)
#Cp is an estimate of prediciotn error

plot(reg.summary$bic, xlab="Number of Variables ", ylab="BIC", type="b")
which.min(reg.summary$bic)
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)],col="red",cex=2,pch =20)

#The regsubsets() function has a built-in plot() command which can
#be used to display the selected variables for the best model with a given
#number of predictors, ranked according to the BIC, Cp, adjusted R2, or
#AIC.
?plot.regsubsets
plot(regfit.full, scale="r2")
plot(regfit.full, scale="adjr2")
plot(regfit.full, scale="Cp")
plot(regfit.full, scale="bic")

#The top row of each plot contains a black square for each variable selected
#according to the optimal model associated with that statistic. For instance,
#we see that several models share a BIC close to -150.

#To see the coefficiants for this model:
names(Hitters)
coef(regfit.full, which.min(reg.summary$bic))

#------------------------Forward and Backward Stepwise Selection

regfit.fwd <- regsubsets(Salary~.,data=Hitters, nvmax=(ncol(Hitters)-1), method ="forward")
summary (regfit.fwd)
regfit.bwd <- regsubsets(Salary~., data = Hitters, nvmax = (ncol(Hitters)-1), method = "backward")
summary(regfit.bwd)

#plots for forward step wise
plot(regfit.fwd, scale="r2")
plot(regfit.fwd, scale="adjr2")
plot(regfit.fwd, scale="Cp")
plot(regfit.fwd, scale="bic")

#plots for backwards step wise
plot(regfit.bwd, scale="r2")
plot(regfit.bwd, scale="adjr2")
plot(regfit.bwd, scale="Cp")
plot(regfit.bwd, scale="bic")

#For instance, we see that using forward stepwise selection, the best one variable
#model contains only CRBI, and the best two-variable model additionally
#includes Hits. For this data, the best one-variable through sixvariable
#models are each identical for best subset and forward selection.
#However, the best seven-variable models identified by forward stepwise selection,
#backward stepwise selection, and best subset selection are different.

coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)

#----------------------Choosing Among Models Using the Validation Set Approach and Cross-Validation

#We just saw that it is possible to choose among a set of models of different
#sizes using Cp, BIC, and adjusted R2. We will now consider how to do this
#using the validation set and cross-validation approaches.

#In order for these approaches to yield accurate estimates of the test
#error, we must use only the training observations to perform all aspects of
#model-fitting-including variable selection. Therefore, the determination of
#which model of a given size is best must be made using only the training
#observations.

#In order to use the *validation* set approach, we begin by splitting the
#observations into a training set and a test set. We do this by creating
#a random vector, train, of elements equal to TRUE if the corresponding
#observation is in the training set, and FALSE otherwise. The vector test has
#a TRUE if the observation is in the test set, and a FALSE otherwise.
set.seed(1)
train <- sample(c(TRUE ,FALSE), nrow(Hitters),rep=TRUE)
test <- !train
regfit.best <- regsubsets(Salary~., data=Hitters[train, ], nvmax=ncol(Hitters)-1)#You can add method = "forward"

#We now compute the validation set error for the best
#model of each model size. We first make a model matrix from the test data.

test.mat <- model.matrix(Salary~., data = Hitters[test,])

#Now we run a loop, and for each size i, we
#extract the coefficients from regfit.best for the best model of that size,
#multiply them into the appropriate columns of the test model matrix to
#form the predictions, and compute the test MSE.

val.errors <- rep(NA ,ncol(Hitters)-1)

for(i in 1:(ncol(Hitters)-1)){
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[,names(coefi)]%*%coefi
  val.errors[i] <- mean((Hitters$Salary[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(regfit.best, which.min(val.errors))

#This was a little tedious, partly because there is no predict() method
#for regsubsets(). Since we will be using this function again, we can capture
#our steps above and write our own predict method.

predict.regsubsets <- function (object , newdata ,id ,...){
  form <- as.formula (object$call[[2]])
  mat <- model.matrix(form ,newdata )
  coefi <- coef(object ,id=id)
  xvars <- names(coefi)
  mat[,xvars]%*%coefi 
}

#Finally, we perform best subset selection on the full data set, and select
#the best which.min(val.errors) model. It is important that we make use of the full
#data set in order to obtain more accurate coefficient estimates. Note that
#we perform best subset selection on the full data set and select the best which.min(val.errors)
#model, rather than simply using the variables that were obtained
#from the training set, because the best which.min(val.errors) model on the full data
#set may differ from the corresponding model on the training set.

regfit.best <- regsubsets(Salary~.,data=Hitters ,nvmax=ncol(Hitters)-1)
coef(regfit.best, which.min(val.errors))


#In fact, we see that the best which.min(val.errors) model on the full data set has a
#different set of variables than the best which.min(val.errors) model on the training set.

#We now try to choose among the models of different sizes using crossvalidation.
#This approach is somewhat involved, as we must perform best
#subset selection within each of the k training sets. First, we
#create a vector that allocates each observation to one of k = 10 folds, and
#we create a matrix in which we will store the results.

k <- 10
folds <- sample(1:k,nrow(Hitters),replace=TRUE)
cv.errors <- matrix(NA, k, ncol(Hitters)-1, dimnames =list(NULL, paste(1:(ncol(Hitters)-1))))#initialize a vector

#Now we write a for loop that performs cross-validation. In the jth fold, the
#elements of folds that equal j are in the test set, and the remainder are in
#the training set. We make our predictions for each model size (using our
#new predict() method), compute the test errors on the appropriate subset,
#and store them in the appropriate slot in the matrix cv.errors.

for(j in 1:k){
  best.fit <- regsubsets(Salary~.,data = Hitters[folds!=j,] ,nvmax=ncol(Hitters)-1)
  for(i in 1:(ncol(Hitters)-1)){
    pred <- predict(best.fit, Hitters[folds ==j,],id=i)
    cv.errors[j,i] <- mean((Hitters$Salary[folds==j]-pred)^2)
  }
}

#This has given us a 10×19 matrix, of which the (i, j)th element corresponds
#to the test MSE for the ith cross-validation fold for the best j-variable model. 
#We use the apply() function to average over the columns of this apply() matrix in order to obtain a vector for which the jth element is the crossvalidation
#error for the j-variable model.

mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow = c(1,1))
plot(mean.cv.errors ,type="b")#lowest point is best
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)],col="red",cex=2,pch =20)
which.min(mean.cv.errors)#is best

#We see that cross-validation selects an which.min(mean.cv.errors)-variable model. We now perform
#best subset selection on the full data set in order to obtain the which.min(mean.cv.errors)-variable
#model.

reg.best <- regsubsets(Salary~., data=Hitters, nvmax=ncol(Hitters)-1)
coef(reg.best, which.min(mean.cv.errors))

#-----------------------------------RIDE AND LASSO

#We will use the glmnet package in order to perform ridge regression and
#the lasso. The main function in this package is glmnet(), which can be used glmnet() to fit ridge regression models, lasso models, and more. This function has
#slightly different syntax from other model-fitting functions that we have
#encountered thus far in this book. In particular, we must pass in an x
#matrix as well as a y vector, and we do not use the y ~ x syntax. We will
#now perform ridge regression and the lasso in order to predict Salary on
#the Hitters data. Before proceeding ensure that the missing values have
#been removed from the data

x <- model.matrix(Salary~.,Hitters)[,-1]#removes intercept
y <- Hitters$Salary

#The model.matrix() function is particularly useful for creating x; not only
#does it produce a matrix corresponding to the (19) predictors but it also
#automatically transforms any qualitative variables into dummy variables.
#The latter property is important because glmnet() can only take numerical,
#quantitative inputs.

#RIDGE

#The glmnet() function has an alpha argument that determines what type
#of model is fit. If alpha=0 then a ridge regression model is fit, and if alpha=1
#then a lasso model is fit. We first fit a ridge regression model.

grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x,y,alpha = 0, lambda=grid)

#By default the glmnet() function performs ridge regression for an automatically
#selected range of lambda values. However, here we have chosen to implement
#the function over a grid of values ranging from lambda = 10^10 to lambda = 10^-2, essentially
#covering the full range of scenarios from the null model containing
#only the intercept, to the least squares fit. As we will see, we can also compute
#model fits for a particular value of lambda that is not one of the original
#grid values. Note that by default, the glmnet() function standardizes the
#variables so that they are on the same scale. To turn off this default setting,
#use the argument standardize=FALSE.

#Associated with each value of lambda is a vector of ridge regression coefficients,
#stored in a matrix that can be accessed by coef().

dim(coef(ridge.mod))# first is predictors + intercept and second is coulmns, one for each lamda.

#We expect the coefficient estimates to be much smaller, in terms of l2 norm,
#when a large value of lambda is used, as compared to when a small value of lambda is
#used. These are the coefficients when lambda = 11,498, along with their l2 norm:

ridge.mod$lambda[50]
coef(ridge.mod)[ ,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2) )

#In contrast, here are the coefficients when lambda = 705, along with their l2
#norm. Note the much larger l2 norm of the coefficients associated with this
#smaller value of lambda.

ridge.mod$lambda[60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

#We can use the predict() function for a number of purposes. For instance,
#we can obtain the ridge regression coefficients for a new value of lambda, say 50:

round(predict(ridge.mod, s=50, type="coefficients")[1:20,], 3)

#We now split the samples into a training set and a test set in order
#to estimate the test error of ridge regression and the lasso. There are two
#common ways to randomly split a data set. The first is to produce a random
#vector of TRUE, FALSE elements and select the observations corresponding to
#TRUE for the training data. The second is to randomly choose a subset of
#numbers between 1 and n; these can then be used as the indices for the
#training observations. The two approaches work equally well. We used the
#former method in Section 6.5.3. Here we demonstrate the latter approach

train <- sample(1: nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

#Next we fit a ridge regression model on the training set, and evaluate
#its MSE on the test set, using lambda = 4(this is an arbitrary chocie in this example, use CV in real life, see below for example).
#Note the use of the predict()
#function again. This time we get predictions for a test set, by replacing
#type="coefficients" with the newx argument.

ridge.mod <- glmnet(x[train,], y[train], alpha=0, lambda = grid, thresh = 1e-12)
par(mfrow = c(1,1))
plot(ridge.mod, "lambda")#left is OLS. Ride shrinks coef towards zero to right
ridge.pred <- predict(ridge.mod ,s=4, newx=x[test ,])
mean((ridge.pred -y.test)^2)

#The test MSE is mean((ridge.pred -y.test)^2). Note that if we had instead simply fit a model
#with just an intercept, we would have predicted each test observation using
#the mean of the training observations. In that case, we could compute the
#test set MSE like this:

mean((mean(y[train])-y.test)^2)

#We could also get the same result by fitting a ridge regression model with
#a very large value of lambda. Note that 1e10 means 10^10.

ridge.pred <- predict(ridge.mod ,s=1e10 ,newx=x[test ,])
mean((ridge.pred -y.test)^2)

#So fitting a ridge regression model with lambda = 4 leads to a much lower test
#MSE than fitting a model with just an intercept. We now check whether
#there is any benefit to performing ridge regression with lambda = 4 instead of
#just performing least squares regression. Recall that least squares is simply
#ridge regression with lambda = 0. (to do this do exact = TRUE)

ridge.pred <- predict(ridge.mod ,s=0, newx=x[test, ], exact=T)
mean((ridge.pred -y.test)^2)

#In general, instead of arbitrarily choosing lambda = 4, it would be better to
#use cross-validation to choose the tuning parameter lambda. We can do this using
#the built-in cross-validation function, cv.glmnet(). By default, the function cv.glmnet() performs ten-fold cross-validation, 
#though this can be changed using the argument nfolds

cv.out <- cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)#Cross-validated MSE
#First line from left is minimum and second line is 1 STD of the minimum. It is a restricted model that does almost as well.
#we have taken the intercept out
bestlam <- cv.out$lambda.min
bestlam

#Therefore, we see that the value of lambda that results in the smallest crossvalidation
#error is bestlam. What is the test MSE associated with this value of
#lambda?

ridge.pred <- predict(ridge.mod ,s=bestlam ,newx=x[test,])
mean((ridge.pred-y.test)^2)

#This represents a further improvement over the test MSE that we got using
#lambda = 4. Finally, we refit our ridge regression model on the full data set,
#using the value of lambda chosen by cross-validation, and examine the coefficient
#estimates.

out <- glmnet(x,y,alpha=0)
predict(out ,type="coefficients",s= bestlam)[1:20,]#1:ncol(data) redo this

#As expected, none of the coefficients are zero-ridge regression does not
#perform variable selection!

#LASSO

#We saw that ridge regression with a wise choice of lambda can outperform least
#squares as well as the null model on the Hitters data set. We now ask
#whether the lasso can yield either a more accurate or a more interpretable
#model than ridge regression. In order to fit a lasso model, we once again
#use the glmnet() function; however, this time we use the argument alpha=1.
#Other than that change, we proceed just as we did in fitting a ridge model.

lasso.mod <- glmnet(x[train,], y[train],alpha=1, lambda =grid)
par(mfrow = c(1,2))
plot(lasso.mod, "lambda")#Top is how many non zero variables are in your model. Shrinkage and model selection
plot(lasso.mod, "norm")
par(mfrow = c(1,1))
plot(lasso.mod, "dev")# xlab is R^2. You see that a lot of the var in the model is explained with quite a few
#zero coefficients! If at the end most of them jump then that is overfitting

#We can see from the coefficient plot that depending on the choice of tuning
#parameter, some of the coefficients will be exactly equal to zero. We now
#perform cross-validation and compute the associated test error.

cv.out <- cv.glmnet(x[train,],y[train],alpha=1)
par(mfrow = c(1,1))
plot(cv.out)#Cross-validated MSE
#First line from left is minimum (about size 16 for model see top) and second line is 1 STD of the minimum(1 std rule!). 
#It is a restricted model that does almost as well.
#we have taken the intercept out
coef(cv.out)#shows you the 1 std rule model!
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s=bestlam, newx=x[test,])
mean((lasso.pred-y.test)^2)

#This is substantially lower than the test set MSE of the null model and of
#least squares, and very similar to the test MSE of ridge regression with lambda
#chosen by cross-validation.

#However, the lasso has a substantial advantage over ridge regression in
#that the resulting coefficient estimates are sparse. Here we see that some of
#the 19 coefficient estimates are exactly zero. So the lasso model with lambda
#chosen by cross-validation contains less variables:

out <- glmnet(x,y,alpha=1, lambda=grid)
lasso.coef <- predict(out, type="coefficients", s= bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]#These are the non-zero coefficients to use in your model

#--------------------PCR and PLS

fix(Hitters)
Hitters <- na.omit(Hitters)
#Principal components regression
#We now apply PCR to the Hitters data, in order to predict Salary. 
pcr.fit <- pcr(Salary~.,data = Hitters, scale = TRUE, validation = "CV")
#Setting scale=TRUE has the effect of standardizing each predictor using the same method as earlier in ride and lasso
#prior to generating the principal components, so that the scale on which each variable is measured will not have an effect.
#Setting validation="CV" causes pcr() to compute the ten-fold cross-validation error
#for each possible value of M, the number of principal components used.
summary(pcr.fit) #the % variance exaplined is provided.
#M = 1 only captures 38.31 % of all the variance, or information, in the predictors.
#The CV score is provided for each possible number of components, ranging
#from M = 0 onwards. Note that pcr() reports the root mean squared error ; in order to obtain
#the usual MSE, we must square this quantity

#One can also plot the cross-validation scores
validationplot(pcr.fit, val.type = "MSEP")
#We see that the smallest cross-validation error occurs when M = 16 components
#are used. This is barely fewer than M = 19, which amounts to
#simply performing least squares, because when all of the components are
#used in PCR no dimension reduction occurs. However, from the plot we
#also see that the cross-validation error is roughly the same when only one
#component is included in the model. This suggests that a model that uses
#just a small number of components might suffice.

x <- model.matrix(Salary~.,Hitters )[,-1]
y <- Hitters$Salary

train <- sample(1: nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

#We now perform PCR on the training data and evaluate its test set
#performance.
set.seed(1)
pcr.fit <- pcr(Salary~., data=Hitters , subset=train ,scale=TRUE ,
            validation ="CV")
validationplot(pcr.fit, val.type="MSEP")#min looks to be about 7 here

#Now we find that the lowest cross-validation error occurs when M = 7
#component are used. We compute the test MSE as follows.

pcr.pred <- predict(pcr.fit, x[test,], ncomp = 7)
mean((pcr.pred -y.test)^2)
#This test set MSE is competitive with the results obtained using ridge regression
#and the lasso. However, as a result of the way PCR is implemented,
#the final model is more difficult to interpret because it does not perform
#any kind of variable selection or even directly produce coefficient estimates.

#Finally, we fit PCR on the full data set, using M = 7, the number of
#components identified by cross-validation.
pcr.fit <- pcr(y~x, scale=TRUE, ncomp=7)
summary(pcr.fit)

#We implement partial least squares (PLS)
pls.fit <- plsr(Salary~., data=Hitters , subset=train , scale=TRUE ,
             validation ="CV")
summary(pls.fit)#%var explained
validationplot(pls.fit, val.type="MSEP")
#The lowest cross-validation error occurs when only M = 2 partial least
#squares directions are used. We now evaluate the corresponding test set
#MSE.

pls.pred <- predict(pls.fit, x[test,], ncomp=2)
mean((pls.pred -y.test)^2)
#The test MSE is comparable to, but slightly higher than, the test MSE
#obtained using ridge regression, the lasso, and PCR.
#Finally, we perform PLS using the full data set, using M = 2, the number
#of components identified by cross-validation.

pls.fit <- plsr(Salary~., data=Hitters, scale=TRUE, ncomp=2)
summary(pls.fit)

#Notice that the percentage of variance in Salary that the two-component
#PLS fit explains, 46.40 %, is almost as much as that explained using the
#final seven-component model PCR fit, 46.69 %. This is because PCR only
#attempts to maximize the amount of variance explained in the predictors,
#while PLS searches for directions that explain variance in both the predictors
#and the response.

#------------------------------
set.seed(1)
X = rnorm(100)
eps = rnorm(100)

#Use the regsubsets() function to perform best subset selection
#in order to choose the best model containing the predictors
#X, X2,...,X10. What is the best model obtained
#choose betas yourself

beta0 <- 3
beta1 <- 2
beta2 <- -3
beta3 <- 0.3
Y <- beta0 + beta1 * X + beta2 * X^2 + beta3 * X^3 + eps

data.full <- data.frame(Y, X)
mod.full <- regsubsets(Y~poly(X, 10, raw=T), data=data.full, nvmax=10)
mod.summary <- summary(mod.full)

#Find the model size for best cp, BIC and adjr2
which.min(mod.summary$cp)
which.min(mod.summary$bic)
which.max(mod.summary$adjr2)
par(mfrow = c(3,1))
#Plot cp, BIC and adjr2
plot(mod.summary$cp, xlab="Subset Size", ylab="Cp", pch=20, type="l")
points(which.min(mod.summary$cp), mod.summary$cp[which.min(mod.summary$cp)], pch=20, col="red", lwd=7)
plot(mod.summary$bic, xlab="Subset Size", ylab="BIC", pch=20, type="l")
points(which.min(mod.summary$bic), mod.summary$bic[which.min(mod.summary$bic)], pch=20, col="red", lwd=7)
plot(mod.summary$adjr2, xlab="Subset Size", ylab="Adjusted R2", pch=20, type="l")
points(which.max(mod.summary$adjr2), mod.summary$adjr2[which.max(mod.summary$adjr2)], pch=20, col="red", lwd=7)
#We find that with Cp, BIC and Adjusted R2 criteria, 3, 3, and 3 variable models are respectively picked.

coefficients(mod.full, id=3)
#All statistics pick X^7 over X^3. The remaining coefficients are quite close to betas.

#Compare results with forward and backward selection
#We fit forward and backward stepwise models to the data.

mod.fwd <- regsubsets(Y~poly(X, 10, raw=T), data=data.full, nvmax=10, method="forward")
mod.bwd <- regsubsets(Y~poly(X, 10, raw=T), data=data.full, nvmax=10, method="backward")
fwd.summary = summary(mod.fwd)
bwd.summary = summary(mod.bwd)
which.min(fwd.summary$cp)
which.min(bwd.summary$cp)
which.min(fwd.summary$bic)
which.min(bwd.summary$bic)
which.max(fwd.summary$adjr2)
which.max(bwd.summary$adjr2)
# Plot the statistics
par(mfrow=c(3, 2))
plot(fwd.summary$cp, xlab="Subset Size", ylab="Forward Cp", pch=20, type="l")
points(which.min(fwd.summary$cp), fwd.summary$cp[which.min(fwd.summary$cp)], pch=4, col="red", lwd=7)
plot(bwd.summary$cp, xlab="Subset Size", ylab="Backward Cp", pch=20, type="l")
points(which.min(bwd.summary$cp), bwd.summary$cp[which.min(bwd.summary$cp)], pch=4, col="red", lwd=7)
plot(fwd.summary$bic, xlab="Subset Size", ylab="Forward BIC", pch=20, type="l")
points(which.min(fwd.summary$bic), fwd.summary$bic[which.min(fwd.summary$bic)], pch=4, col="red", lwd=7)
plot(bwd.summary$bic, xlab="Subset Size", ylab="Backward BIC", pch=20, type="l")
points(which.min(bwd.summary$bic), bwd.summary$bic[which.min(bwd.summary$bic)], pch=4, col="red", lwd=7)
plot(fwd.summary$adjr2, xlab="Subset Size", ylab="Forward Adjusted R2", pch=20, type="l")
points(which.max(fwd.summary$adjr2), fwd.summary$adjr2[which.max(fwd.summary$adjr2)], pch=4, col="red", lwd=7)
plot(bwd.summary$adjr2, xlab="Subset Size", ylab="Backward Adjusted R2", pch=20, type="l")
points(which.max(bwd.summary$adjr2), bwd.summary$adjr2[which.max(bwd.summary$adjr2)], pch=4, col="red", lwd=7)

#We see that all statistics pick 3 variable models except backward stepwise with adjusted R2. Here are the coefficients

coefficients(mod.fwd, id=3)
coefficients(mod.bwd, id=3)
coefficients(mod.fwd, id=4)

#Here forward stepwise picks X^7 over X^3. Backward stepwise with 3 variables picks X^9 while backward stepwise with 4 variables picks X^4 and X^7. All other coefficients are close to betas.

#Now fit a Lasso model
#Training Lasso on the data
xmat <- model.matrix(Y~poly(X, 10, raw=T), data=data.full)[, -1]
mod.lasso <- cv.glmnet(xmat, Y, alpha=1)
best.lambda <- mod.lasso$lambda.min
best.lambda
par(mfrow = c(1,1))
plot(mod.lasso)
#Next fit the model on entire data using best lambda
best.model <- glmnet(xmat, Y, alpha=1)
predict(best.model, s=best.lambda, type="coefficients")
#Lasso also picks X^5 over X^3. It also picks X^7 with negligible coefficient.

#Now generate a response vector Y according to the model
#Y = B0 + B7X7 + e,
#and perform best subset selection and the lasso. 

beta7 <- 7
Y <- beta0 + beta7 * X^7 + eps
#Predict using regsubsets
data.full <- data.frame("y" = Y, "x" = X)
mod.full <- regsubsets(y~poly(x, 10, raw=T), data=data.full, nvmax=10)
mod.summary <- summary(mod.full)

# Find the model size for best cp, BIC and adjr2
which.min(mod.summary$cp)
which.min(mod.summary$bic)
which.max(mod.summary$adjr2)
coefficients(mod.full, id=which.min(mod.summary$bic))
coefficients(mod.full, id=which.min(mod.summary$cp))
coefficients(mod.full, id=which.max(mod.summary$adjr2))

#We see that BIC picks the most accurate 1-variable model with matching coefficients. Other criteria pick additional variables.

xmat <- model.matrix(y~poly(x, 10, raw=T), data=data.full)[, -1]
mod.lasso <- cv.glmnet(xmat, Y, alpha=1)
best.lambda <- mod.lasso$lambda.min
best.lambda
best.model <- glmnet(xmat, Y, alpha=1)
predict(best.model, s=best.lambda, type="coefficients")
#Lasso also picks the best 1-variable model but intercept is quite off (3.8 vs 3).

#---------------------------OLS vs Ridge vs Lasso vs PCR vs PLS
# In this exercise, we will predict the number of applications received
#using the other variables in the College data set.

# Split the data set into a training set and a test set
set.seed(11)
sum(is.na(College))
train.size <- nrow(College)/ 2
train <- sample(1:nrow(College), train.size)
test <- -train
College.train <- College[train, ]
College.test <- College[test, ]

#Fit a linear model using least squares on the training set, and
#report the test error obtained

lm.fit <- lm(Apps~., data=College.train)
lm.pred <- predict(lm.fit, College.test)
mean((College.test[, "Apps"] - lm.pred)^2)#test error RSS

#Fit a ridge regression model on the training set, with lambda chosen
#by cross-validation. Report the test error obtained.
#Pick lambda using College.train and report error on College.test
train.mat <- model.matrix(Apps~., data=College.train)
test.mat <- model.matrix(Apps~., data=College.test)
grid <- 10^seq(4, -2, length=100)
mod.ridge <- cv.glmnet(train.mat, College.train[, "Apps"], alpha=0, lambda=grid, thresh=1e-12)
plot(mod.ridge)
lambda.best <- mod.ridge$lambda.min
lambda.best
ridge.pred <- predict(mod.ridge, newx=test.mat, s=lambda.best)
mean((College.test[, "Apps"] - ridge.pred)^2)#Test RSS is slightly higher that OLS

#Fit a lasso model on the training set, with lamnda chosen by crossvalidation.
#Report the test error obtained, along with the number
#of non-zero coefficient estimates.
#Pick lambda using College.train and report error on College.test

mod.lasso <- cv.glmnet(train.mat, College.train[, "Apps"], alpha=1, lambda=grid, thresh=1e-12)
lambda.best <- mod.lasso$lambda.min
lambda.best
lasso.pred <- predict(mod.lasso, newx=test.mat, s=lambda.best)
mean((College.test[, "Apps"] - lasso.pred)^2)#RSS slightly higher than OLS
#The coefficients look like
mod.lasso = glmnet(model.matrix(Apps~., data=College), College[, "Apps"], alpha=1)
predict(mod.lasso, s=lambda.best, type="coefficients")
par(mfrow = c(1,2))
plot(mod.lasso, "lambda", label = FALSE)
plot(mod.lasso, "norm", label = FALSE)
#In both plots, each colored line represents the value taken by a different coefficient in your model. 
#Lambda is the weight given to the regularization term (the L1 norm), 
#so as lambda approaches zero, the loss function of your model approaches the OLS loss function.
#Therefore, when lambda is very small, the LASSO solution should be very close to the OLS solution, 
#and all of your coefficients are in the model.
#Perhaps a better way to look at it is that the x-axis is the maximum permissible value the L1 norm can take. 
#So when you have a small L1 norm, you have a lot of regularization.
#The plot on the left and the plot on the right are basically showing you the same thing, just on different scales.
#We use corss-validation to tell us which value of the L1 norm (or equivalently, which log(lambda)) yields the model with best predictive ability


#Fit a PCR model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value
#of M selected by cross-validation.

#Use validation to fit pcr

pcr.fit <- pcr(Apps~., data=College.train, scale=T, validation="CV")
validationplot(pcr.fit, val.type="MSEP")
pcr.pred <- predict(pcr.fit, College.test, ncomp=10)
mean((College.test[, "Apps"] - data.frame(pcr.pred))^2)
#Test RSS for PCR is very high

#Fit a PLS model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value
#of M selected by cross-validation.

pls.fit = plsr(Apps~., data=College.train, scale=T, validation="CV")
validationplot(pls.fit, val.type="MSEP")
pls.pred = predict(pls.fit, College.test, ncomp=10)
mean((College.test[, "Apps"] - data.frame(pls.pred))^2)#This is quite low


# Comment on the results obtained. How accurately can we predict
#the number of college applications received? Is there much
#difference among the test errors resulting from these five approaches?

#Results for OLS, Lasso, Ridge are comparable. Lasso reduces the 
#F.Undergrad and Books variables to zero and shrinks coefficients of other variables. 
#Here are the test R^2 for all models.
test.avg <- mean(College.test[, "Apps"])
lm.test.r2 <- 1 - mean((College.test[, "Apps"] - lm.pred)^2) /mean((College.test[, "Apps"] - test.avg)^2)
ridge.test.r2 <- 1 - mean((College.test[, "Apps"] - ridge.pred)^2) /mean((College.test[, "Apps"] - test.avg)^2)
lasso.test.r2 <- 1 - mean((College.test[, "Apps"] - lasso.pred)^2) /mean((College.test[, "Apps"] - test.avg)^2)
pcr.test.r2 <- 1 - mean((College.test[, "Apps"] - data.frame(pcr.pred))^2) /mean((College.test[, "Apps"] - test.avg)^2)
pls.test.r2 <- 1 - mean((College.test[, "Apps"] - data.frame(pls.pred))^2) /mean((College.test[, "Apps"] - test.avg)^2)
barplot(c(lm.test.r2, ridge.test.r2, lasso.test.r2, pcr.test.r2, pls.test.r2), col="red", names.arg=c("OLS", "Ridge", "Lasso", "PCR", "PLS"), main="Test R-squared")

#The plot shows that test R^2 for all models except PCR are around 0.9, 
#with PLS having slightly higher test R^2 than others. 
#PCR has a smaller test R^2 of less than 0.8. 
#All models except PCR predict college applications with high accuracy.

#----------------------------10

#Generate a data set with p = 20 features, n = 1,000 observations,
#and an associated quantitative response vector generated
#according to the model
#Y = XB + e,
#where B has some elements that are exactly equal to zero.

set.seed(1)
p = 20
n = 1000
x = matrix(rnorm(n*p), n, p)
B = rnorm(p)
B[3] = 0
B[4] = 0
B[9] = 0
B[19] = 0
B[10] = 0
eps = rnorm(p)
y = x %*% B + eps

# Split your data set into a training set containing 100 observations
#and a test set containing 900 observations.
train = sample(seq(1000), 100, replace = FALSE)
y.train = y[train,]
y.test = y[-train,]
x.train = x[train,]
x.test = x[-train,]

#Perform best subset selection on the training set, and plot the
#training set MSE associated with the best model of each size.

regfit.full <- regsubsets(y~., data=data.frame(x=x.train, y=y.train), nvmax=p)
val.errors <- rep(NA, p)#initialize vector
x_cols <- colnames(x, do.NULL=FALSE, prefix="x.")
for (i in 1:p) {
  coefi <- coef(regfit.full, id=i)
  pred <- as.matrix(x.train[, x_cols %in% names(coefi)]) %*% coefi[names(coefi) %in% x_cols]
  val.errors[i] = mean((y.train - pred)^2)
}
plot(val.errors, ylab="Training MSE", pch=19, type="b")

#Plot the test set MSE associated with the best model of each size.

val.errors = rep(NA, p)
for (i in 1:p) {
  coefi = coef(regfit.full, id=i)
  pred = as.matrix(x.test[, x_cols %in% names(coefi)]) %*% coefi[names(coefi) %in% x_cols]
  val.errors[i] = mean((y.test - pred)^2)
}
plot(val.errors, ylab="Test MSE", pch=19, type="b")

#For which model size does the test set MSE take on its minimum value?
which.min(val.errors)

#How does the model at which the test set MSE is minimized
#compare to the true model used to generate the data?
coef(regfit.full, id=which.min(val.errors))
#Caught all but one zeroed out coefficient at x.19.

val.errors = rep(NA, p)
a = rep(NA, p)
b = rep(NA, p)
for (i in 1:p) {
  coefi = coef(regfit.full, id=i)
  a[i] = length(coefi)-1
  b[i] = sqrt(
    sum((B[x_cols %in% names(coefi)] - coefi[names(coefi) %in% x_cols])^2) +
      sum(B[!(x_cols %in% names(coefi))])^2)
}
plot(x=a, y=b, xlab="number of coefficients",
     ylab="error between estimated and true coefficients", type = "b")
which.min(b)
#Model with 9 coefficients (10 with intercept) minimizes the error between the estimated and true coefficients. 
#Test error is minimized with 16 parameter model. 
#A better fit of true coefficients as measured here doesn't mean the model will have a lower test MSE.

#-----------------------------Predict per capita crime rate on Boston data set
set.seed(1)
fix(Boston)
#Try out some of the regression methods explored in this chapter,
#such as best subset selection, the lasso, ridge regression, and
#PCR. Present and discuss results for the approaches that you
#consider.

#Best subset selection
predict.regsubsets <- function (object , newdata ,id ,...){
  form <- as.formula (object$call[[2]])
  mat <- model.matrix(form ,newdata )
  coefi <- coef(object ,id=id)
  xvars <- names(coefi)
  mat[,xvars]%*%coefi 
}
k <- 10
folds <- sample(1:k,nrow(Boston),replace=TRUE)
a <- ncol(Boston)-1
cv.errors <- matrix(NA, k, a)#initialize a vector

for (i in 1:k) {
  best.fit <- regsubsets(crim~., data=Boston[folds!=i,], nvmax=a)
  for (j in 1:a) {
    pred <- predict(best.fit, Boston[folds==i, ], id=j)
    cv.errors[i,j] = mean((Boston$crim[folds==i] - pred)^2)
  }
}

rmse.cv <- sqrt(apply(cv.errors, 2, mean))
plot(rmse.cv, pch=1, type="b")
which.min(rmse.cv)
rmse.cv[which.min(rmse.cv)]

#Lasso

x <- model.matrix(crim~.-1, data=Boston)
y <- Boston$crim
cv.lasso <- cv.glmnet(x, y, type.measure="mse")
plot(cv.lasso)
coef(cv.lasso)
sqrt(cv.lasso$cvm[cv.lasso$lambda == cv.lasso$lambda.1se])

#Ridge Regression

x <- model.matrix(crim~.-1, data=Boston)
y <- Boston$crim
cv.ridge <- cv.glmnet(x, y, type.measure="mse", alpha=0)
plot(cv.ridge)
plot(cv.ridge, "lambda")
coef(cv.ridge)
sqrt(cv.ridge$cvm[cv.ridge$lambda == cv.ridge$lambda.1se])

#PCR

pcr.fit <- pcr(crim~., data=Boston, scale=TRUE, validation="CV")
validationplot(pcr.fit, val.type="MSEP")
summary(pcr.fit)
#13 component pcr fit has lowest CV/adjCV RMSEP.

#Propose a model (or set of models) that seem to perform well on
#this data set, and justify your answer. Make sure that you are
#evaluating model performance using validation set error, crossvalidation,
#or some other reasonable alternative, as opposed to
#using training error.
#Does your chosen model involve all of the features in the data set?

#See above answers for cross-validate mean squared errors of selected models.
#I would choose the 12 parameter best subset model because it had the best cross-validated RMSE, next to PCR, but it was simpler model than the 13 component PCR model.


