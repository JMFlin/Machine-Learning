library(ISLR)
library(MASS)
library(e1071)

#Chapter 9: Support Vector Machines

#----------------------------SUPPORT VECTOR CLASSIFIER

#The e1071 library contains implementations for a number of statistical
#learning methods. In particular, the svm() function can be used to fit a
#support vector classifier when the argument kernel="linear" is used. This
#function uses a slightly different formulation from (9.14) and (9.25) for the
#support vector classifier. A cost argument allows us to specify the cost of
#a violation to the margin. When the cost argument is small, then the margins
#will be wide and many support vectors will be on the margin or will
#violate the margin. When the cost argument is large, then the margins will
#be narrow and there will be few support vectors on the margin or violating the margin.

#We now use the svm() function to fit the support vector classifier for a
#given value of the cost parameter. Here we demonstrate the use of this
#function on a two-dimensional example so that we can plot the resulting
#decision boundary. We begin by generating the observations, which belong to two classes.

set.seed(1)
x <- matrix(rnorm (20*2), ncol=2)
y <- c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1

#We begin by checking whether the classes are linearly separable.
plot(x, col=(3-y))

#They are not. Next, we fit the support vector classifier. Note that in order
#for the svm() function to perform classification (as opposed to SVM-based
#regression), we must encode the response as a factor variable. We now
#create a data frame with the response coded as a factor.

dat <- data.frame(x=x, y=as.factor(y))
svmfit <- svm(y~., data=dat , kernel ="linear", cost=10, scale=FALSE)

#The argument scale=FALSE tells the svm() function not to scale each feature
#to have mean zero or standard deviation one; depending on the application,
#one might prefer to use scale=TRUE.

#We can now plot the support vector classifier obtained:
plot(svmfit, dat)

#Note that the two arguments to the plot.svm() function are the output
#of the call to svm(), as well as the data used in the call to svm(). The
#region of feature space that will be assigned to the ???1 class is shown in
#light blue, and the region that will be assigned to the +1 class is shown in
#purple. The decision boundary between the two classes is linear (because we
#used the argument kernel="linear"), though due to the way in which the
#plotting function is implemented in this library the decision boundary looks
#somewhat jagged in the plot. We see that in this case only one observation
#is misclassified. (Note that here the second feature is plotted on the x-axis
#and the first feature is plotted on the y-axis, in contrast to the behavior of
#the usual plot() function in R.) 

#The support vectors are plotted as crosses
#and the remaining observations are plotted as circles; we see here that there
#are seven support vectors. We can determine their identities as follows:

svmfit$index

#We can obtain some basic information about the support vector classifier
#fit using the summary() command:
summary(svmfit)

#This tells us, for instance, that a linear kernel was used with cost=10, and
#that there were seven support vectors, four in one class and three in the other.

#What if we instead used a smaller value of the cost parameter?
svmfit <- svm(y~., data=dat , kernel ="linear", cost =0.1,scale=FALSE)
plot(svmfit , dat)
svmfit$index

#Now that a smaller value of the cost parameter is being used, we obtain a
#larger number of support vectors, because the margin is now wider. Unfortunately,
#the svm() function does not explicitly output the coefficients of
#the linear decision boundary obtained when the support vector classifier is
#fit, nor does it output the width of the margin.


#The e1071 library includes a built-in function, tune(), to perform cross-
#validation. By default, tune() performs ten-fold cross-validation on a set
#of models of interest. In order to use this function, we pass in relevant
#information about the set of models that are under consideration. The
#following command indicates that we want to compare SVMs with a linear
#kernel, using a range of values of the cost parameter.

set.seed(1)
tune.out <- tune(svm ,y~.,data=dat, kernel ="linear", ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

#We can easily access the cross-validation errors for each of these models using the summary() command:
summary (tune.out)

#We see that cost=0.1 results in the lowest cross-validation error rate. The
#tune() function stores the best model obtained, which can be accessed as follows:

bestmod <- tune.out$best.model
summary(bestmod)

#The predict() function can be used to predict the class label on a set of
#test observations, at any given value of the cost parameter. We begin by generating a test data set.

xtest <- matrix(rnorm (20*2) , ncol=2)
ytest <- sample (c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]= xtest[ytest==1,] + 1
testdat <- data.frame(x=xtest , y=as.factor(ytest))

#Now we predict the class labels of these test observations. Here we use the
#best model obtained through cross-validation in order to make predictions.

ypred <- predict (bestmod ,testdat)
pred.table <- table(predict=ypred , truth=testdat$y )
pred.table[1,1] + pred.table[2,2]#this number is the correct classified count

#Thus, with this value of cost, that amount of the test observations are correctly
#classified. What if we had instead used cost=0.01?

svmfit <- svm(y~., data=dat , kernel ="linear", cost =.01, scale=FALSE)
ypred <- predict (svmfit ,testdat )
pred.table <- table(predict =ypred , truth=testdat$y )
pred.table[1,1] + pred.table[2,2]#this number is the correct classified count

#Now consider a situation in which the two classes are linearly separable.
#Then we can find a separating hyperplane using the svm() function. We
#first further separate the two classes in our simulated data so that they are linearly separable:

x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch =19)

#Now the observations are just barely linearly separable. We fit the support
#vector classifier and plot the resulting hyperplane, using a very large value
#of cost so that no observations are misclassified.

dat <- data.frame(x=x,y=as.factor(y))
svmfit <- svm(y~., data=dat , kernel ="linear", cost=1e5)
summary(svmfit)
plot(svmfit , dat)

#No training errors were made and only three support vectors were used.
#However, we can see from the figure that the margin is very narrow (because
#the observations that are not support vectors, indicated as circles, are very
#close to the decision boundary). It seems likely that this model will perform
#poorly on test data. We now try a smaller value of cost:

svmfit <- svm(y~., data=dat , kernel ="linear", cost=1)
summary(svmfit)
plot(svmfit ,dat)

#Using cost=1, we misclassify a training observation, but we also obtain
#a much wider margin and make use of seven support vectors. It seems
#likely that this model will perform better on test data than the model with cost=1e5.

#-----------------SUPPORT VECTOR CLASSIFIER

#In order to fit an SVM using a non-linear kernel, we once again use the svm()
#function. However, now we use a different value of the parameter kernel.
#To fit an SVM with a polynomial kernel we use kernel="polynomial", and
#to fit an SVM with a radial kernel we use kernel="radial". In the former
#case we also use the degree argument to specify a degree for the polynomial
#kernel (this is d in (9.22)), and in the latter case we use gamma to specify a
#value of gamma for the radial basis kernel (9.24).

#We first generate some data with a non-linear class boundary, as follows:

set.seed(1)
x <- matrix(rnorm (200*2) , ncol=2)
x[1:100,] <- x[1:100,]+2
x[101:150 ,] <- x[101:150,]-2
y <- c(rep(1,150) ,rep(2,50))
dat <- data.frame(x=x,y=as.factor(y))

#Plotting the data makes it clear that the class boundary is indeed nonlinear:
plot(x, col=y)

#The data is randomly split into training and testing groups. We then fit
#the training data using the svm() function with a radial kernel and gamma = 1:

train <- sample(200,100)
svmfit <- svm(y~., data=dat[train ,], kernel ="radial", gamma=1, cost=1)
plot(svmfit , dat[train,])

#The plot shows that the resulting SVM has a decidedly non-linear
#boundary. The summary() function can be used to obtain some
#information about the SVM fit:

summary(svmfit)

#We can see from the figure that there are a fair number of training errors
#in this SVM fit. If we increase the value of cost, we can reduce the number
#of training errors. However, this comes at the price of a more irregular
#decision boundary that seems to be at risk of overfitting the data.

svmfit <- svm(y~., data=dat[train,], kernel ="radial",gamma=1, cost=1e5)

plot(svmfit ,dat[train,])

#We can perform cross-validation using tune() to select the best choice of
#gamma and cost for an SVM with a radial kernel:

set.seed(1)
tune.out <- tune(svm, y~., data=dat[train,], kernel ="radial",ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4) ))
summary (tune.out)

#Therefore, the best choice of parameters involves cost=1 and gamma=0.5 (see summary output). We
#can view the test set predictions for this model by applying the predict()
#function to the data. Notice that to do this we subset the dataframe dat
#using -train as an index set.

pred.table <- table(true=dat[-train,"y"], pred=predict (tune.out$best.model, newx=dat[-train,]))

1-sum(diag(pred.table))/sum(pred.table)#40% of test observations are misclassified by this SVM.

#-----------------------ROC CURVE

#The ROCR package can be used to produce ROC curves such as those in
#Figures 9.10 and 9.11. We first write a short function to plot an ROC curve
#given a vector containing a numerical score for each observation, pred, and
#a vector containing the class label for each observation, truth.

rocplot <- function (pred, truth, ...){
predob = prediction (pred, truth)
perf = performance (predob, "tpr", "fpr")
plot(perf,...)}

#SVMs and support vector classifiers output class labels for each observation.
#However, it is also possible to obtain fitted values for each observation,
#which are the numerical scores used to obtain the class labels.

#For an SVM with a non-linear kernel, the equation that yields the fitted
#value is given in (9.23). In essence, the sign of the fitted value determines
#on which side of the decision boundary the observation lies. Therefore, the
#relationship between the fitted value and the class prediction for a given
#observation is simple: if the fitted value exceeds zero then the observation
#is assigned to one class, and if it is less than zero than it is assigned to the
#other. In order to obtain the fitted values for a given SVM model fit, we
#use decision.values=TRUE when fitting svm(). Then the predict() function
#will output the fitted values.

svmfit.opt <- svm(y~., data=dat[train ,], kernel ="radial", gamma=2, cost=1, decision.values =T)
fitted <- attributes(predict(svmfit.opt ,dat[train,], decision.values=TRUE))$decision.values
#Now we can produce the ROC plot.
par(mfrow=c(1,2))
rocplot(fitted ,dat[train,"y"], main="Training Data")

#SVM appears to be producing accurate predictions. By increasing gamma we can
#produce a more flexible fit and generate further improvements in accuracy.

svmfit.flex <- svm(y~., data=dat[train,], kernel ="radial", gamma=50, cost=1, decision.values =T)
fitted <- attributes(predict(svmfit.flex ,dat[train,], decision.values=T))$decision.values
rocplot(fitted, dat[train ,"y"], add=T, col="red")

#However, these ROC curves are all on the training data. We are really
#more interested in the level of prediction accuracy on the test data. When
#we compute the ROC curves on the test data, the model with gamma = 2 appears
#to provide the most accurate results.

fitted <- attributes(predict(svmfit.opt ,dat[-train ,], decision.values=T))$decision.values
rocplot(fitted ,dat[-train ,"y"], main="Test Data")
fitted <- attributes(predict(svmfit.flex ,dat[-train ,], decision.values=T))$decision.values
rocplot (fitted ,dat[-train ,"y"],add=T,col="red")

#----------------------SVM WITH MULTIPLE CLASSES

#If the response is a factor containing more than two levels, then the svm()
#function will perform multi-class classification using the one-versus-one approach.
#We explore that setting here by generating a third class of observations.

set.seed(1)
x <- rbind(x, matrix(rnorm (50*2) , ncol=2))
y <- c(y, rep(0,50))
x[y==0,2] <-  x[y==0 ,2]+2
dat <- data.frame(x=x, y=as.factor(y))
par(mfrow=c(1,1))
plot(x,col=(y+1))

#We now fit an SVM to the data:

svmfit <- svm(y~., data=dat , kernel ="radial", cost=10, gamma =1)
plot(svmfit , dat)

#The e1071 library can also be used to perform support vector regression,
#if the response vector that is passed in to svm() is numerical rather than a factor.

#We now examine the Khan data set, which consists of a number of tissue
#samples corresponding to four distinct types of small round blue cell tumors.
#For each tissue sample, gene expression measurements are available.
#The data set consists of training data, xtrain and ytrain, and testing data, xtest and ytest.

names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)

#This data set consists of expression measurements for 2,308 genes.
#The training and test sets consist of 63 and 20 observations respectively.

table(Khan$ytrain)
table(Khan$ytest)

#We will use a support vector approach to predict cancer subtype using gene
#expression measurements. In this data set, there are a very large number
#of features relative to the number of observations. This suggests that we
#should use a linear kernel, because the additional flexibility that will result
#from using a polynomial or radial kernel is unnecessary.

dat <- data.frame(x=Khan$xtrain , y=as.factor(Khan$ytrain ))
out <- svm(y~., data=dat , kernel ="linear",cost=10)
summary(out)

#We see that there are no training errors. In fact, this is not surprising,
#because the large number of variables relative to the number of observations
#implies that it is easy to find hyperplanes that fully separate the classes. We
#are most interested not in the support vector classifier's performance on the
#training observations, but rather its performance on the test observations.

dat.te <- data.frame(x=Khan$xtest , y=as.factor(Khan$ytest ))
pred.te <- predict (out , newdata =dat.te)
pred.table <- table(pred.te, dat.te$y)
pred.table
#We see that using cost=10 yields two test set errors on this data.
1-sum(diag(pred.table))/sum(pred.table)#misclassification rate

#------------------APPLIED

#------SUPPORT VECTOR MACHINE

#Generate a simulated two-class data set with 100 observations and
#two features in which there is a visible but non-linear separation between
#the two classes. Show that in this setting, a support vector
#machine with a polynomial kernel (with degree greater than 1) or a
#radial kernel will outperform a support vector classifier on the training
#data. Which technique performs best on the test data? Make
#plots and report training and test error rates in order to back up your assertions.

#We create a random initial dataset which lies along the parabola y = 3*x^2 + 4. 
#We then separate the two classes by translating them along Y-axis.

x <- rnorm(100)
y <- 3*x^2 + 4 + rnorm(100)
train <- sample(100, 50)
y[train] <- y[train] + 3
y[-train] <- y[-train] - 3

# Plot using different colors
plot(x[train], y[train], pch="+", lwd=4, col="red", ylim=c(-4, 20), xlab="X", ylab="Y")
points(x[-train], y[-train], pch="o", lwd=4, col="blue")

#The plot clearly shows non-linear separation. 
#We now create both train and test dataframes by taking half of positive and negative 
#classes and creating a new z vector of 0 and 1 for classes.

z <- rep(0, 100)
z[train] <- 1
# Take 25 observations each from train and -train
final.train <- c(sample(train, 25), sample(setdiff(1:100, train), 25))
data.train <- data.frame(x=x[final.train], y=y[final.train], z=as.factor(z[final.train]))
data.test <- data.frame(x=x[-final.train], y=y[-final.train], z=as.factor(z[-final.train]))
svm.linear <- svm(z~., data=data.train, kernel="linear", cost=10)
plot(svm.linear, data.train)
pred.table <- table(z[final.train], predict(svm.linear, data.train))
pred.table
1-sum(diag(pred.table))/sum(pred.table)#misclassification rate
#The plot shows the linear boundary. The classifier makes 5 classification errors on train data.

#Next, we train an SVM with polynomial kernel
svm.poly <- svm(z~., data=data.train, kernel="polynomial", cost=10)
plot(svm.poly, data.train)
pred.table <- table(z[final.train], predict(svm.poly, data.train))
pred.table
1-sum(diag(pred.table))/sum(pred.table)#misclassification rate
#This is a default polynomial kernel with degree 3. It makes 9 errors on train data.

#Finally, we train an SVM with radial basis kernel with gamma of 1.
svm.radial <- svm(z~., data=data.train, kernel="radial", gamma=1, cost=10)
plot(svm.radial, data.train)
pred.table <- table(z[final.train], predict(svm.radial, data.train))
pred.table
1-sum(diag(pred.table))/sum(pred.table)#misclassification rate
#This classifier perfectly classifies train data!

#Here are how the test errors look like.
plot(svm.linear, data.test)
plot(svm.poly, data.test)
plot(svm.radial, data.test)
table(z[-final.train], predict(svm.linear, data.test))
table(z[-final.train], predict(svm.poly, data.test))
table(z[-final.train], predict(svm.radial, data.test))

#The tables show that linear, polynomial and radial basis kernels classify 5, 13, and 1 test points incorrectly respectively. 
#Radial basis kernel is the best and has a zero test misclassification error.

#----------------------------

#We have seen that we can fit an SVM with a non-linear kernel in order
#to perform classification using a non-linear decision boundary. We will
#now see that we can also obtain a non-linear decision boundary by
#performing logistic regression using non-linear transformations of the features.

#Generate a data set with n = 500 and p = 2, such that the observations
#belong to two classes with a quadratic decision boundary
#between them. For instance, you can do this as follows:

x1 <- runif (500) -0.5
x2 <- runif (500) -0.5
y <- 1*(x1^2-x2^2 > 0)

#Plot the observations, colored according to their class labels.
#Your plot should display X1 on the x-axis, and X2 on the yaxis.

plot(x1[y==0], x2[y==0], col="red", xlab="X1", ylab="X2", pch="+")
points(x1[y==1], x2[y==1], col="blue", pch=4)
#The plot clearly shows non-linear decision boundary.

#Fit a logistic regression model to the data, using X1 and X2 as predictors.

lm.fit <- glm(y~x1+x2, family=binomial)
summary(lm.fit)
#Both variables are insignificant for predicting y.

#Apply this model to the training data in order to obtain a predicted
#class label for each training observation. Plot the observations,
#colored according to the predicted class labels. The decision boundary should be linear.

data <- data.frame(x1=x1, x2=x2, y=y)
lm.prob <- predict(lm.fit, data, type="response")
lm.pred <- ifelse(lm.prob > 0.52, 1, 0)
data.pos <- data[lm.pred == 1, ]
data.neg <- data[lm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col="blue", xlab="X1", ylab="X2", pch="+")
points(data.neg$x1, data.neg$x2, col="red", pch=4)
#With the given model and a probability threshold of 0.5, all points are classified to single class and no decision boundary can be shown. 
#Hence we shift the probability threshold to 0.52 to show a meaningful decision boundary. 
#This boundary is linear as seen in the figure.

#Now fit a logistic regression model to the data using non-linear
#functions of X1 and X2 as predictors (e.g. X21 , X1×X2, log(X2),and so forth).

#We use squares, product interaction terms to fit the model.
lm.fit <- glm(y~poly(x1, 2)+poly(x2, 2) + I(x1 * x2), data=data, family=binomial)

#Apply this model to the training data in order to obtain a predicted
#class label for each training observation. Plot the observations,
#colored according to the predicted class labels. The
#decision boundary should be obviously non-linear. If it is not,
#then repeat (a)-(e) until you come up with an example in which
#the predicted class labels are obviously non-linear.

lm.prob <- predict(lm.fit, data, type="response")
lm.pred <- ifelse(lm.prob > 0.5, 1, 0)
data.pos <- data[lm.pred == 1, ]
data.neg <- data[lm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col="blue", xlab="X1", ylab="X2", pch="+")
points(data.neg$x1, data.neg$x2, col="red", pch=4)
#This non-linear decision boundary closely resembles the true decision boundary.

#Fit a support vector classifier to the data with X1 and X2 as
#predictors. Obtain a class prediction for each training observation.
#Plot the observations, colored according to the predicted class labels.

svm.fit <- svm(as.factor(y)~x1+x2, data, kernel="linear", cost=0.1)
svm.pred <- predict(svm.fit, data)
data.pos <- data[svm.pred == 1, ]
data.neg <- data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col="blue", xlab="X1", ylab="X2", pch="+")
points(data.neg$x1, data.neg$x2, col="red", pch=4)
#A linear kernel, even with low cost fails to find non-linear decision boundary and classifies all points to a single class.

#Fit a SVM using a non-linear kernel to the data. Obtain a class
#prediction for each training observation. Plot the observations,
#colored according to the predicted class labels.

svm.fit <- svm(as.factor(y)~x1+x2, data, gamma=1)
svm.pred <- predict(svm.fit, data)
data.pos <- data[svm.pred == 1, ]
data.neg <- data[svm.pred == 0, ]
plot(data.pos$x1, data.pos$x2, col="blue", xlab="X1", ylab="X2", pch="+")
points(data.neg$x1, data.neg$x2, col="red", pch=4)
#Again, the non-linear decision boundary on predicted labels closely resembles the true decision boundary.

#Comment on your results.

#This experiment enforces the idea that SVMs with non-linear kernel are extremely powerful in finding non-linear boundary. 
#Both, logistic regression with non-interactions and SVMs with linear kernels fail to find the decision boundary. 
#Adding interaction terms to logistic regression seems to give them same power as radial-basis kernels. 
#However, there is some manual efforts and tuning involved in picking right interaction terms. 
#This effort can become prohibitive with large number of features. 
#Radial basis kernels, on the other hand, only require tuning of one parameter - gamma - which can be easily done using cross-validation.

#---------------------------

#At the end of Section 9.6.1, it is claimed that in the case of data that
#is just barely linearly separable, a support vector classifier with a
#small value of cost that misclassifies a couple of training observations
#may perform better on test data than one with a huge value of cost
#that does not misclassify any training observations. You will now investigate this claim.

#Generate two-class data with p = 2 in such a way that the classes
#are just barely linearly separable.

#We randomly generate 1000 points and scatter them across line $x = y$ with wide margin. 
#We also create noisy points along the line $5x -4y - 50 = 0$. 
#These points make the classes barely separable and also shift the maximum margin classifier.

# Class one 
x.one <- runif(500, 0, 90)
y.one <- runif(500, x.one + 10, 100)
x.one.noise <- runif(50, 20, 80)
y.one.noise <- 5 / 4 * (x.one.noise - 10) + 0.1

# Class zero
x.zero <- runif(500, 10, 100)
y.zero <- runif(500, 0, x.zero - 10)
x.zero.noise <- runif(50, 20, 80)
y.zero.noise <- 5 / 4 * (x.zero.noise - 10) - 0.1

# Combine all
class.one <- seq(1, 550)
x <- c(x.one, x.one.noise, x.zero, x.zero.noise)
y <- c(y.one, y.one.noise, y.zero, y.zero.noise)

plot(x[class.one], y[class.one], col="blue", pch="+", ylim=c(0, 100))
points(x[-class.one], y[-class.one], col="red", pch=4)
#The plot shows that classes are barely separable. The noisy points create a fictitious boundary $5x - 4y - 50 = 0$.

#Compute the cross-validation error rates for support vector
#classifiers with a range of cost values. How many training errors
#are misclassified for each value of cost considered, and how
#does this relate to the cross-validation errors obtained?

#We create a z variable according to classes.
z <- rep(0, 1100)
z[class.one] = 1
data <- data.frame(x=x, y=y, z=z)
tune.out <- tune(svm, as.factor(z)~., data=data, kernel="linear", ranges=list(cost=c(0.01, 0.1, 1, 5, 10, 100, 1000, 10000)))
summary(tune.out)
data.frame(cost=tune.out$performances$cost, misclass=tune.out$performances$error * 1100)
#The table above shows train-misclassification error for all costs. 
#A cost of 10000 seems to classify all points correctly. This also corresponds to a cross-validation error of 0.

# Generate an appropriate test data set, and compute the test
#errors corresponding to each of the values of cost considered.
#Which value of cost leads to the fewest test errors, and how
#does this compare to the values of cost that yield the fewest
#training errors and the fewest cross-validation errors?

#We now generate a random test-set of same size. This test-set satisfies the true decision boundary $x = y$.

x.test <- runif(1000, 0, 100)
class.one <- sample(1000, 500)
y.test <- rep(NA, 1000)
# Set y > x for class.one
for(i in class.one) {
  y.test[i] <- runif(1, x.test[i], 100)
}
# set y < x for class.zero
for (i in setdiff(1:1000, class.one)) {
  y.test[i] <- runif(1, 0, x.test[i])
}
plot(x.test[class.one], y.test[class.one], col="blue", pch="+")
points(x.test[-class.one], y.test[-class.one], col="red", pch=4)

#We now make same predictions using all linear svms with all costs used in previous part.

z.test <- rep(0, 1000)
z.test[class.one] <- 1
all.costs <- c(0.01, 0.1, 1, 5, 10, 100, 1000, 10000)
test.errors <- rep(NA, 8)
data.test <- data.frame(x=x.test, y=y.test, z=z.test)
for (i in 1:length(all.costs)) {
  svm.fit <- svm(as.factor(z)~., data=data, kernel="linear", cost=all.costs[i])
  svm.predict <- predict(svm.fit, data.test)
  test.errors[i] <- sum(svm.predict != data.test$z)
}
data.frame(cost=all.costs, "test misclass"=test.errors)
#$\tt{cost} = 10$ seems to be performing better on test data, making the least number of classification errors. 
#This is much smaller than optimal value of 10000 for training data.

#Discuss your results.

#We again see an overfitting phenomenon for linear kernel. 
#A large cost tries to fit correctly classify noisy-points and hence overfits the train data. 
#A small cost, however, makes a few errors on the noisy test points and performs better on test data.

#----------------------------SUPPORT VECTORS ON AUTO DATA

#In this problem, you will use support vector approaches in order to
#predict whether a given car gets high or low gas mileage based on the Auto data set.

#Create a binary variable that takes on a 1 for cars with gas
#mileage above the median, and a 0 for cars with gas mileage below the median.
gas.med <- median(Auto$mpg)
new.var <- ifelse(Auto$mpg > gas.med, 1, 0)
Auto$mpglevel <- as.factor(new.var)

#Fit a support vector classifier to the data with various values
#of cost, in order to predict whether a car gets high or low gas
#mileage. Report the cross-validation errors associated with different
#values of this parameter. Comment on your results.
tune.out <- tune(svm, mpglevel~., data=Auto, kernel="linear", ranges=list(cost=c(0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)
#We see that cross-validation error is minimized for $\tt{cost}=1$.

#Now repeat (b), this time using SVMs with radial and polynomial
#basis kernels, with different values of gamma and degree and
#cost. Comment on your results.

tune.out <- tune(svm, mpglevel~., data=Auto, kernel="polynomial", ranges=list(cost=c(0.1, 1, 5, 10), degree=c(2, 3, 4)))
summary(tune.out)
#The lowest cross-validation error is obtained for $\tt{cost} = 10$ and $\tt{degree} = 2$.

tune.out <- tune(svm, mpglevel~., data=Auto, kernel="radial", ranges=list(cost=c(0.1, 1, 5, 10), gamma=c(0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)
#Finally, for radial basis kernel, $\tt{cost} = 5$ and $\tt{gamma} = 0.1$.

#Make some plots to back up your assertions in (b) and (c).
#Hint: In the lab, we used the plot() function for svm objects
#only in cases with p = 2. When p > 2, you can use the plot()
#function to create plots displaying pairs of variables at a time.
#Essentially, instead of typing
#plot(svmfit , dat)
#where svmfit contains your fitted model and dat is a data frame
#containing your data, you can type
#plot(svmfit , dat , x1~x4)
#in order to plot just the first and fourth variables. However, you
#must replace x1 and x4 with the correct variable names. To find
#out more, type ?plot.svm

svm.linear <- svm(mpglevel~., data=Auto, kernel="linear", cost=1)
svm.poly <- svm(mpglevel~., data=Auto, kernel="polynomial", cost=10, degree=2)
svm.radial <- svm(mpglevel~., data=Auto, kernel="radial", cost=10, gamma=0.01)
plotpairs <- function(fit){
  for(name in names(Auto)[!(names(Auto) %in% c("mpg", "mpglevel","name"))]) {
    plot(fit, Auto, as.formula(paste("mpg~", name, sep="")))
  }
}
plotpairs(svm.linear)
plotpairs(svm.poly)
plotpairs(svm.radial)


#---------------------------SUPPORT VECTORS ON OJ DATA

#This problem involves the OJ data set which is part of the ISLR package.

#Create a training set containing a random sample of 800
#observations, and a test set containing the remaining observations.

set.seed(9004)
train <- sample(dim(OJ)[1], 800)
OJ.train <- OJ[train, ]
OJ.test <- OJ[-train, ]

#Fit a support vector classifier to the training data using
#cost=0.01, with Purchase as the response and the other variables
#as predictors. Use the summary() function to produce summary
#statistics, and describe the results obtained.

svm.linear <- svm(Purchase~., kernel="linear", data=OJ.train, cost=0.01)
summary(svm.linear)

#Support vector classifier creates 432 support vectors out of 800 training points. 
#Out of these, 217 belong to level $\tt{CH}$ and remaining 215 belong to level $\tt{MM}$.

#What are the training and test error rates?
train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)
(82 + 53) / (439 + 53 + 82 + 226)
test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)
(19 + 29) / (142 + 19 + 29 + 80)
#The training error rate is $16.9$% and test error rate is about $17.8$%.

# Use the tune() function to select an optimal cost. Consider values
#in the range 0.01 to 10.
tune.out <- tune(svm, Purchase~., data=OJ.train, kernel="linear", ranges=list(cost=10^seq(-2, 1, by=0.25)))
summary(tune.out)
#Tuning shows that optimal cost is 0.3162

#Compute the training and test error rates using this new value for cost.
svm.linear <- svm(Purchase~., kernel="linear", data=OJ.train, cost=tune.out$best.parameters$cost)
train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)
(57 + 71) / (435 + 57 + 71 + 237)
test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)
(29 + 20) / (141 + 20 + 29 + 80)
#The training error decreases to $16$% but test error slightly increases to $18.1$% by using best cost.

#Repeat parts (b) through (e) using a support vector machine
#with a radial kernel. Use the default value for gamma.

set.seed(410)
svm.radial <- svm(Purchase~., data=OJ.train, kernel="radial")
summary(svm.radial)
train.pred <- predict(svm.radial, OJ.train)
table(OJ.train$Purchase, train.pred)
(40 + 78) / (452 + 40 + 78 + 230)
test.pred <- predict(svm.radial, OJ.test)
table(OJ.test$Purchase, test.pred)
(27 + 15) / (146 + 15 + 27 + 82)
#The radial basis kernel with default gamma creates 367 support vectors, out of which, 
#184 belong to level $\tt{CH}$ and remaining 183 belong to level $\tt{MM}$. 
#The classifier has a training error of $14.7$% and a test error of $15.6$% which is a slight improvement over linear kernel. 
#We now use cross validation to find optimal gamma.

set.seed(755)
tune.out <- tune(svm, Purchase~., data=OJ.train, kernel="radial", ranges=list(cost=10^seq(-2, 1, by=0.25)))
summary(tune.out)
svm.radial <- svm(Purchase~., data=OJ.train, kernel="radial", cost=tune.out$best.parameters$cost)
train.pred <- predict(svm.radial, OJ.train)
table(OJ.train$Purchase, train.pred)
(77 + 40) / (452 + 40 + 77 + 231)
test.pred <- predict(svm.radial, OJ.test)
table(OJ.test$Purchase, test.pred)
(28 + 15) / (146 + 15 + 28 + 81)
#Tuning slightly decreases training error to $14.6$% and slightly increases test error to $16$% 
#which is still better than linear kernel.

#Repeat parts (b) through (e) using a support vector machine
#with a polynomial kernel. Set degree=2.

set.seed(8112)
svm.poly <- svm(Purchase~., data=OJ.train, kernel="poly", degree=2)
summary(svm.poly)
train.pred <- predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
(32 + 105) / (460 + 32 + 105 + 203)
test.pred <- predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)
(12 + 37) / (149 + 12 + 37 + 72)

#Summary shows that polynomial kernel produces 452 support vectors, out of which, 232 belong to level $\tt{CH}$ and 
#remaining 220 belong to level $\tt{MM}$. This kernel produces a train error of $17.1$% and a test error of $18.1$% 
#which are slightly higher than the errors produces by radial kernel but lower than the errors produced by linear kernel.

set.seed(322)
tune.out <- tune(svm, Purchase~., data=OJ.train, kernel="poly", degree=2, ranges=list(cost=10^seq(-2, 1, by=0.25)))
summary(tune.out)
svm.poly <- svm(Purchase~., data=OJ.train, kernel="poly", degree=2, cost=tune.out$best.parameters$cost)
train.pred <- predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred)
(37 + 84) / (455 + 37 + 84 + 224)
test.pred = predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred)
(13 + 34) / (148 + 13 + 34 + 75)
#Tuning reduces the training error to $15.12$% and test error to $17.4$% which is worse than radial kernel 
#but slightly better than linear kernel.

#Overall, radial basis kernel seems to be producing minimum misclassification error on both train and test data.


#-------------------------GRAPHICS Linear SVM
x <- matrix(rnorm(40),20,2)
y <- rep(c(-1,1), c(10,10))
x[y==1,] <- x[y==1,]+1
plot(x,col=y+3,pch=19)

dat <- data.frame(x,y=as.factor(y))
svmfit <- svm(y~.,data = dat, kernel = "linear", cost = 10, scale = FALSE)#Use CV to select cost
svmfit#Support vectors are close to the boundry or on the wrong side of the boundry

plot(svmfit, dat)#ugly plot we will make a better one

#we make a grid of values that colors the whole domain

make.grid <- function(x,n=75){
    grange <- apply(x,2,range)
    x1 <- seq(from = grange[1,1], to = grange[2,1], length=n)
    x2 <- seq(from = grange[1,2], to = grange[2,2], length=n)
    expand.grid(X1=x1, X2=x2)
}

xgrid <- make.grid(x)
ygrid <- predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)],pch=20,cex=0.2)
points(x,col=y+3,pch=19)
points(x[svmfit$index,],pch=5,cex=2)

beta <- drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 <- svmfit$rho
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)],pch=20,cex=0.2)
points(x,col=y+3,pch=19)
points(x[svmfit$index,],pch=5,cex=2)
abline((beta0)/beta[2], -beta[1]/beta[2])
abline((beta0-1)/beta[2],-beta[1]/beta[2],lty=2)
abline((beta0+1)/beta[2],-beta[1]/beta[2],lty=2)


#--------------------GRAPHICS Nonlinear SVM

load(url("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/ESL.mixture.rda"))
names(ESL.mixture)
rm(x,y)
attach(ESL.mixture)
plot(x,col=y+1)
dat <- data.frame(y=factor(y),x)#factorize response
fit <- svm(factor(y)~.,data=dat,scale = FALSE, kernel="radial",cost=5)#Use CV to select cost

xgrid <- expand.grid(X1=px1, X2=px2)#these were supplied in the dataset! You can do it with the above make.grid function
ygrid <- predict(fit,xgrid)
plot(xgrid,col=as.numeric(ygrid),pch=20,cex=0.2)
points(x,col=y+1,pch=19)

func <- predict(fit,xgrid,decision.values=TRUE)
func <- attributes(func)$decision

contour(px1,px2,matrix(func,69,99),level = 0, add = TRUE)
contour(px1,px2,matrix(prob,69,99), level = 0.5, add = TRUE, col = "blue",lwd=2)#THis is the true decision boundry
#also called the bayesian decision boundry. If we had a large amount of data
#we would hope to get very close to this.
#out nonlinear SVM gets pretty close to this!

#redoing with make.grid function
load(url("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/ESL.mixture.rda"))
names(ESL.mixture)
rm(x,y)
attach(ESL.mixture)
plot(x,col=y+1)
dat <- data.frame(y=factor(y),x)
fit <- svm(factor(y)~.,data=dat,scale = FALSE, kernel="radial",cost=5)#Use CV to select cost

xgrid <- make.grid(ESL.mixture$x)
ygrid <- predict(fit, xgrid)
plot(xgrid, col = c("black","red")[as.numeric(ygrid)],pch=20,cex=0.2)
points(ESL.mixture$x,col=c("red","black"),pch=19)
