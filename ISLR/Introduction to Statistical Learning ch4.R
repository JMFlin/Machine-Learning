library(ISLR)
library(MASS)#functions
library(class)#functions for knn classification

###Chapter 4: Classification


#-------------------LOGISTIC REGRESSION redo by Hung style
#This data set consists of
#percentage returns for the S&P 500 stock index over 1,250 days, from the
#beginning of 2001 until the end of 2005. For each date, we have recorded
#the percentage returns for each of the five previous trading days, Lag1
#through Lag5. We have also recorded Volume (the number of shares traded
#on the previous day, in billions), Today (the percentage return on the date
#in question) and Direction (whether the market was Up or Down on this date).
smarket <- Smarket
names(Smarket)
pairs(Smarket, col = Smarket$Direction)
cor(Smarket[,-9])
#Little correlation between todays and previous days returns
plot(Smarket$Volume)
#avg number of shares traded has been increasing

#Next, we will fit a logistic regression model in order to predict Direction
#using Lag1 through Lag5 and Volume.

model.fit <- glm(Direction ~Lag1 + Lag2 + Lag3+Lag4+Lag5+Volume, data = Smarket, family = "binomial")
summary(model.fit)#No clear evidence of associations (large p-values)
exp(cbind(OR = coef(model.fit), confint(model.fit))) #confint shoulnd't include 1.
coef(model.fit)

contrasts(Smarket$Direction)#Up is coded as 1

model.probs <- predict(model.fit, type = "response")
model.probs[1:10]#These values correspond to market going up because up is coded as 1

#we will first create a vector corresponding
#to the observations from 2001 through 2004. We will then use this vector
#to create a held out data set of observations from 2005.
train <- Smarket$Year<2005
Smarket.2005 <- Smarket[!train,]
direction.2005 <- Smarket$Direction[!train]

#We now fit a logistic regression model using only the subset of the observations
#that correspond to dates before 2005, using the subset argument.
#We then obtain predicted probabilities of the stock market going up for
#each of the days in our test set-that is, for the days in 2005.
model.fit <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)

model.probs <- predict(model.fit, Smarket.2005, type = "response")


#Notice that we have trained and tested our model on two completely separate
#data sets: training was performed using only the dates before 2005,
#and testing was performed using only the dates in 2005. Finally, we compute
#the predictions for 2005 and compare them to the actual movements
#of the market over that time period.

model.pred <- rep("Down", 252)
model.pred[model.probs > 0.5] = "Up" #Can use ifelse()

pred.table <- table(model.pred, direction.2005)
1-sum(diag(pred.table))/sum(pred.table) #This is worse than a random guess!

#Try again and remove the highest p-values

model.fit <- glm(Direction~Lag1+Lag2,
                 data=Smarket ,family="binomial", subset=train)

model.probs <- predict(model.fit, Smarket.2005, type = "response")
model.pred <- rep("Down", 252)#can use ifelse
model.pred[model.probs > 0.5] = "Up"#can use ifelse
pred.table <- table(model.pred, direction.2005)
1-sum(diag(pred.table))/sum(pred.table) #Missclassification rate

sum(diag(pred.table))/sum(pred.table)#Correctly predicted

Accuracy <- pred.table[2,2]/(pred.table[2,2] + pred.table[2,1])
Accuracy # on days when logistic regression predicts an increase in the market, it has a 58% accuracy rate.

#Suppose that we want to predict the returns associated with particular
#values of Lag1 and Lag2. In particular, we want to predict Direction on a
#day when Lag1 and Lag2 equal 1.2 and 1.1, respectively, and on a day when
#they equal 1.5 and -0.8.

predict(model.fit, newdata = data.frame(Lag1=c(1.2 ,1.5),
                                      Lag2=c(1.1,-0.8)),type="response")

#----------------------LDA (in MASS file)161

lda.fit <- lda(Direction ~ Lag1+Lag2 ,data=Smarket ,subset=train)#train is Year<2005 later on we predict for 2005!
lda.fit
plot(lda.fit)#Not much difference in Down or Up. Hard to predict then.

#in other words,
#49.2 % of the training observations correspond to days during which the
#market went down. It also provides the group means; these are the average
#of each predictor within each class, and are used by LDA as estimates
#of uk. These suggest that there is a tendency for the previous 2 days'
#returns to be negative on days when the market increases, and a tendency
#for the previous days' returns to be positive on days when the market
#declines.

#test this with lindas method with 5 sma and 2 days down while above the sma
lda.pred <- predict(lda.fit, Smarket.2005)
names(lda.pred)
lda.class <- lda.pred$class
table(lda.class, direction.2005)
mean(lda.class == direction.2005)#Correctly predicted


#Notice that the posterior probability output by the model corresponds to
#the probability that the market will decrease:
sum(lda.pred$posterior[,1] >=0.5) #Threshold of 50% of observations
sum(lda.pred$posterior[,1] <0.5)

sum(lda.pred$posterior[,1] >=0.5)/sum(lda.pred$posterior[,1])

#-------------------------QDA
qda.fit=qda(Direction~Lag1+Lag2 ,data=Smarket ,subset=train)
qda.fit#output contains group means

qda.class <- predict(qda.fit, Smarket.2005)$class
table(qda.class, direction.2005)
sum(diag(pred.table))/sum(pred.table)#Prediction are accurate 60% of the time
#We recommend to test this methods performance on a bigger test set before putting money into it

#------------------------KNN
train.X <- cbind(Smarket$Lag1, Smarket$Lag2)[train,]
test.X <- cbind(Smarket$Lag1, Smarket$Lag2)[!train,]
train.Direction <- Smarket$Direction[train]

knn.pred <- knn(train.X, test.X, train.Direction, k=1)
pred.table <- table(knn.pred, direction.2005)
sum(diag(pred.table))/sum(pred.table)#same result as flipping a coin

#QDA is best for this data set

#------------------------KNN for insurance data
attach(Caravan)
summary(Purchase)
#Large scale variables will have big effects.
#A good way to handle this problem is to standardize the data so that all standardize
#variables are given a mean of zero and a standard deviation of one. Then
#all variables will be on a comparable scale. 
#The scale() function does just scale() this. In standardizing the data, we exclude column 86, because that is the
#qualitative Purchase variable

standardized.X <- scale(Caravan [,-86])
nrow(standardized.X)

#We now split the observations into a test set, containing the first 1,000
#observations, and a training set, containing the remaining observations.
#We fit a KNN model on the training data using K = 1, and evaluate its
#performance on the test data.

test <- 1:1000
train.X <- standardized.X[-test,]
test.X <- standardized.X[test,]
train.Y <- Caravan$Purchase[-test]
test.Y <- Caravan$Purchase[test]
set.seed(1)
knn.pred <- knn(train.X,test.X,train.Y,k=1)
mean(test.Y!=knn.pred)#The KNN error rate on the
#1,000 test observations is just under 12 %. At first glance, this may appear
#to be fairly good.
mean(test.Y!="No")#If the company tries to sell insurance to a random
#selection of customers, then the success rate will be only 6 %

pred.table <- table(knn.pred ,test.Y)
pred.table[2,2]/(pred.table[2,2] + pred.table[2,1])#Among 77 such
#customers, 9, or 11.7 %, actually do purchase insurance. This is double the
#rate that one would obtain from random guessing.

knn.pred=knn(train.X,test.X,train.Y,k=3)
pred.table <- table(knn.pred ,test.Y)
pred.table[2,2]/(pred.table[2,2] + pred.table[2,1])#success rate is now 19%

knn.pred=knn(train.X,test.X,train.Y,k=5)
pred.table <- table(knn.pred ,test.Y)
pred.table[2,2]/(pred.table[2,2] + pred.table[2,1])#success rate is 26,7%
#This is over four times the rate that results from random guessing.

#Let's try a logistic regression model

#If we use 0.5 as the predicted probability cut-off for the classifier, then
#we have a problem: only seven of the test observations are predicted to
#purchase insurance. Even worse, we are wrong about all of these! However,
#we are not required to use a cut-off of 0.5. If we instead predict a purchase
#any time the predicted probability of purchase exceeds 0.25, we get much
#better results: we predict that 33 people will purchase insurance, and we
#are correct for about 33 % of these people. This is over five times better
#than random guessing!

glm.fit <-glm(Purchase~.,data=Caravan ,family=binomial ,
            subset=-test)#the warning is ok
glm.probs <- predict(glm.fit ,Caravan [test ,], type="response")
glm.pred <- rep("No",1000)
glm.pred[glm.probs >.5] <- "Yes"
table(glm.pred ,test.Y)

glm.pred <- rep("No",1000)
glm.pred[glm.probs >.25] <- "Yes"
table(glm.pred ,test.Y)
pred.table <- table(glm.pred ,test.Y)
pred.table[2,2]/(pred.table[2,1] + pred.table[2,2])


#----------------------LDA, QDA, KNN
attach(Weekly)
pairs(Weekly)
cor(Weekly[,-9])

glm.fit <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
                               data=Weekly,
                               family=binomial)
summary(glm.fit)
#Lag2 is stat signif

glm.probs <- predict(glm.fit, type="response")
glm.pred <- rep("Down", length(glm.probs))
glm.pred[glm.probs>.5] = "Up"
pred.table <- table(glm.pred, Direction)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])##overall fraction of correct predictions

pred.table[2,2]/(pred.table[1,2] + pred.table[2,2])#Weeks the market goes up the logistic regression is right most of the time

pred.table[1,1]/(pred.table[1,1] + pred.table[2,1])#Weeks the market goes down the logistic regression is wrong most of the time

#Fit the logistic regression model using a training data period from 1990
#to 2008 with Lag 2 only predictor. Compute the overall fraction of correct
#predictions for 2009-2010(held out data)
train <- (Year < 2009)
Weekly.0910 <- Weekly[!train,]
glm.fit <- glm(Direction ~Lag2, data=Weekly, family=binomial, subset=train)
glm.probs <- predict(glm.fit, Weekly.0910, type = "response")
glm.pred <- rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] <- "Up"
Direction.0910 = Direction[!train]
pred.table <- table(glm.pred, Direction.0910)
mean(glm.pred == Direction.0910)#overall fraction of correct predictions
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

#Same thing but for LDA
lda.fit <- lda(Direction ~Lag2, data=Weekly, subset=train)
lda.pred <- predict(lda.fit, Weekly.0910)
pred.table <- table(lda.pred$class, Direction.0910)
mean(lda.pred$class == Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

#Same thing but for QDA
qda.fit <- qda(Direction~Lag2, data=Weekly, subset=train)
qda.class <- predict(qda.fit, Weekly.0910)$class
pred.table <- table(qda.class, Direction.0910)#A correctness of 58.7% even though it picked Up the whole time!
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

#Same thing but for KNN
train.X <- as.matrix(Lag2[train])
test.X <- as.matrix(Lag2[!train])
train.Direction <- Direction[train]
knn.pred <- knn(train.X, test.X, train.Direction, k=1)
pred.table <- table(knn.pred, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

#Logistic regression and LDA methods provide similar test error rates.

#Now try Lag1:Lag2. You could try experimenting with other transformations and interactions
#Logistic Regression
glm.fit <- glm(Direction ~ Lag1:Lag2, data = Weekly, family = binomial, subset = train)
glm.probs <- predict(glm.fit, Weekly.0910, type = "response")
glm.pred <- rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] <- "Up"
Direction.0910 <- Direction[!train]
pred.table <- table(glm.pred, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions
#LDA
lda.fit <- lda(Direction ~ Lag1:Lag2, data = Weekly, subset = train)
lda.pred <- predict(lda.fit, Weekly.0910)
pred.table <- table(lda.pred$class, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions
#QDA
qda.fit <- qda(Direction~Lag2:Lag1, data=Weekly, subset=train)
qda.class <- predict(qda.fit, Weekly.0910)$class
pred.table <- table(qda.class, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions
#KNN k = 10
knn.pred <- knn(train.X, test.X, train.Direction, k = 10)
pred.table <- table(knn.pred, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions
#KNN k = 100
knn.pred <- knn(train.X, test.X, train.Direction, k = 100)
pred.table <- table(knn.pred, Direction.0910)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

#Out of these permutations, the original LDA and logistic regression have better performance in terms of test error rate.

#---------------------------------------
attach(Auto)

mpg01 <- ifelse(mpg > median(mpg),1,0)
auto <- cbind(Auto, mpg01)
cor(auto[,-9])
pairs(auto)#which features seems likely to predict if mpg is going to be above it's median

train <- (year %% 2 == 0) # if the year is even
test <- !train
auto.train <- auto[train,]
auto.test <- auto[test,]
mpg01.test <- mpg01[test]

#LDA 
lda.fit <- lda(mpg01~cylinders+weight+displacement+horsepower, data = auto, subset = train)
lda.pred <- predict(lda.fit, auto.test)
pred.table <- table(lda.pred$class, mpg01.test)
mean(lda.pred$class != mpg01.test)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

1-(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                       + pred.table[1,1] + pred.table[1,2])#test error rate


#QDA
qda.fit <- qda(mpg01~cylinders+weight+displacement+horsepower,
               data=auto, subset=train)
qda.pred <- predict(qda.fit, auto.test)
pred.table <- table(qda.pred$class, mpg01.test)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions

1-(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                       + pred.table[1,1] + pred.table[1,2])#test error rate

#Logistic Regression
glm.fit <- glm(mpg01~cylinders+weight+displacement+horsepower,
               data=auto, family = binomial, subset=train)
glm.probs <- predict(glm.fit, auto.test, type = "response")
glm.pred <- rep(0, length(glm.probs)) 
glm.pred <- ifelse(glm.probs > 0.5,1,0)
mean(glm.pred != mpg01.test)#12.1% test error rate.
pred.table <- table(glm.pred, mpg01.test)
(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                     + pred.table[1,1] + pred.table[1,2])#overall fraction of correct predictions
1-(pred.table[1,1] + pred.table[2,2])/(pred.table[2,1] + pred.table[2,2]
                                       + pred.table[1,1] + pred.table[1,2])#test error rate

#KNN
train.X <- cbind(cylinders, weight, displacement, horsepower)[train,]
test.X <- cbind(cylinders, weight, displacement, horsepower)[test,]
train.mpg01 <- mpg01[train]
knn.pred <- knn(train.X, test.X, train.mpg01, k=1)
mean(knn.pred != mpg01.test)
knn.pred <- knn(train.X, test.X, train.mpg01, k=10)
mean(knn.pred != mpg01.test)
knn.pred = knn(train.X, test.X, train.mpg01, k=100)
mean(knn.pred != mpg01.test)
#k=1, 15.4% test error rate. k=10, 16.5% test error rate. k=100, 14.3% test error rate. K of 100 seems to perform the best. 
#100 nearest neighbors.

#-----------------------------QDA, LDA, KNN
attach(Boston)
#Create test and training sets
crime01[crim>median(crim)] = 1
Boston <- data.frame(Boston, crime01)

train <- 1:(dim(Boston)[1]/2)
test <- (dim(Boston)[1]/2+1):dim(Boston)[1]
Boston.train <- Boston[train,]
Boston.test <- Boston[test,]
crime01.test <- crime01[test]

#Logistic Regression
glm.fit <- glm(crime01~-crime01-crim, 
               data=Boston, family=binomial, subset=train)
glm.probs <- predict(glm.fit, data = Boston.test, type = "response")
glm.pred <- rep(0, length(glm.probs))
glm.pred[glm.pred > 0.5] <- 1
mean(glm.pred != crime01.test)#test error rate. Something is wrong

#LDA
lda.fit <- lda(crime01 ~. -crime01-crim, data=Boston, subset=train)
lda.pred <- predict(lda.fit, Boston.test)
mean(lda.pred$class != crime01.test)#test error rate

lda.fit = lda(crime01~.-crime01-crim-chas-tax, data=Boston, subset=train)
lda.pred = predict(lda.fit, Boston.test)
mean(lda.pred$class != crime01.test)#test error rate

#KNN
train.X <- cbind(zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black,
                lstat, medv)[train,]
test.X <- cbind(zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black,
               lstat, medv)[test,]
train.crime01 <- crime01[train]

knn.pred <- knn(train.X, test.X, train.crime01, k=1)
mean(knn.pred != crime01.test)# test rate error very high

knn.pred <- knn(train.X, test.X, train.crime01, k=10)
mean(knn.pred != crime01.test)

knn.pred <- knn(train.X, test.X, train.crime01, k=100)
mean(knn.pred != crime01.test)

#try another subset of predictors for knn 10

train.X <- cbind(zn, nox, rm, dis, rad, ptratio, black, medv)[train,]
test.X <- cbind(zn, nox, rm, dis, rad, ptratio, black, medv)[test,]
knn.pred <- knn(train.X, test.X, train.crime01, k=10)
mean(knn.pred != crime01.test)#higher test error



