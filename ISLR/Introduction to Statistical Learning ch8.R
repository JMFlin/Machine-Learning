library(ISLR)
library(MASS)
library(tree)
library(randomForest)
library(gbm)
library(glmnet)

#Chapter 8: Tree-Based Methods

#----------------------------DECISION TREES AND PRUNING

#The tree library is used to construct classification and regression trees.

#We first use classification trees to analyze the Carseats data set. In these
#data, Sales is a continuous variable, and so we begin by recoding it as a
#binary variable. We use the ifelse() function to create a variable, called
#High, which takes on a value of Yes if the Sales variable exceeds 8, and
#takes on a value of No otherwise.
fix(Carseats)
High <- ifelse(Carseats$Sales <=8,"No","Yes")

#merge it back
Carseats <- data.frame(Carseats, High)

#We now use the tree() function to fit a classification tree in order to predict
#High using all variables but Sales. The syntax of the tree() function is quite
#similar to that of the lm() function.

tree.carseats <- tree(High~.-Sales, Carseats)

#The summary() function lists the variables that are used as internal nodes
#in the tree, the number of terminal nodes, and the (training) error rate.
summary(tree.carseats)#We see that the training error rate is 9 %.

#We use the plot() function to display the tree structure,
#and the text() function to display the node labels. The argument
#pretty=0 instructs R to include the category names for any qualitative predictors,
#rather than simply displaying a letter for each category.

plot(tree.carseats)
text(tree.carseats, pretty = 0)
#The most important indicator of Sales appears to be shelving location,
#since the first branch differentiates Good locations from Bad and Medium locations.

#If we just type the name of the tree object, R prints output corresponding
#to each branch of the tree. R displays the split criterion (e.g. Price<92.5), the
#number of observations in that branch, the deviance, the overall prediction
#for the branch (Yes or No), and the fraction of observations in that branch
#that take on values of Yes and No. Branches that lead to terminal nodes are
#indicated using asterisks.
tree.carseats

#In order to properly evaluate the performance of a classification tree on
#these data, we must estimate the test error rather than simply computing
#the training error. We split the observations into a training set and a test
#set, build the tree using the training set, and evaluate its performance on
#the test data. The predict() function can be used for this purpose. In the
#case of a classification tree, the argument type="class" instructs R to return
#the actual class prediction. This approach leads to correct predictions for
#around 71.5 % of the locations in the test data set.

train <- sample(1: nrow(Carseats), 200)
Carseats.test <- Carseats[-train,]
High.test <- High[-train]

tree.carseats <- tree(High~.-Sales, Carseats, subset=train)
tree.pred <- predict(tree.carseats, Carseats.test, type="class")
pred.table <- table(tree.pred, High.test)
pred.table
(pred.table[1,1] + pred.table[2,2])/(pred.table[1,1] + pred.table[2,2] + pred.table[1,2] + pred.table[2,1])#Classification rate

#Next, we consider whether pruning the tree might lead to improved
#results. The function cv.tree() performs cross-validation in order to 
#determine the optimal level of tree complexity; cost complexity pruning
#is used in order to select a sequence of trees for consideration. We use
#the argument FUN=prune.misclass in order to indicate that we want the
#classification error rate to guide the cross-validation and pruning process,
#rather than the default for the cv.tree() function, which is deviance. The
#cv.tree() function reports the number of terminal nodes of each tree considered
#(size) as well as the corresponding error rate and the value of the
#cost-complexity parameter used (k, which corresponds to alpha in (8.4)).

cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats
#Note that, despite the name, dev corresponds to the cross-validation error
#rate in this instance. The tree with 9 terminal nodes results in the lowest
#cross-validation error rate, with 50 cross-validation errors. We plot the error
#rate as a function of both size and k.

par(mfrow=c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")

a <- data.frame(cv.carseats$size,cv.carseats$dev)
num <- which.min(a$cv.carseats.dev)
a$cv.carseats.size[num]

#We now apply the prune.misclass() function in order to the tree to prune.
#obtain the nine-node tree.

prune.carseats <- prune.misclass(tree.carseats, best=a$cv.carseats.size[num])
par(mfrow=c(1,1))
plot(prune.carseats)
text(prune.carseats, pretty =0)

#How well does this pruned tree perform on the test data set? Once again, we apply the predict() function.

tree.pred <- predict(prune.carseats, Carseats.test, type="class")
pred.table <- table(tree.pred, High.test)
pred.table
(pred.table[1,1] + pred.table[2,2])/(pred.table[1,1] + pred.table[2,2] + pred.table[1,2] + pred.table[2,1])

#-------------------------------------REGRESSION TREES AND PRUNING

#Here we fit a regression tree to the Boston data set. First, we create a
#training set, and fit the tree to the training data.
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv~.,Boston, subset=train)
summary(tree.boston)

#Notice that the output of summary() indicates that only three of the variables
#have been used in constructing the tree. In the context of a regression
#tree, the deviance is simply the sum of squared errors for the tree. We now
#plot the tree.

plot(tree.boston)
text(tree.boston, pretty =0)

#Now we use the cv.tree() function to see whether pruning the tree will improve performance.

cv.boston <- cv.tree(tree.boston)
cv.boston
plot(cv.boston$size ,cv.boston$dev ,type="b")

a <- data.frame(cv.boston$size,cv.boston$dev)
num <- which.min(a$cv.boston.dev)
a$cv.boston.size[num]

#In this case, the most complex tree is selected by cross-validation. However,
#if we wish to prune the tree, we could do so as follows, using the
#prune.tree() function:

prune.boston <- prune.tree(tree.boston, best = a$cv.boston.size[num])
plot(prune.boston)
text(prune.boston, pretty=0)

#In keeping with the cross-validation results, we use the unpruned tree to
#make predictions on the test set.

yhat <- predict(tree.boston, newdata=Boston[-train,])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

#In other words, the test set MSE associated with the regression tree is
#25.05. The square root of the MSE is therefore around 5.005, indicating
#that this model leads to test predictions that are within around $5, 005 of
#the true median home value for the suburb.

#-------------------------------------BAGGING AND RANDOM FORESTS

#Here we apply bagging and random forests to the Boston data, using the
#randomForest package in R. The exact results obtained in this section may
#depend on the version of R and the version of the randomForest package
#installed on your computer. Recall that bagging is simply a special case of
#a random forest with m = p. Therefore, the randomForest() function can
#be used to perform both random forests and bagging.
fix(Boston)

bag.boston <- randomForest(medv~.,data=Boston, subset=train, mtry=(ncol(Boston)-1),importance =TRUE)
bag.boston

#The argument mtry=13 indicates that all 13 predictors should be considered
#for each split of the tree-in other words, that bagging should be done. How
#well does this bagged model perform on the test set?

yhat.bag <- predict(bag.boston, newdata = Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)
#The test set MSE associated with the bagged regression tree is 13.16, almost
#half that obtained using an optimally-pruned single tree. We could change
#the number of trees grown by randomForest() using the ntree argument:

bag.boston <- randomForest( medv~.,data=Boston, subset=train, mtry=13,ntree=25)
yhat.bag = predict(bag.boston , newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)

#Growing a random forest proceeds in exactly the same way, except that
#we use a smaller value of the mtry argument. By default, randomForest()
#uses p/3 variables when building a random forest of regression trees, and
#sqrt(p) variables when building a random forest of classification trees. Here we
#use mtry = 6.

rf.boston <- randomForest(medv~.,data=Boston , subset=train, mtry=6, importance =TRUE)
rf.boston#Mean of squared residusls is the OOB unbiased prediction errors
yhat.rf = predict(rf.boston, newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
#The test set MSE is lower; this indicates that random forests yielded an
#improvement over bagging in this case

#Two measures of variable importance are reported. The former is based
#upon the mean decrease of accuracy in predictions on the out of bag samples
#when a given variable is excluded from the model. The latter is a measure
#of the total decrease in node impurity that results from splits over that
#variable, averaged over all trees (this was plotted in Figure 8.9). In the
#case of regression trees, the node impurity is measured by the training
#RSS, and for classification trees by the deviance. Plots of these importance
#measures can be produced using the varImpPlot() function.

#Using the importance() function, we can view the importance of each variable.
importance(rf.boston)
varImpPlot(rf.boston, main = "Feature Importance")#First shows the decreaser in accuracy if a variable is not included
#second is how well the split happens in the trees
#Usually only show the second one to avoid confusion
par(mfrow = c(1,1))
plot(rf.boston, main = "Stabilization")

#The results indicate that across all of the trees considered in the random
#forest, the wealth level of the community (lstat) and the house size (rm)
#are by far the two most important variables.

#We fit a series of random forests with mtry 1-13 and record the errors
oob.err <- double(13)
test.err <- double(13)
for(mtry in 1:13){
  fit <- randomForest(medv~., data = Boston, subset = train, mtry = mtry, ntree = 400)
  oob.err[mtry] <- fit$mse[400]
  pred <- predict(fit, Boston[-train,])
  test.err[mtry] <- with(Boston[-train,], mean((medv-pred)^2))
}
matplot(1:mtry, cbind(test.err, oob.err), pch = 19, col = c("red","blue"), type = "b", ylab = "Mean Squared Error")#idealy the lines should line up
legend("topright", legend = c("OOB", "Test"), pch = 19, col = c("red", "blue"))
#the left hand side is a single tree and you can see how much the MSE drops. Righthand side if bagging.

#-------------------------------------------BOOSTING

#We run gbm() with the option
#distribution="gaussian" since this is a regression problem; if it were a binary
#classification problem, we would use distribution="bernoulli". The
#argument n.trees=5000 indicates that we want 5000 trees, and the option
#interaction.depth=4 limits the depth of each tree.

boost.boston <- gbm(medv~.,data=Boston[train ,], distribution="gaussian",n.trees=5000, interaction.depth=4)#ntrees,shrinkage and depth use CV. 
#boosting using depth-one trees (or stumps) leads to an additive model

#The summary() function produces a relative influence plot and also outputs
#the relative influence statistics.

summary(boost.boston)

#We see that lstat and rm are by far the most important variables. We can
#also produce partial dependence plots for these two variables. These plots
#illustrate the marginal effect of the selected variables on the response after
#integrating out the other variables. In this case, as we might expect, median
#house prices are increasing with rm and decreasing with lstat.

par(mfrow=c(1,2))
plot(boost.boston, i="rm")#As rooms go up (x-axis) the price goes up (y-axis)
plot(boost.boston, i="lstat")#the more there are lstat, the lower there are response(lower housing price in this case)
par(mfrow=c(1,1))

#We now use the boosted model to predict medv on the test set:

yhat.boost <- predict(boost.boston, newdata=Boston[-train,], n.trees=5000)
mean((yhat.boost - boston.test)^2)
#The test MSE obtained is 11-13; similar to the test MSE for random forests
#and superior to that for bagging. If we want to, we can perform boosting
#with a different value of the shrinkage parameter lambda in (8.10). The default
#value is 0.001, but this is easily modified. Here we take lambda = 0.2.

boost.boston <- gbm(medv~.,data=Boston[train,], distribution="gaussian",n.trees =5000, 
                    interaction.depth =4, shrinkage =0.2,verbose=F)

yhat.boost <- predict(boost.boston, newdata = Boston[-train,], n.trees = 5000)
mean((yhat.boost - boston.test)^2)

#In this case, using lambda = 0.2 leads to a slightly lower test MSE than lambda = 0.001

#-----------------------------APPLIED

#------------------TREE

#In the lab, we applied random forests to the Boston data using mtry=6
#and using ntree=25 and ntree=500. Create a plot displaying the test
#error resulting from random forests on this data set for a more comprehensive
#range of values for mtry and ntree. You can model your
#plot after Figure 8.10. Describe the results obtained.


#We will try a range of tt{ntree} from 1 to 500 and tt{mtry} taking typical values of p, p/2, sqrt{p}. 
#For Boston data, p = 13. We use an alternate call to tt{randomForest}$
#which takes tt{xtest}$ and tt{ytest} as additional arguments and computes test MSE on-the-fly. 
#Test MSE of all tree sizes can be obtained by accessing tt{mse} list member of tt{test} list member of the model.
set.seed(1101)

# Construct the train and test matrices
train <- sample(dim(Boston)[1], dim(Boston)[1]/2)
X.train <- Boston[train, -14]
X.test <- Boston[-train, -14]
Y.train <- Boston[train, 14]
Y.test <- Boston[-train, 14]

p <- dim(Boston)[2] - 1
p.2 <- p/2
p.sq <- sqrt(p)

rf.boston.p <- randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                           mtry = p, ntree = 500)
rf.boston.p.2 <- randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                             mtry = p.2, ntree = 500)
rf.boston.p.sq <- randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                              mtry = p.sq, ntree = 500)

plot(1:500, rf.boston.p$test$mse, col = "green", type = "l", xlab = "Number of Trees", 
     ylab = "Test MSE", ylim = c(10, 19))
lines(1:500, rf.boston.p.2$test$mse, col = "red", type = "l")
lines(1:500, rf.boston.p.sq$test$mse, col = "blue", type = "l")
legend("topright", c("m=p", "m=p/2", "m=sqrt(p)"), col = c("green", "red", "blue"), 
       cex = 1, lty = 1)

#The plot shows that test MSE for single tree is quite high (around 18). 
#It is reduced by adding more trees to the model and stabilizes around a few hundred trees. 
#Test MSE for including all variables at split is slightly higher (around 11) as compared to both using half or 
#square-root number of variables (both slightly less than 10).

#------------TREE AND RANDOM FOREST

# In the lab, a classification tree was applied to the Carseats data set after
#converting Sales into a qualitative response variable. Now we will
#seek to predict Sales using regression trees and related approaches,
#treating the response as a quantitative variable.

#Split the data set into a training set and a test set.

fix(Carseats)
train <- sample(dim(Carseats)[1], dim(Carseats)[1]/2)
Carseats.train <- Carseats[train, ]
Carseats.test <- Carseats[-train, ]

#Fit a regression tree to the training set. Plot the tree, and interpret
#the results. What test MSE do you obtain?

tree.carseats <- tree(Sales ~ ., data = Carseats.train)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty = 0)

#Use cross-validation in order to determine the optimal level of
#tree complexity. Does pruning the tree improve the test MSE?

pred.carseats <- predict(tree.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.carseats)^2)#Test MSE

cv.carseats <- cv.tree(tree.carseats, FUN = prune.tree)
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")


a <- data.frame(cv.carseats$size, cv.carseats$dev)
num <- which.min(a$cv.carseats.dev)
a$cv.carseats.size[num]

pruned.carseats <- prune.tree(tree.carseats, best = a$cv.carseats.size[num])
par(mfrow = c(1, 1))
plot(pruned.carseats)
text(pruned.carseats, pretty = 0)

pred.pruned = predict(pruned.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.pruned)^2)#test MSE
#Pruning the tree in this case increases the test MSE to 4.99.

#Use the bagging approach in order to analyze this data. What
#test MSE do you obtain? Use the importance() function to determine
#which variables are most important.

bag.carseats <- randomForest(Sales ~ ., data = Carseats.train, mtry = 10, ntree = 500, 
                            importance = T)
bag.pred <- predict(bag.carseats, Carseats.test)
mean((Carseats.test$Sales - bag.pred)^2)#test MSE
importance(bag.carseats)
#Bagging improves the test MSE to 2.58. 
#We also see that tt{Price}, tt{ShelveLoc} and tt{Age} are three most important predictors of tt{Sale}.

# Use random forests to analyze this data. What test MSE do you
#obtain? Use the importance() function to determine which variables
#are most important. Describe the effect of m, the number of
#variables considered at each split, on the error rate obtained.

rf.carseats <- randomForest(Sales ~ ., data = Carseats.train, mtry = 5, ntree = 500, 
                           importance = T)
rf.pred <- predict(rf.carseats, Carseats.test)
mean((Carseats.test$Sales - rf.pred)^2)

#In this case, random forest worsens the MSE on test set to 2.87. Changing m varies test MSE between 2.6 to 3. 
#We again see that tt{Price}, tt{ShelveLoc} and tt{Age} are three most important predictors of tt{Sale}.


#----------------TREE

#This problem involves the OJ data set which is part of the ISLR package.

# Create a training set containing a random sample of 800 observations,
#and a test set containing the remaining observations.
set.seed(1013)
train <- sample(dim(OJ)[1], 800)
OJ.train <- OJ[train, ]
OJ.test <- OJ[-train, ]

#Fit a tree to the training data, with Purchase as the response
#and the other variables except for Buy as predictors. Use the
#summary() function to produce summary statistics about the
#tree, and describe the results obtained. What is the training
#error rate? How many terminal nodes does the tree have?

oj.tree <- tree(Purchase ~ ., data = OJ.train)
summary(oj.tree)
#The tree only uses two variables: tt{LoyalCH} and tt{PriceDiff}. 
#It has 7 terminal nodes. Training error rate (misclassification error) for the tree is 0.155.

#Type in the name of the tree object in order to get a detailed
#text output. Pick one of the terminal nodes, and interpret the
#information displayed.

oj.tree
#Let's pick terminal node labeled "10)". The splitting variable at this node is tt{PriceDiff}. 
#The splitting value of this node is 0.05. There are 79 points in the subtree below this node. 
#The deviance for all points contained in region below this node is 80. 
#A * in the line denotes that this is in fact a terminal node. 
#The prediction at this node is tt{Sales} = tt{MM}. 
#About 19% points in this node have tt{CH} as value of tt{Sales}. 
#Remaining 81% points have tt{MM} as value of tt{Sales}.

# Create a plot of the tree, and interpret the results.
plot(oj.tree)
text(oj.tree, pretty =0)
#tt{LoyalCH} is the most important variable of the tree, in fact top 3 nodes contain tt{LoyalCH}. 
#If tt{LoyalCH} < 0.27, the tree predicts tt{MM}. If tt{LoyalCH} > 0.76, the tree predicts tt{CH}. 
#For intermediate values of tt{LoyalCH}, the decision also depends on the value of tt{PriceDiff}.

#Predict the response on the test data, and produce a confusion
#matrix comparing the test labels to the predicted test labels.
#What is the test error rate?
oj.pred <- predict(oj.tree, OJ.test, type = "class")
table(OJ.test$Purchase, oj.pred)

#Apply the cv.tree() function to the training set in order to
#determine the optimal tree size.
cv.oj <- cv.tree(oj.tree, FUN = prune.tree)

#Produce a plot with tree size on the x-axis and cross-validated
#classification error rate on the y-axis.
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

a <- data.frame(cv.oj$size, cv.oj$dev)
num <- which.min(cv.oj$dev)
cv.oj$size[num]

#Which tree size corresponds to the lowest cross-validated classi-fication error rate?
#Size of 6 gives lowest cross-validation error.

#Produce a pruned tree corresponding to the optimal tree size
#obtained using cross-validation. If cross-validation does not lead
#to selection of a pruned tree, then create a pruned tree with five
#terminal nodes.
oj.pruned <- prune.tree(oj.tree, best = cv.oj$size[num])

# Compare the training error rates between the pruned and unpruned
#trees. Which is higher?
summary(oj.pruned)
#Misclassification error of pruned tree is exactly same as that of original tree --- 0.155.

#Compare the test error rates between the pruned and unpruned
#trees. Which is higher?
pred.unpruned <- predict(oj.tree, OJ.test, type = "class")
misclass.unpruned <- sum(OJ.test$Purchase != pred.unpruned)
misclass.unpruned/length(pred.unpruned)

pred.pruned <- predict(oj.pruned, OJ.test, type = "class")
misclass.pruned <- sum(OJ.test$Purchase != pred.pruned)
misclass.pruned/length(pred.pruned)

#Pruned and unpruned trees have about the same test error rate of 0.189.

#---------------------BOOSTING AND BAGGING

# Remove the observations for whom the salary information is
#unknown, and then log-transform the salaries.

sum(is.na(Hitters$Salary))
Hitters <- Hitters[-which(is.na(Hitters$Salary)), ]
sum(is.na(Hitters$Salary))
Hitters$Salary <- log(Hitters$Salary)

#Create a training set consisting of the first 200 observations, and
#a test set consisting of the remaining observations.

train <- 1:200
Hitters.train <- Hitters[train, ]
Hitters.test <- Hitters[-train, ]

#Perform boosting on the training set with 1,000 trees for a range
#of values of the shrinkage parameter lambda. Produce a plot with
#different shrinkage values on the x-axis and the corresponding
#training set MSE on the y-axis.

pows <- seq(-10, -0.2, by = 0.1)
lambdas <- 10^pows
length.lambdas <- length(lambdas)
train.errors <- rep(NA, length.lambdas)
test.errors <- rep(NA, length.lambdas)
for (i in 1:length.lambdas) {
  boost.hitters <- gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", 
                      n.trees = 1000, shrinkage = lambdas[i])
  train.pred <- predict(boost.hitters, Hitters.train, n.trees = 1000)
  test.pred <- predict(boost.hitters, Hitters.test, n.trees = 1000)
  train.errors[i] <- mean((Hitters.train$Salary - train.pred)^2)
  test.errors[i] <- mean((Hitters.test$Salary - test.pred)^2)
}

plot(lambdas, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", 
     col = "blue", pch = 20)

#Produce a plot with different shrinkage values on the x-axis and
#the corresponding test set MSE on the y-axis.

plot(lambdas, test.errors, type = "b", xlab = "Shrinkage", ylab = "Test MSE", 
     col = "red", pch = 20)

min(test.errors)
lambdas[which.min(test.errors)]
#Minimum test error is obtained at lambda = lambdas[which.min(test.errors)].

#Compare the test MSE of boosting to the test MSE that results
#from applying two of the regression approaches seen in Chapters 3 and 6.

lm.fit <- lm(Salary ~ ., data = Hitters.train)
lm.pred <- predict(lm.fit, Hitters.test)
mean((Hitters.test$Salary - lm.pred)^2)

x <- model.matrix(Salary ~ ., data = Hitters.train)
y <- Hitters.train$Salary
x.test <- model.matrix(Salary ~ ., data = Hitters.test)
lasso.fit <- glmnet(x, y, alpha = 1)
lasso.pred <- predict(lasso.fit, s = 0.01, newx = x.test)
mean((Hitters.test$Salary - lasso.pred)^2)
#Both linear model and regularization like Lasso have higher test MSE than boosting.

#Which variables appear to be the most important predictors in the boosted model?
boost.best <- gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", 
                 n.trees = 1000, shrinkage = lambdas[which.min(test.errors)])
summary(boost.best)
#tt{CAtBat}, tt{CRBI} and tt{CWalks} are three most important variables in that order.

#Now apply bagging to the training set. What is the test set MSE for this approach?
rf.hitters <- randomForest(Salary ~ ., data = Hitters.train, ntree = 500, mtry = (ncol(Hitters)-1))
rf.pred <- predict(rf.hitters, Hitters.test)
mean((Hitters.test$Salary - rf.pred)^2)
#Test MSE for bagging is about 0.23, which is slightly lower than the best test MSE for boosting.

#------------BOOSTING, KNN, LOG REG. The error messages are fine.

#Create a training set consisting of the first 1,000 observations,
#and a test set consisting of the remaining observations.
fix(Caravan)
train <- 1:1000
Caravan$Purchase <- ifelse(Caravan$Purchase == "Yes", 1, 0)
Caravan.train <- Caravan[train, ]
Caravan.test <- Caravan[-train, ]

#Fit a boosting model to the training set with Purchase as the
#response and the other variables as predictors. Use 1,000 trees,
#and a shrinkage value of 0.01. Which predictors appear to be
#the most important?
boost.caravan <- gbm(Purchase ~ ., data = Caravan.train, n.trees = 1000, shrinkage = 0.01, 
                    distribution = "bernoulli")
summary(boost.caravan)
#tt{PPERSAUT}, tt{MKOOPKLA} and tt{MOPLHOOG} are three most important variables in that order.

#Use the boosting model to predict the response on the test data.
#Predict that a person will make a purchase if the estimated probability
#of purchase is greater than 20 %. Form a confusion matrix.
#What fraction of the people predicted to make a purchase
#do in fact make one? How does this compare with the results
#obtained from applying KNN or logistic regression to this data set?

boost.prob <- predict(boost.caravan, Caravan.test, n.trees = 1000, type = "response")
boost.pred <- ifelse(boost.prob > 0.2, 1, 0)
pred.table <- table(Caravan.test$Purchase, boost.pred)
pred.table
pred.table[2,2]/(pred.table[2,2] + pred.table[1,2])#About 20% of people predicted to make purchase actually end up making one.

lm.caravan <- glm(Purchase ~ ., data = Caravan.train, family = binomial)
lm.prob <- predict(lm.caravan, Caravan.test, type = "response")
lm.pred <- ifelse(lm.prob > 0.2, 1, 0)
pred.table <- table(Caravan.test$Purchase, lm.pred)
pred.table
pred.table[2,2]/(pred.table[2,2] + pred.table[1,2])#About 14% of people predicted to make purchase using logistic regression actually end up making one. This is lower than boosting.

knn.pred <- knn(Caravan.train, Caravan.test, Caravan$Purchase[train], k=1)
pred.table <- table(knn.pred, Caravan$Purchase[-train])
pred.table
pred.table[2,2]/(pred.table[2,2] + pred.table[1,2])#About 10% of people predicted to make purchase using KNN actually end up making one. This is lower than boosting.

#-----------------------------BOOSTING, BAGGING, RANDOM FOREST AND LOG RES

#Apply boosting, bagging, and random forests to a data set of your
#choice. Be sure to fit the models on a training set and to evaluate their
#performance on a test set. How accurate are the results compared
#to simple methods like linear or logistic regression? Which of these
#approaches yields the best performance?

#Weekly stock data from ISLR
fix(Weekly)
summary(Weekly)

train <- sample(nrow(Weekly), 2/3 * nrow(Weekly))
test <- -train

#LOGISTIC REGRESSION
glm.fit <- glm(Direction ~ . - Year - Today, data = Weekly[train, ], family = "binomial")
glm.probs <- predict(glm.fit, newdata = Weekly[test, ], type = "response")
glm.pred <- rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
table(glm.pred, Weekly$Direction[test])
mean(glm.pred != Weekly$Direction[test])#validation set test error rate

#BOOSTING
Weekly$BinomialDirection <- ifelse(Weekly$Direction == "Up", 1, 0)
boost.weekly <- gbm(BinomialDirection ~ . - Year - Today - Direction, data = Weekly[train, ], distribution = "bernoulli", n.trees = 5000)
yhat.boost <- predict(boost.weekly, newdata = Weekly[test, ], n.trees = 5000)
yhat.pred <- rep(0, length(yhat.boost))
yhat.pred[yhat.boost > 0.5] = 1
table(yhat.pred, Weekly$BinomialDirection[test])
mean(yhat.pred != Weekly$BinomialDirection[test])#validation set test error rate

#BAGGING
Weekly <- Weekly[, !(names(Weekly) %in% c("BinomialDirection"))]
bag.weekly <- randomForest(Direction ~ . - Year - Today, data = Weekly, subset = train, mtry = 6)
yhat.bag <- predict(bag.weekly, newdata = Weekly[test, ])
table(yhat.bag, Weekly$Direction[test])
mean(yhat.bag != Weekly$Direction[test])#validation set test error rate

#RANDOM FOREST
rf.weekly <- randomForest(Direction ~ . - Year - Today, data = Weekly, subset = train, mtry = 2)
yhat.bag <- predict(rf.weekly, newdata = Weekly[test, ])
table(yhat.bag, Weekly$Direction[test])
mean(yhat.bag != Weekly$Direction[test])#validation set test error rate

#The lowest number is the best because it is the validation set test error rate.