###################################################################
## Code for the useR 2016 tutorial "Never Tell Me the Odds! Machine 
## Learning with Class Imbalances" by Max Kuhn
## 
## Slides and this code can be found at
##    https://github.com/topepo/useR2016
## 
## 
## Data are at: https://github.com/rudeboybert/JSE_OkCupid
##              https://github.com/topepo/useR2016
##
## OkC data are created in a different file
##

#Measuring performance with imbalances
#Sampling methods for combating class imblanaces

#In classification problems, a disparity in the frequencies of the observed classes can have a significant negative impact on model fitting. 
#One technique for resolving such a class imbalance is to subsample the training data in a manner that mitigates the issues. 
#Examples of sampling methods for this purpose are:

#down-sampling: randomly subset all the classes in the training set so that their class frequencies match the least prevalent class. 
#For example, suppose that 80% of the training set samples are the first class and the remaining 20% are in the second class. 

#Down-sampling would randomly sample the first class to be the same size as the second class (so that only 40% of the total training set is used to fit the model). 
#caret contains a function (downSample) to do this.

#up-sampling: randomly sample (with replacement) the minority class to be the same size as the majority class. caret contains a function (upSample) to do this.

#hybrid methods: techniques such as SMOTE and ROSE down-sample the majority class and synthesize new data points in the minority class.
#SMOTE (Synthetic minority over-sampling technique) nearst neighbors
#ROSE (Randomly over-sampling examples) kernel density

library(AppliedPredictiveModeling)
library(ggplot2)
library(caret)
library(rpart)
library(partykit)
library(pROC)
library(ROSE)
library(C50)
library(kernlab)
library(ggthemes)
library(plyr)
library(ROCR)

## Slide 22 "Example Data - Electronic Medical Records"

load("emr.RData")
emr <- emr[sample(nrow(emr), size = 500, replace = FALSE),]#making the dataset smaller so it runs faster

str(emr, list.len = 20)

## Slide 23 "Example Data - Electronic Medical Records"

set.seed(1732)
emr_ind <- createDataPartition(emr$Class, p = 2/3, list = FALSE)
emr_train <- emr[ emr_ind,]
emr_test  <- emr[-emr_ind,]

mean(emr_train$Class == "event")
mean(emr_test$Class == "event")

table(emr_train$Class)
table(emr_test$Class)


## Slide 40 and 43 "A Single Shallow Tree"

rp1 <- rpart(Class ~ ., data = emr_train, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp1))

## Slide 44 "A Single Shallow Tree (Bootstrapped)"

set.seed(9595)
dat2 <- emr_train[sample(1:nrow(emr_train), nrow(emr_train), replace = TRUE),]
rp2 <- rpart(Class ~ ., data = dat2, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp2))

## Slide 45 "A Single Shallow Tree (Bootstrapped)"

set.seed(1976)
dat3 <- emr_train[sample(1:nrow(emr_train), nrow(emr_train), replace = TRUE),]
rp3 <- rpart(Class ~ ., data = dat3, control = rpart.control(maxdepth = 3, cp = 0))
plot(as.party(rp3))

## Slide 47 "Random Forests with the EMR Data"

## on Windows, try the doParallel package
## **if** your computer has multiple cores and sufficient memory
#cmd -> WMIC CPU Get DeviceID,NumberOfCores,NumberOfLogicalProcessors

ctrl <- trainControl(method = "repeatedcv",#cv
                     repeats = 1, # this does 1 repeats of 10-fold corss validation
                     classProbs = TRUE,#used to calculate the area under ROC, sensitivity and specificity. Only for 2 class problems.
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)
emr_grid <- data.frame(mtry = c(2,5))#c(1:15, (4:9)*5)

set.seed(1537)
rf_emr_mod <- train(Class ~ ., 
                    data = emr_train,
                    method = "rf",
                    metric = "ROC",#evaluation is done with Area Under Curve
                    tuneGrid = emr_grid,
                    ntree = 1000,
                    trControl = ctrl)

#Confusion matrix and associated statistics for model fit:

rfClasses <- predict(rf_emr_mod, emr_test)

sensitivity(rfClasses, emr_test$Class)#Sensitivity: given that a result is truly an event, what is the probability that the model will predict an event result? True positive
specificity(rfClasses, emr_test$Class)#Specificity: given that a result is truly not an event, what is the probabiliy that the model will predict a negative result? True negative

confusionMatrix(data = rfClasses, emr_test$Class)
#The "no--information rate" is the largest proportion of the observed classes.
#A hypothesis test is also computed to evaluate whether the overall accuracy rate is greater than the rate of the largest class.
#Also, the prevalence of the "positive event" is computed from the data (unless passed in as an argument), 
#the detection rate (the rate of true events also predicted to be events) and the detection prevalence (the prevalence of predicted events). 
#https://topepo.github.io/caret/measuring-performance.html

postResample(rfClasses, emr_test$Class)
#The Kappa statistic (or value) is a metric that compares an Observed Accuracy with an Expected Accuracy (random chance).
#Landis and Koch considers 0-0.20 as slight, 0.21-0.40 as fair, 0.41-0.60 as moderate, 0.61-0.80 as substantial, and 0.81-1 as almost perfect. 
#Fleiss considers kappas > 0.75 as excellent, 0.40-0.75 as fair to good, and < 0.40 as poor.

## Slide 50 "Random Forest Results - EMR Example"

ggplot(rf_emr_mod)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_mod, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

roc.data <- data.frame(Model='Random Forest',y=rf.ROC$sensitivities, x=1-rf.ROC$specificities)

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


#KS

pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")

ks <- data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]), Model='Random Forest')

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val <- data.frame(normal = max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]])))
#K-S or Kolmogorov-Smirnov chart measures performance of classification models. 
#More accurately, K-S is a measure of the degree of separation between the positive and negative distributions. 
#The K-S is 100, if the scores partition the population into two separate groups in which one group contains all the positives and the other all the negatives.

#On the other hand, If the model cannot differentiate between positives and negatives, 
#then it is as if the model selects cases randomly from the population. 
#The K-S would be 0. In most classification models the K-S will fall between 0 and 100, 
#and that the higher the value the better the model is at separating the positive from negative cases.

# Recall-Precision curve (Also known as Lift Chart) 
RP.perf <- performance(pred, "prec", "rec")

perf.data <- data.frame(Model='Random Forest',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]])

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#If your question is: "How meaningful is a positive result from my classifier given the baseline probabilities of my problem?", use a PR curve. 
#If your question is: "How well can this classifier be expected to perform in general, at a variety of different baseline probabilities?", go with a ROC curve.

#Accurary
f.perf <- performance(pred, "acc")
f.data <- data.frame(Model='Random Forest',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]])

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


## Slide 59 "Down-Sampling - EMR Data"

down_ctrl <- ctrl
down_ctrl$sampling <- "down"
set.seed(1537)
rf_emr_down <- train(Class ~ ., 
                     data = emr_train,
                     method = "rf",
                     metric = "ROC",#evaluation is done with Area Under Curve
                     tuneGrid = emr_grid,
                     ntree = 1000,
                     trControl = down_ctrl)

rfClasses_down <- predict(rf_emr_down, emr_test)
confusionMatrix(data = rfClasses_down, emr_test$Class)
postResample(rfClasses_down, emr_test$Class)
sensitivity(rfClasses_down, emr_test$Class)

## Slide 60 "Down-Sampling - EMR Data"

ggplot(rf_emr_down)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_down, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Down-Sampling',x=1-rf.ROC$specificities, y=rf.ROC$sensitivities))

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


#KS
pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Down-Sampling'))

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val$down <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Down-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Down-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 



## Slide 63 "Internal Down-Sampling - EMR Data"

set.seed(1537)
rf_emr_down_int <- train(Class ~ ., 
                         data = emr_train,
                         method = "rf",
                         metric = "ROC",
                         ntree = 1000,
                         tuneGrid = emr_grid,
                         trControl = ctrl,
                         ## These are passed to `randomForest`
                         strata = emr_train$Class,
                         sampsize = rep(sum(emr_train$Class == "event"), 2))

#Confusion matrix and associated statistics for model fit:

rfClasses_down_int <- predict(rf_emr_down_int, emr_test)
confusionMatrix(data = rfClasses_down_int, emr_test$Class)
postResample(rfClasses_down_int, emr_test$Class)
sensitivity(rfClasses_down_int, emr_test$Class)

## Slide 64 "Internal Down-Sampling - EMR Data"

ggplot(rf_emr_down_int)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_down_int, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=1-rf.ROC$specificities, y=rf.ROC$sensitivities))

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


#KS
pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Internal Down-Sampling'))

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val$down_int <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 



## Slide 67 "Up-Sampling - EMR Data"

up_ctrl <- ctrl
up_ctrl$sampling <- "up"
set.seed(1537)
rf_emr_up <- train(Class ~ ., 
                   data = emr_train,
                   method = "rf",
                   tuneGrid = emr_grid,
                   ntree = 1000,
                   metric = "ROC",
                   trControl = up_ctrl)

#Confusion matrix and associated statistics for model fit:

rfClasses_up <- predict(rf_emr_up, emr_test)
confusionMatrix(data = rfClasses_up, emr_test$Class)
postResample(rfClasses_up, emr_test$Class)
sensitivity(rfClasses_up, emr_test$Class)

## Slide 68 "Up-Sampling - EMR Data"

ggplot(rf_emr_up)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_up, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Up-Sampling',x=1-rf.ROC$specificities, y=rf.ROC$sensitivities))

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#KS
pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Up-Sampling'))

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val$up <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Up-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Up-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


## Slide 73 "SMOTE - EMR Data"

smote_ctrl <- ctrl
smote_ctrl$sampling <- "smote"
set.seed(1537)
rf_emr_smote <- train(Class ~ ., 
                      data = emr_train,
                      method = "rf",
                      tuneGrid = emr_grid,
                      ntree = 1000,
                      metric = "ROC",
                      trControl = smote_ctrl)

#Confusion matrix and associated statistics for model fit:

rfClasses_smote <- predict(rf_emr_smote, emr_test)
confusionMatrix(data = rfClasses_smote, emr_test$Class)
postResample(rfClasses_smote, emr_test$Class)
sensitivity(rfClasses_smote, emr_test$Class)

## Slide 74 "SMOTE - EMR Data"

ggplot(rf_emr_smote)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_smote, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n SMOTE',x=1-rf.ROC$specificities, y=rf.ROC$sensitivities))

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#KS
pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n SMOTE'))

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val$smote <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec")

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n SMOTE',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n SMOTE',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


##"ROSE - EMR Data"

rose_ctrl <- ctrl
rose_ctrl$sampling <- "rose"
set.seed(1537)
rf_emr_rose <- train(Class ~ ., 
                     data = emr_train,
                     method = "rf",
                     tuneGrid = emr_grid,
                     ntree = 1000,
                     metric = "ROC",
                     trControl = rose_ctrl)

#Confusion matrix and associated statistics for model fit:

rfClasses_rose <- predict(rf_emr_rose, emr_test)
confusionMatrix(data = rfClasses_rose, emr_test$Class)
postResample(rfClasses_rose, emr_test$Class)
sensitivity(rfClasses_rose, emr_test$Class)

##"ROSE - EMR Data"

ggplot(rf_emr_rose)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_rose, emr_test,type="prob")

rf.ROC <- roc(predictor=rf.probs$event,
               response=emr_test$Class,
               levels=rev(levels(emr_test$Class)))

roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n ROSE',x=1-rf.ROC$specificities, y=rf.ROC$sensitivities))

q <- ggplot(data=roc.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#KS
pred <- prediction(rf.probs$noevent, emr_test$Class)
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]), Model='Random Forest\n ROSE'))

d <- ggplot(ks, aes(x=seq_along(ks[,1]), y=ks[,1], group = Model, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val$rose <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n ROSE',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n ROSE',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


## Slide 75 "SMOTE - EMR Data"

emr_test_pred <- data.frame(Class = emr_test$Class)
emr_test_pred$normal <- predict(rf_emr_mod, emr_test, type = "prob")[, "event"]
emr_test_pred$down <- predict(rf_emr_down, emr_test, type = "prob")[, "event"]
emr_test_pred$down_int <- predict(rf_emr_down_int, emr_test, type = "prob")[, "event"]
emr_test_pred$up <- predict(rf_emr_up, emr_test, type = "prob")[, "event"]
emr_test_pred$smote <- predict(rf_emr_smote, emr_test, type = "prob")[, "event"]
emr_test_pred$rose <- predict(rf_emr_rose, emr_test, type = "prob")[, "event"]

get_auc <- function(pred, ref){
  auc(roc(ref, pred, levels = rev(levels(ref))))
}

apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)#AUC
tst <- data.frame(apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class))
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value,fill=names)) +
  geom_bar(stat='identity') + ylab(label="AUC")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


2*apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)-1#Gini Coefficient/Ratio Above 60% corresponds to a good model
tst <- data.frame(2*apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)-1)
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value,fill=names)) +
  geom_bar(stat='identity') + ylab(label="Gini Coefficient")+geom_hline(yintercept = .60, linetype = "dashed")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val#It is the maximum difference between the cumulative true positive rate and the cumulative false positive rate
tst <- data.frame(t(ks.val))
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value,fill=names)) +
  geom_bar(stat='identity') + ylab(label="Kolmogorov-Smirnov Maximums")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


resamps <- resamples(list(normal = rf_emr_mod, down = rf_emr_down, down_int = rf_emr_down_int,
                          up=rf_emr_up,smote=rf_emr_smote,rose=rf_emr_rose))
summary(resamps)

trellis.par.set(caretTheme())
# boxplots of results
bwplot(resamps)
# dot plots of results
dotplot(resamps)

###################################################################

## Slide 25 "Example Data - OKCupid"

load("okc.RData") ## create this using the file "okc_data.R"
str(okc, list.len = 20, vec.len = 2)
okc <- okc[sample(nrow(okc), size = 500, replace = FALSE),]#making the dataset smaller so it runs faster

## Slide 26 "Example Data - OKCupid"

set.seed(1732)
okc_ind <- createDataPartition(okc$Class, p = 2/3, list = FALSE)
okc_train <- okc[ okc_ind,]
okc_test  <- okc[-okc_ind,]

mean(okc_train$Class == "stem")
mean(okc_test$Class == "stem")



## Slide 81 "CART and Costs - OkC Data"

fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}

ctrl_cost <- trainControl(method = "repeatedcv",
                          repeats = 5,
                          savePredictions = TRUE,
                          summaryFunction = fourStats)


## Slide 82 "CART and Costs - OkC Data"

## Get an initial grid of Cp values
rpart_init <- rpart(Class ~ ., data = okc_train, cp = 0)$cptable

cost_grid <- expand.grid(cp = rpart_init[, "CP"], Cost = 1:5)

## Use the non-formula method. Many of the predictors are factors and
## this will preserve the factor encoding instead of using dummy variables. 

set.seed(1537)
rpart_costs <- train(x = okc_train[, names(okc_train) != "Class"],
                     y = okc_train$Class,
                     method = "rpartCost",
                     tuneGrid = cost_grid,
                     metric = "Kappa",
                     trControl = ctrl_cost)


## Slide 84 "CART and Costs - OkC Data"

ggplot(rpart_costs) + 
  scale_x_log10() + 
  theme(legend.position = "top")


## Slide 85 "CART and Costs - OkC Data"

ggplot(rpart_costs, metric = "Sens") + 
  scale_x_log10() + 
  theme(legend.position = "top")


## Slide 86 "CART and Costs - OkC Data"

ggplot(rpart_costs, metric = "Spec") + 
  scale_x_log10() + 
  theme(legend.position = "top")


## Slide 87 "C5.0 and Costs - OkC Data"

cost_grid <- expand.grid(trials = c(10, 20, 30),#c(1:10, 20, 30)
                         winnow = FALSE, model = "tree",
                         cost = c(1, 5))#cost = c(1, 5, 10, 15)
set.seed(1537)
c5_costs <- train(x = okc_train[, names(okc_train) != "Class"],
                  y = okc_train$Class,
                  method = "C5.0Cost",
                  tuneGrid = cost_grid,
                  metric = "Kappa",
                  trControl = ctrl_cost)

## Slide 89 "C5.0 and Costs - OkC Data"

ggplot(c5_costs) + theme(legend.position = "top")


## Slide 91 "OkC Test Results - C5.0"

rp_pred <- predict(rpart_costs, newdata = okc_test)
confusionMatrix(rp_pred, okc_test$Class)


## Slide 90 "OkC Test Results - CART"

c5_pred <- predict(c5_costs, newdata = okc_test)
confusionMatrix(c5_pred, okc_test$Class)


## Slide 103 "CART and Costs and Probabilities"

cost_mat <-matrix(c(0, 1, 5, 0), ncol = 2)
rownames(cost_mat) <- colnames(cost_mat) <- levels(okc_train$Class)
rp_mod <- rpart(Class ~ ., data = okc_train, parms = list(loss = cost_mat))
pred_1 <- predict(rp_mod, okc_test, type = "class")
pred_2 <- ifelse(predict(rp_mod, okc_test)[, "stem"] >= .5, "stem", "other")
pred_2 <- factor(pred_2, levels = levels(pred_1))

table(pred_1, pred_2)


###################################################################

## Slide 51 "Approximate Random Forest Resampled ROC Curve"

## This function averages the class probability values per sample
## across the hold-outs to get an averaged ROC curve

roc_train <- function(object, best_only = TRUE, ...) {
  
  
  lvs <- object$modelInfo$levels(object$finalModel)
  
  if(best_only) {
    object$pred <- merge(object$pred, object$bestTune)
  }
  
  ## find tuning parameter names
  p_names <- as.character(object$modelInfo$parameters$parameter)
  p_combos <- object$pred[, p_names, drop = FALSE]
  
  ## average probabilities across resamples
  object$pred <- ddply(.data = object$pred, #plyr::
                       .variables = c("obs", "rowIndex", p_names),
                       .fun = function(dat, lvls = lvs) {
                         out <- mean(dat[, lvls[1]])
                         names(out) <- lvls[1]
                         out
                       })
  
  make_roc <- function(x, lvls = lvs, nms = NULL, ...) {
    out <- roc(response = x$obs,#pROC::
               predictor = x[, lvls[1]],
               levels = rev(lvls))
    
    out$model_param <- x[1,nms,drop = FALSE]
    out
  }
  out <- plyr::dlply(.data = object$pred, 
                     .variables = p_names,
                     .fun = make_roc,
                     lvls = lvs,
                     nms = p_names)
  if(length(out) == 1)  out <- out[[1]]
  out
}

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")


## Slide 52 "A Better Cutoff"

plot(roc_train(rf_emr_mod), 
     legacy.axes = TRUE,
     print.thres.pattern = "Cutoff: %.2f (Sp = %.2f, Sn = %.2f)",
     print.thres = "best")
