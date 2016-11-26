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
#Sampling methods for combating class imbalances

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
library(reshape2)
library(gridExtra)
library(colorspace)

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
prop.table(table(emr_train$Class))
prop.table(table(emr_test$Class))

## Slide 47 "Random Forests with the EMR Data"

## on Windows, try the doParallel package
## **if** your computer has multiple cores and sufficient memory
#cmd -> WMIC CPU Get DeviceID,NumberOfCores,NumberOfLogicalProcessors

ctrl <- trainControl(method = "repeatedcv",
                     number = 2,
                     repeats = 2, #  number = 10, repeats = 10 is 10 fold cv
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

sensitivity(rfClasses, emr_test$Class)#Sensitivity: given that a result is truly an event, what is the probability that the model will predict an event result? True positive rate
#Put another way it is the number of positive predictions divided by the number of positive class values in the test data.
specificity(rfClasses, emr_test$Class)#Specificity: given that a result is truly not an event, what is the probabiliy that the model will predict a nonevent result? True negative rate
precision(rfClasses, emr_test$Class)#Precision: it is the number of positive predictions divided by the total number of positive class values predicted. It is also called the Positive Predictive Value (PPV).
#Precision can be thought of as a measure of a classifiers exactness.
recall(rfClasses, emr_test$Class)#Recall: Same as Sensitivity

sens.spec <- data.frame(t(confusionMatrix(data = rfClasses, emr_test$Class)$byClass["Sensitivity"]))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses, emr_test$Class)$byClass["Specificity"])))

confu <- data.frame(t(confusionMatrix(data = rfClasses, emr_test$Class)$byClass["Balanced Accuracy"]))
#If the classifier performs equally well on either class, this term reduces to the conventional accuracy (number of correct predictions divided by number of predictions). 
#In contrast, if the conventional accuracy is high only because the classifier takes advantage of an imbalanced test set, then the balanced accuracy, as desired, will drop to chance.
#The balanced accuracy used here is symmetric about the type of class.
confusionMatrix(data = rfClasses, emr_test$Class)
#The "no--information rate" is the largest proportion of the observed classes (Baseline!!).
#A hypothesis test is also computed to evaluate whether the overall accuracy rate is greater than the rate of the largest class.
#Also, the prevalence of the "positive event" is computed from the data (unless passed in as an argument), 
#the detection rate (the rate of true events also predicted to be events) and the detection prevalence (the prevalence of predicted events). 
#https://topepo.github.io/caret/measuring-performance.html

postResample(rfClasses, emr_test$Class)
Kap <- data.frame(t(postResample(rfClasses, emr_test$Class)))
#The Kappa statistic (or value) is a metric that compares an Observed Accuracy with an Expected Accuracy (random chance).
#Landis and Koch considers 0-0.20 as slight, 0.21-0.40 as fair, 0.41-0.60 as moderate, 0.61-0.80 as substantial, and 0.81-1 as almost perfect. 
#Fleiss considers kappas > 0.75 as excellent, 0.40-0.75 as fair to good, and < 0.40 as poor.

## Slide 50 "Random Forest Results - EMR Example"

ggplot(rf_emr_mod)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_mod, emr_test,type="prob")
pr <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
pe <- performance(pr, "tpr", "fpr")
roc.data <- data.frame(Model='Random Forest',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))

q <- ggplot(data=roc.data, aes(x=fpr, y=tpr, group = Model, colour = Model)) 
q <- q + geom_line() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


#KS

pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))#Ideally, labels should be supplied as ordered factor(s), the lower level corresponding to the
#negative class, the upper level to the positive class.
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

q <- ggplot(data=perf.data, aes(x=x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#If your question is: "How meaningful is a positive result from my classifier given the baseline probabilities of my problem?", use a PR curve. 
#If your question is: "How well can this classifier be expected to perform in general, at a variety of different baseline probabilities?", go with a ROC curve.
#If a model shows good AUC, but still has poor early retrieval, the Precision-Recall curve will leave a lot to be desired. 
#For this reason, Saito et al. recommend using area under the Precision-Recall curve rather than AUC when you have imbalanced classes.

#Accurary
f.perf <- performance(pred, "acc")
f.data <- data.frame(Model='Random Forest',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]])

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#Another cost measure that is popular is overall accuracy. 
#This measure optimizes the correct results, but may be skewed if there are many more negatives than positives, or vice versa.

#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")

perfss.data <- data.frame(Model='Random Forest',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]])

q <- ggplot(data=perfss.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + xlab("Sensitivity") + ylab("Specificity") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 

#Lift
Lift.perf <- performance(pred, "lift", "rpp")

perflift.data <- data.frame(Model='Random Forest',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]])

q <- ggplot(data=perflift.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + xlab("Lift") + ylab("Rate of positive predictions") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#lift can be understood as a ratio of two percentages: the percentage of correct positive classifications made by the model to the percentage of actual positive classifications in the test data.


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
specificity(rfClasses_down, emr_test$Class)
precision(rfClasses_down, emr_test$Class)
recall(rfClasses_down, emr_test$Class)

sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_down, emr_test$Class)$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_down, emr_test$Class)$byClass["Specificity"])))


Kap <- rbind(Kap, data.frame(t(postResample(rfClasses_down, emr_test$Class))))
confu <- rbind(confu, data.frame(t(confusionMatrix(data = rfClasses_down, emr_test$Class)$byClass["Balanced Accuracy"])))


## Slide 60 "Down-Sampling - EMR Data"

ggplot(rf_emr_down)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_down, emr_test,type="prob")
pr <- prediction(rf.probs$noevent, emr_test$Class)
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Down-Sampling',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))


#KS
pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Down-Sampling'))

ks.val$down <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Down-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Down-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")
perfss.data <- rbind(perfss.data, data.frame(Model='Random Forest\n Down-Sampling',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]]))

#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Random Forest\n Down-Sampling',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))



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
specificity(rfClasses_down_int, emr_test$Class)
precision(rfClasses_down_int, emr_test$Class)
recall(rfClasses_down_int, emr_test$Class)

sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$byClass["Specificity"])))


Kap <- rbind(Kap, data.frame(t(postResample(rfClasses_down_int, emr_test$Class))))
confu <- rbind(confu, data.frame(t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$byClass["Balanced Accuracy"])))

## Slide 64 "Internal Down-Sampling - EMR Data"

ggplot(rf_emr_down_int)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_down_int, emr_test,type="prob")
pr <- prediction(rf.probs$noevent, emr_test$Class)
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Internal Down-Sampling',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))


rf.ROC <- roc(predictor=rf.probs$event,
              response=emr_test$Class,
              levels=rev(levels(emr_test$Class)))

#KS
pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Internal Down-Sampling'))

ks.val$down_int <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")
perfss.data <- rbind(perfss.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]]))

#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Random Forest\n Internal Down-Sampling',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))



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
specificity(rfClasses_up, emr_test$Class)
precision(rfClasses_up, emr_test$Class)
recall(rfClasses_up, emr_test$Class)


sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_up, emr_test$Class)$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_up, emr_test$Class)$byClass["Specificity"])))


Kap <- rbind(Kap, data.frame(t(postResample(rfClasses_up, emr_test$Class))))
confu <- rbind(confu, data.frame(t(confusionMatrix(data = rfClasses_up, emr_test$Class)$byClass["Balanced Accuracy"])))


## Slide 68 "Up-Sampling - EMR Data"

ggplot(rf_emr_up)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_up, emr_test,type="prob")
pr <- prediction(rf.probs$noevent, emr_test$Class)
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n Up-Sampling',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))

#KS
pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n Up-Sampling'))

ks.val$up <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n Up-Sampling',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n Up-Sampling',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")
perfss.data <- rbind(perfss.data, data.frame(Model='Random Forest\n Up-Sampling',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]]))

#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Random Forest\n Up-Sampling',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))


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
specificity(rfClasses_smote, emr_test$Class)
precision(rfClasses_smote, emr_test$Class)
recall(rfClasses_smote, emr_test$Class)

sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$byClass["Specificity"])))


Kap <- rbind(Kap, data.frame(t(postResample(rfClasses_smote, emr_test$Class))))
confu <- rbind(confu, data.frame(t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$byClass["Balanced Accuracy"])))


## Slide 74 "SMOTE - EMR Data"

ggplot(rf_emr_smote)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_smote, emr_test,type="prob")
pr <- prediction(rf.probs$noevent, emr_test$Class)
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n SMOTE',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))


#KS
pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Random Forest\n SMOTE'))

ks.val$smote <- max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec")

perf.data <- rbind(perf.data, data.frame(Model='Random Forest\n SMOTE',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n SMOTE',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")
perfss.data <- rbind(perfss.data, data.frame(Model='Random Forest\n SMOTE',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]]))

#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Random Forest\n SMOTE',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))


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
specificity(rfClasses_rose, emr_test$Class)
precision(rfClasses_rose, emr_test$Class)
recall(rfClasses_rose, emr_test$Class)


sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$byClass["Specificity"])))


confu <- rbind(confu, data.frame(t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$byClass["Balanced Accuracy"])))

Kap <- rbind(Kap, data.frame(t(postResample(rfClasses_rose, emr_test$Class))))

##"ROSE - EMR Data"

ggplot(rf_emr_rose)

#Draw the ROC curve 
rf.probs <- predict(rf_emr_rose, emr_test,type="prob")
pr <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Random Forest\n ROSE',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))

q <- ggplot(data=roc.data, aes(x=fpr, y=tpr, group = Model, colour = Model)) 
q <- q + geom_line() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#KS
pred <- prediction(rf.probs$event, factor(emr_test$Class, levels = c("noevent", "event"), ordered = TRUE))
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
q <- q + geom_line() + xlab("Recall (Sensitivity)") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())
#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Random Forest\n ROSE',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))

q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#Sensitivity-Specificity
SS.perf <- performance(pred, "sens", "spec")
perfss.data <- rbind(perfss.data, data.frame(Model='Random Forest\n ROSE',x=SS.perf@x.values[[1]], y=SS.perf@y.values[[1]]))

q <- ggplot(data=perfss.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_path() + xlab("Specificity") + ylab("Sensitivity") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 
#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Random Forest\n ROSE',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))
q <- ggplot(data=perflift.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Rate of positive predictions") + ylab("Lift") 
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
ggplot(dat.m, aes(x = names, y = value)) + #fill = names
  geom_bar(stat='identity', colour="black") + ylab(label="AUC")+ geom_hline(yintercept = .90, linetype = "dashed")+ geom_hline(yintercept = .70, linetype = "dashed")+xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


2*apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)-1#Gini Coefficient/Ratio Above 60% corresponds to a good model
tst <- data.frame(2*apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)-1)
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) +xlab(label="")+
  geom_bar(stat='identity', colour="black") + ylab(label="Gini Coefficient")+geom_hline(yintercept = .60, linetype = "dashed")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

ks.val#It is the maximum difference between the cumulative true positive rate and the cumulative false positive rate
tst <- data.frame(t(ks.val))
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) +
  geom_bar(stat='identity', colour="black") + ylab(label="Kolmogorov-Smirnov Maximums")+xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

Kap$names <- names(emr_test_pred)[2:length(names(emr_test_pred))]
Kap$Accuracy <- NULL
dat.m <- melt(Kap,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) +
  geom_bar(stat='identity', colour="black") + ylab(label="Kappa")+ geom_hline(yintercept = .40, linetype = "dashed")+ geom_hline(yintercept = .75, linetype = "dashed")+ geom_hline(yintercept = .20, linetype = "dashed") +xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


Accus <- data.frame(t(confusionMatrix(data = rfClasses, emr_test$Class)$overall["Accuracy"]), 
                    t(confusionMatrix(data = rfClasses_down, emr_test$Class)$overall["Accuracy"]),
                      t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$overall["Accuracy"]),
                        t(confusionMatrix(data = rfClasses_up, emr_test$Class)$overall["Accuracy"]),
                          t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$overall["Accuracy"]),
                            t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$overall["Accuracy"]))
Accus.1 <- data.frame(t(Accus))
Accus.1$names <- names(emr_test_pred)[2:length(names(emr_test_pred))]
dat.Accus <- melt(Accus.1,id.vars = "names")
dat.Accus$variable <- "Accuracy"


confu$names <- names(emr_test_pred)[2:length(names(emr_test_pred))]
dat.confu <- melt(confu,id.vars = "names")
dat.confu <- rbind(dat.confu, dat.Accus)
ggplot(dat.confu, aes(x = names, y = value, fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="")+ geom_hline(yintercept = confusionMatrix(data = rfClasses, emr_test$Class)$overall["AccuracyNull"], linetype = "dashed")+ xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

Prev <- data.frame(t(confusionMatrix(data = rfClasses, emr_test$Class)$byClass["Detection Rate"]), 
                   t(confusionMatrix(data = rfClasses_down, emr_test$Class)$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses_up, emr_test$Class)$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses, emr_test$Class)$byClass["Detection Prevalence"]), 
                   t(confusionMatrix(data = rfClasses_down, emr_test$Class)$byClass["Detection Prevalence"]),
                   t(confusionMatrix(data = rfClasses_down_int, emr_test$Class)$byClass["Detection Prevalence"]),
                   t(confusionMatrix(data = rfClasses_up, emr_test$Class)$byClass["Detection Prevalence"]),
                   t(confusionMatrix(data = rfClasses_smote, emr_test$Class)$byClass["Detection Prevalence"]),
                   t(confusionMatrix(data = rfClasses_rose, emr_test$Class)$byClass["Detection Prevalence"]))

Prev.1 <- data.frame(t(Prev))
Prev.1$names <- names(emr_test_pred)[2:length(names(emr_test_pred))]
Prev.1$variable <- row.names(Prev.1)
Prev.1$variable <- gsub("\\.[[:digit:]]", "", Prev.1$variable)

dat.Prev.1 <- melt(Prev.1,id.vars = "names")

ggplot(Prev.1, aes(x = names, y = Prev.1[,1], fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="")+ xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


sens.1 <- data.frame(t(data.frame(sens.spec)))
a <- data.frame(names(emr_test_pred)[2:length(names(emr_test_pred))])
b <- data.frame(a[rep(seq_len(nrow(a)), each=2),])
sens.1$names <- b[,1]
dat.sens.1 <- melt(sens.1,id.vars = "names")
dat.sens.1$variable <- row.names(sens.1)

dat.sens.1$variable <- gsub("\\.[[:digit:]]", "",dat.sens.1$variable)

prec <- data.frame(precision(rfClasses, emr_test$Class),
                   precision(rfClasses_down, emr_test$Class),
                   precision(rfClasses_down_int, emr_test$Class),
                   precision(rfClasses_up, emr_test$Class),
                   precision(rfClasses_smote, emr_test$Class),
                   precision(rfClasses_smote, emr_test$Class))

prec.1 <- data.frame(t(prec))
a <- data.frame(names(emr_test_pred)[2:length(names(emr_test_pred))])
b <- data.frame(a[rep(seq_len(nrow(a)), each=1),])
prec.1$names <- b[,1]
dat.prec.1 <- melt(prec.1,id.vars = "names")
dat.prec.1$variable <- rep(c("Precision"), times=nrow(dat.prec.1)/2)

all <- rbind(dat.prec.1, dat.sens.1)
out <- split(all, f = all$variable)
#same as from confusionMatrix
out.df <- data.frame(names = names(emr_test_pred)[2:length(names(emr_test_pred))], variable = "F1 Score", value = 2*((out$Precision$value*out$Sensitivity$value)/(out$Precision$value+out$Sensitivity$value)))
all <- rbind(dat.prec.1, dat.sens.1, out.df)

ggplot(all, aes(x = names, y = value, fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="") + xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

resamps <- resamples(list(normal = rf_emr_mod, down = rf_emr_down, down_int = rf_emr_down_int,
                          up=rf_emr_up,smote=rf_emr_smote,rose=rf_emr_rose))
summary(resamps)

df <- data.frame(resamps$values)

my_subset_ROC <- df[,grep("ROC", names(df))]
my_subset_Sens <- df[,grep("Sens", names(df))]
my_subset_Spec <- df[,grep("Spec", names(df))]

my_subset_ROC_t <- data.frame(t(my_subset_ROC))
my_subset_ROC_t$names <- names(my_subset_ROC)

ROC_melt <- melt(my_subset_ROC_t ,id.vars = "names")

plot1 <- ggplot(data = ROC_melt, aes( names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(aes(color = "blue")) + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="")+ggtitle("AUC")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


my_subset_Sens_t <- data.frame(t(my_subset_Sens))
my_subset_Sens_t$names <- names(my_subset_Sens)

Sens_melt <- melt(my_subset_Sens_t ,id.vars = "names")

plot2 <- ggplot(data = Sens_melt, aes( names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(color = "red") + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="")+ggtitle("Sensitivity")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

my_subset_Spec_t <- data.frame(t(my_subset_Spec))
my_subset_Spec_t$names <- names(my_subset_Spec)

Spec_melt <- melt(my_subset_Spec_t ,id.vars = "names")

plot3 <- ggplot(data = Spec_melt, aes( names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(color = "blue") + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="") + ggtitle("Specificity")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

grid.arrange(plot1, plot2, plot3, ncol=1)
