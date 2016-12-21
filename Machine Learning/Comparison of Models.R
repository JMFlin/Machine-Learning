library(ggplot2)
library(caret)
library(pROC)
library(ggthemes)
library(plyr)
library(ROCR)
library(reshape2)
library(gridExtra)

#load("emr.RData")
#my_data <- emr[sample(nrow(emr), size = 4000, replace = FALSE),]#making the dataset smaller so it runs faster


set.seed(1732)

my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")

names(my_data)[1] <- "Class"
my_data$Class <- ifelse(my_data$Class == 1, "event", "noevent")

my_data$Class <- as.factor(my_data$Class)#my_data$Class <- factor(my_data$Class, levels = c("noevent", "event"), ordered = TRUE)

ind <- createDataPartition(my_data$Class, p = 2/3, list = FALSE)
train <- my_data[ ind,]
test  <- my_data[-ind,]

table(train$Class)
table(test$Class)
prop.table(table(train$Class))
prop.table(table(test$Class))


## on Windows, try the doParallel package
## **if** your computer has multiple cores and sufficient memory
#cmd -> WMIC CPU Get DeviceID,NumberOfCores,NumberOfLogicalProcessors

#Train RF
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)

grid <- data.frame(mtry = seq(1,3,1))

set.seed(1537)
rf_mod <- train(Class ~ ., 
                data = train,
                method = "rf",
                metric = "ROC",
                tuneGrid = grid,
                ntree = 1000,
                trControl = ctrl)


rfClasses <- predict(rf_mod, test)

sensitivity(rfClasses, test$Class, positive = "event")#Sensitivity: given that a result is truly an event, what is the probability that the model will predict an event result? True positive rate
#Put another way it is the number of positive predictions divided by the number of positive class values in the test data.
#specificity(rfClasses, test$Class)#Specificity: given that a result is truly not an event, what is the probabiliy that the model will predict a nonevent result? True negative rate
precision(rfClasses, test$Class, positive = "event")#Precision: it is the number of positive predictions divided by the total number of positive class values predicted. It is also called the Positive Predictive Value (PPV).
#Precision can be thought of as a measure of a classifiers exactness.
recall(rfClasses, test$Class, positive = "event")#Recall: Same as Sensitivity

sens.spec <- data.frame(t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$byClass["Sensitivity"]))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$byClass["Specificity"])))

confu <- data.frame(t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$byClass["Balanced Accuracy"]))
#If the classifier performs equally well on either class, this term reduces to the conventional accuracy (number of correct predictions divided by number of predictions). 
#In contrast, if the conventional accuracy is high only because the classifier takes advantage of an imbalanced test set, then the balanced accuracy, as desired, will drop to chance.
#The balanced accuracy used here is symmetric about the type of class.
confusionMatrix(data = rfClasses, test$Class, positive = "event")
#The "no--information rate" is the largest proportion of the observed classes (Baseline!!).
#A hypothesis test is also computed to evaluate whether the overall accuracy rate is greater than the rate of the largest class.
#Also, the prevalence of the "positive event" is computed from the data (unless passed in as an argument), 
#the detection rate (the rate of true events also predicted to be events) and the detection prevalence (the prevalence of predicted events). 
#https://topepo.github.io/caret/measuring-performance.html

postResample(rfClasses, test$Class)
Kap <- data.frame(t(postResample(rfClasses, test$Class)))
#The Kappa statistic (or value) is a metric that compares an Observed Accuracy with an Expected Accuracy (random chance).
#Landis and Koch considers 0-0.20 as slight, 0.21-0.40 as fair, 0.41-0.60 as moderate, 0.61-0.80 as substantial, and 0.81-1 as almost perfect. 
#Fleiss considers kappas > 0.75 as excellent, 0.40-0.75 as fair to good, and < 0.40 as poor.
#a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate.

## Slide 50 "Random Forest Results - EMR Example"

ggplot(rf_mod)

#ROC
rf.probs <- predict(rf_mod, test,type="prob")
pr <- prediction(rf.probs$event, factor(test$Class, levels = c("noevent", "event"), ordered = TRUE))
pe <- performance(pr, "tpr", "fpr")
roc.data <- data.frame(Model='Random Forest',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values))


#KS

pred <- prediction(rf.probs$event, factor(test$Class, levels = c("noevent", "event"), ordered = TRUE))#Ideally, labels should be supplied as ordered factor(s), the lower level corresponding to the
#negative class, the upper level to the positive class.
perf <- performance(pred, "tpr", "fpr")

ks <- data.frame(value = attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]), Model='Random Forest')

ks.val <- data.frame(value = max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]])), Model='Random Forest')
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

#If your question is: "How meaningful is a positive result from my classifier given the baseline probabilities of my problem?", use a PR curve. 
#If your question is: "How well can this classifier be expected to perform in general, at a variety of different baseline probabilities?", go with a ROC curve.
#If a model shows good AUC, but still has poor early retrieval, the Precision-Recall curve will leave a lot to be desired. 
#For this reason, Saito et al. recommend using area under the Precision-Recall curve rather than AUC when you have imbalanced classes.

#Accurary
f.perf <- performance(pred, "acc")
f.data <- data.frame(Model='Random Forest',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]])

#Another cost measure that is popular is overall accuracy. 
#This measure optimizes the correct results, but may be skewed if there are many more negatives than positives, or vice versa.

#Lift
Lift.perf <- performance(pred, "lift", "rpp")

perflift.data <- data.frame(Model='Random Forest',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]])
#lift can be understood as a ratio of two percentages: the percentage of correct positive classifications made by the model to the percentage of actual positive classifications in the test data.


#------------------------------#------------------------------#------------------------------
#------------------------------#------------------------------#------------------------------
#------------------------------#------------------------------#------------------------------


#Build SVM
set.seed(1537)
svm_mod <- train(Class ~ ., 
                 data = train,
                 method = "svmRadial",
                 metric = "ROC",
                 trControl = ctrl)

svmClasses <- predict(svm_mod, test)

confusionMatrix(data = svmClasses, test$Class, positive = "event")
postResample(svmClasses, test$Class)
sensitivity(svmClasses, test$Class,  positive = "event")
#specificity(svmClasses, test$Class)
precision(svmClasses, test$Class, positive = "event")
recall(svmClasses, test$Class, positive = "event")

sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$byClass["Sensitivity"])))
sens.spec <- cbind(sens.spec,data.frame(t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$byClass["Specificity"])))


Kap <- rbind(Kap, data.frame(t(postResample(svmClasses, test$Class))))
confu <- rbind(confu, data.frame(t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$byClass["Balanced Accuracy"])))


#Draw the ROC curve 
svm.probs <- predict(svm_mod, test,type="prob")
pr <- prediction(svm.probs$event, factor(test$Class, levels = c("noevent", "event"), ordered = TRUE))
pe <- performance(pr, "tpr", "fpr")
roc.data <- rbind(roc.data, data.frame(Model='Support Vector Machine',fpr=unlist(pe@x.values), tpr=unlist(pe@y.values)))


#KS
pred <- prediction(svm.probs$event, factor(test$Class, levels = c("noevent", "event"), ordered = TRUE))
perf <- performance(pred, "tpr", "fpr")
ks <- rbind(ks, data.frame(value = attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]]),Model='Support Vector Machine'))

ks.val <- rbind(ks.val, data.frame(value = max(attr(perf, "y.values")[[1]] - (attr(perf, "x.values")[[1]])), Model='Support Vector Machine'))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");

perf.data <- rbind(perf.data, data.frame(Model='Support Vector Machine',x=RP.perf@x.values[[1]], y=RP.perf@y.values[[1]]))

#Accuracy
f.perf <- performance(pred, "acc")
f.data <- rbind(f.data, data.frame(Model='Support Vector Machine',x=f.perf@x.values[[1]], y=f.perf@y.values[[1]]))


#Lift
Lift.perf <- performance(pred, "lift", "rpp")
perflift.data <- rbind(perflift.data, data.frame(Model='Support Vector Machine',x=Lift.perf@x.values[[1]], y=Lift.perf@y.values[[1]]))


#------------------------------#------------------------------#------------------------------
#------------------------------#------------------------------#------------------------------
#------------------------------#------------------------------#------------------------------



#Plotting and  more statistics

#ROC of testing set

q <- ggplot(data=roc.data, aes(x=fpr, y=tpr, group = Model, colour = Model)) 
q <- q + geom_line() + geom_abline(intercept = 0, slope = 1) + xlab("False Positive Rate (1-Specificity)") + ylab("True Positive Rate (Sensitivity)") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'),     legend.text=element_text(), legend.title=element_text())


#KS
d <- ggplot(ks, aes(x=1:nrow(ks), y=value, colour = Model)) + geom_line() + xlab(label="Index") + ylab(label="Kolmogorov-Smirnov Values")
d + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

# Recall-Precision curve             
q <- ggplot(data=perf.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Recall (Sensitivity)") + ylab("Precision") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#Accuracy
q <- ggplot(data=f.data, aes(x, y=y, group = Model, colour = Model)) 
q <- q + geom_line() + xlab("Cutoff") + ylab("Accuracy") 
q + theme(axis.line = element_line(), axis.text=element_text(color='black'), 
          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


#AUC
test_pred <- data.frame(Class = test$Class)
test_pred$Rf <- predict(rf_mod, test, type = "prob")[, "event"]
test_pred$Svm <- predict(svm_mod, test, type = "prob")[, "event"]

get_auc <- function(pred, ref){
  auc(roc(ref, pred, levels = rev(levels(ref))))
}

#Lift
trellis.par.set(caretTheme())
lift_obj <- lift(Class ~ Rf + Svm, data = test_pred)
plot(lift_obj, values = 60, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE))


apply(test_pred[, -1], 2, get_auc, ref = test_pred$Class)
tst <- data.frame(apply(test_pred[, -1], 2, get_auc, ref = test_pred$Class))
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) + #fill = names
  geom_bar(stat='identity', colour="black", width=.5) + ylab(label="AUC")+ geom_hline(yintercept = .90, linetype = "dashed")+ geom_hline(yintercept = .70, linetype = "dashed")+xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#Gini Coefficient
2*apply(test_pred[, -1], 2, get_auc, ref = test_pred$Class)-1#Gini Coefficient/Ratio Above 60% corresponds to a good model
tst <- data.frame(2*apply(test_pred[, -1], 2, get_auc, ref = test_pred$Class)-1)
tst$names <- row.names(tst)
dat.m <- melt(tst,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) +xlab(label="")+
  geom_bar(stat='identity', colour="black", width=.5) + ylab(label="Gini Coefficient")+geom_hline(yintercept = .60, linetype = "dashed")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


#Calibration curves can be used to characterisze how consistent the predicted class probabilities are with the observed event rates.
cal_obj <- calibration(Class ~ Rf+Svm, data = test_pred)
ggplot(cal_obj) + geom_line()

#It is the maximum difference between the cumulative true positive rate and the cumulative false positive rate
ggplot(ks.val, aes(x = Model, y = ks.val[,1])) +
  geom_bar(stat='identity', colour="black", width=.5) + ylab(label="Kolmogorov-Smirnov Maximums")+xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate.
Kap$names <- names(test_pred)[2:length(names(test_pred))]
Kap$Accuracy <- NULL
dat.m <- melt(Kap,id.vars = "names")
ggplot(dat.m, aes(x = names, y = value)) +
  geom_bar(stat='identity', colour="black") + ylab(label="Kappa")+ geom_hline(yintercept = .40, linetype = "dashed")+ geom_hline(yintercept = .75, linetype = "dashed")+ geom_hline(yintercept = .20, linetype = "dashed") +xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

Accus <- data.frame(t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$overall["Accuracy"]), 
                    t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$overall["Accuracy"]))
Accus.1 <- data.frame(t(Accus))
Accus.1$names <- names(test_pred)[2:length(names(test_pred))]
dat.Accus <- melt(Accus.1,id.vars = "names")
dat.Accus$variable <- "Accuracy"


confu$names <- names(test_pred)[2:length(names(test_pred))]
dat.confu <- melt(confu,id.vars = "names")
dat.confu <- rbind(dat.confu, dat.Accus)
ggplot(dat.confu, aes(x = names, y = value, fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="")+ geom_hline(yintercept = confusionMatrix(data = rfClasses, test$Class)$overall["AccuracyNull"], linetype = "dashed")+ xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

Prev <- data.frame(t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$byClass["Detection Rate"]), 
                   t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$byClass["Detection Rate"]),
                   t(confusionMatrix(data = rfClasses, test$Class, positive = "event")$byClass["Detection Prevalence"]), 
                   t(confusionMatrix(data = svmClasses, test$Class, positive = "event")$byClass["Detection Prevalence"]))

Prev.1 <- data.frame(t(Prev))
Prev.1$names <- names(test_pred)[2:length(names(test_pred))]
Prev.1$variable <- row.names(Prev.1)
Prev.1$variable <- gsub("\\.[[:digit:]]", "", Prev.1$variable)

ggplot(Prev.1, aes(x = names, y = Prev.1[,1], fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="")+ xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

logLoss <- suppressWarnings(data.frame(t(mnLogLoss(data = rf_mod$pred, lev = levels(factor(test$Class, order = FALSE)))),
                      t(mnLogLoss(data = svm_mod$pred, lev = levels(factor(test$Class, order = FALSE))))))
logLoss <- data.frame(t(logLoss))
logLoss$names <- names(test_pred)[2:length(names(test_pred))]
logLoss$variable <- row.names(logLoss)
logLoss$variable <- gsub("\\.[[:digit:]]", "", logLoss$variable)
#Smaller LogLoss is better!
#Log Loss heavily penalises classifiers that are confident about an incorrect classification.
ggplot(logLoss, aes(x = names, y = logLoss[,1])) +
  geom_bar(stat='identity', position = "dodge", colour="black", width = 0.5) + ylab(label="LogLoss")+ xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

sens.1 <- data.frame(t(data.frame(sens.spec)))
a <- data.frame(names(test_pred)[2:length(names(test_pred))])
b <- data.frame(a[rep(seq_len(nrow(a)), each=2),])
sens.1$names <- b[,1]
dat.sens.1 <- melt(sens.1,id.vars = "names")
dat.sens.1$variable <- row.names(sens.1)

dat.sens.1$variable <- gsub("\\.[[:digit:]]", "",dat.sens.1$variable)

prec <- data.frame(precision(rfClasses, test$Class, positive = "event"),
                   precision(svmClasses, test$Class, positive = "event"))

prec.1 <- data.frame(t(prec))
a <- data.frame(names(test_pred)[2:length(names(test_pred))])
b <- data.frame(a[rep(seq_len(nrow(a)), each=1),])
prec.1$names <- b[,1]
dat.prec.1 <- melt(prec.1,id.vars = "names")
dat.prec.1$variable <- rep(c("Precision"), times=nrow(dat.prec.1)/2)

all <- rbind(dat.prec.1, dat.sens.1)
out <- split(all, f = all$variable)
#same as from confusionMatrix
out.df <- data.frame(names = names(test_pred)[2:length(names(test_pred))], variable = "F1 Score", value = 2*((out$Precision$value*out$Sensitivity$value)/(out$Precision$value+out$Sensitivity$value)))
all <- rbind(dat.prec.1, dat.sens.1, out.df)

ggplot(all, aes(x = names, y = value, fill=variable)) +
  geom_bar(stat='identity', position = "dodge", colour="black") + ylab(label="") + xlab(label="")+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

#hold-out-sample AUC
resamps <- resamples(list(Rf = rf_mod, Svm = svm_mod))
summary(resamps)

df <- data.frame(resamps$values)

my_subset_ROC <- df[,grep("ROC", names(df))]
my_subset_Sens <- df[,grep("Sens", names(df))]
my_subset_Spec <- df[,grep("Spec", names(df))]

my_subset_ROC_t <- data.frame(t(my_subset_ROC))
my_subset_ROC_t$names <- names(my_subset_ROC)

ROC_melt <- melt(my_subset_ROC_t ,id.vars = "names")

plot1 <- ggplot(data = ROC_melt, aes(names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(aes(color = "blue")) + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="")+ggtitle("AUC")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())


my_subset_Sens_t <- data.frame(t(my_subset_Sens))
my_subset_Sens_t$names <- names(my_subset_Sens)

Sens_melt <- melt(my_subset_Sens_t ,id.vars = "names")

plot2 <- ggplot(data = Sens_melt, aes(names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(color = "red") + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="")+ggtitle("Sensitivity")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

my_subset_Spec_t <- data.frame(t(my_subset_Spec))
my_subset_Spec_t$names <- names(my_subset_Spec)

Spec_melt <- melt(my_subset_Spec_t ,id.vars = "names")

plot3 <- ggplot(data = Spec_melt, aes(names, value)) +
  stat_boxplot(geom ='errorbar')+ geom_boxplot() + geom_jitter(color = "blue") + coord_flip(ylim = c(0, 1)) + xlab(label="") + ylab(label="") + ggtitle("Specificity")+guides(colour=FALSE)+
  theme(axis.line = element_line(), axis.text=element_text(color='black'), 
        axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text())

grid.arrange(plot1, plot2, plot3, ncol=1)

trellis.par.set(caretTheme())
xyplot(resamps, what = "BlandAltman")
#Since, for each resample, there are paired results a paired t test can be used to asses wheather there is a difference in the average AUC

diffs <- diff(resamps)
summary(diffs)
#Based on this analysis, the difference between the models is -0.0183 ROC units (rda is better) and the two sided p-value
#for this difference is 0.3477