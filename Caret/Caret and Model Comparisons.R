#https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf
#List of models:
#http://topepo.github.io/caret/modelList.html
#http://topepo.github.io/caret/bytag.html

#install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
library(mlbench)
data(Sonar)


#The Sonar data consist of 208 data points collected on 60 predictors.  The goal is to predict the two
#classes (M for metal cylinder or R for rock).
#First, we split the data into two groups: a training set and a test set. To do this, the
#createDataPartition function is used:
set.seed(123)
inTrain <- createDataPartition(y = Sonar$Class, #the response variable
                               p = 0.75, #data plot %
                               list = FALSE)
## The output is a set of integers for the rows of Sonar
## that belong in the training set.

training <- Sonar[inTrain,]
testing <- Sonar[-inTrain,]
nrow(training)
nrow(testing)

#Default tunes over 3 values
#Default resampling is bootstrap
#Default performance measure is accuray and Kappa for classification and RMSE and R2 for regression.
set.seed(123)
ctrl <- trainControl(method = "repeatedcv", number = 10 ,repeats = 3,# this does 3 repeats of 10-fold corss validation
                     classProbs = TRUE,#used to calculate the area under ROC, sensitivity and specificity. Only for 2 class problems.
                     summaryFunction = twoClassSummary) 

plsFit <- train(Class~., 
                data = training,
                method = "pls",
                tuneLength = 15,#tune parameter from 1 to 15
                preProc = c("center", "scale"),
                trControl = ctrl,
                metric = "ROC")#evaluation is done with Area Under Curve

#could also do 
#grid <- expand.grid(.alpha=seq(1,1,1),         #alpha 1 is lasso                
#                                  .lambda=seq(100,0,-.1)
#and add this inside the train tuneGrid=grid

plsFit # tells the optimal components number. Based on this value, a final PLS model is fit to the whole data set using this
#specification and this is the model that is used to predict future samples.
plot(plsFit)#y-axis is the area under curve. Higher the better

#to predict new samples. For classification models, the default behavior is to calculate the predicted class. Using the option
#type = "prob" can be used to compute class probabilities from the model.

plsClasses <- predict(plsFit, newdata = testing)
str(plsClasses)

plsProbs <- predict(plsFit, newdata = testing, type = "prob")
head(plsProbs)

#Confusion matrix and associated statistics for model fit:

confusionMatrix(data = plsClasses, testing$Class)
postResample(plsClasses, testing$Class)
sensitivity(plsClasses, testing$Class)
#The "no--information rate" is the largest proportion of the observed classes (there were more actives than inactives in this test set).
#A hypothesis test is also computed to evaluate whether the overall accuracy rate is greater than the rate of the largest class. 
#Also, the prevalence of the "positive event" is computed from the data (unless passed in as an argument), 
#the detection rate (the rate of true events also predicted to be events) and the detection prevalence 
#(the prevalence of predicted events). 
#Fore more: http://topepo.github.io/caret/other.html

#Regularized discriminant analysis
set.seed(123)
grid <- expand.grid(.gamma=(0:4)/4,                  
                     .lambda=seq(1,0,-.1)
)

ctrl <- trainControl(method = "repeatedcv", number = 10 ,repeats = 3,# this does 3 repeats of 10-fold corss validation
                     classProbs = TRUE,#used to calculate the area under ROC, sensitivity and specificity. Only for 2 class problems.
                     summaryFunction = twoClassSummary) 

rdaFit <- train(Class~.,
                data = training,
                method = "rda",
                tuneGrid = grid,
                trControl = ctrl,
                metric = "ROC")
rdaFit

rdaClasses <- predict(rdaFit, newdata = testing)
confusionMatrix(rdaClasses, testing$Class)

#How do these two models compare in terms of their resampling results? You need to select the set.seed before running the models
#so you get the same folds to be able to compare these models!

resamps <- resamples(list(pls = plsFit, rda = rdaFit))
summary(resamps)

xyplot(resamps, what = "BlandAltman")
#Since, for each resample, there are paired results a paired t test can be used to asses wheather there is a difference in the average AUC

diffs <- diff(resamps)
summary(diffs)
#Based on this analysis, the difference between the models is -0.0183 ROC units (rda is better) and the two sided p-value
#for this difference is 0.3477
