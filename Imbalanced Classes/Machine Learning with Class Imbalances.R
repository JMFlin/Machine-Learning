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

ctrl <- trainControl(method = "cv",
                     repeats = 10, 
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = twoClassSummary)
emr_grid <- data.frame(mtry = c(2,5))#c(1:15, (4:9)*5)

set.seed(1537)
rf_emr_mod <- train(Class ~ ., 
                    data = emr_train,
                    method = "rf",
                    metric = "ROC",
                    tuneGrid = emr_grid,
                    ntree = 1000,
                    trControl = ctrl)


## Slide 50 "Random Forest Results - EMR Example"

ggplot(rf_emr_mod)



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


## Slide 59 "Down-Sampling - EMR Data"

down_ctrl <- ctrl
down_ctrl$sampling <- "down"
set.seed(1537)
rf_emr_down <- train(Class ~ ., 
                     data = emr_train,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = emr_grid,
                     ntree = 1000,
                     trControl = down_ctrl)


## Slide 60 "Down-Sampling - EMR Data"

ggplot(rf_emr_down)


## Slide 61 "Approximate Resampled ROC Curve with Down-Sampling"

plot(roc_train(rf_emr_down), 
     legacy.axes = TRUE,
     print.thres = .5,
     print.thres.pattern="   <- default %.1f threshold")


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


## Slide 64 "Internal Down-Sampling - EMR Data"

ggplot(rf_emr_down_int)


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


## Slide 68 "Up-Sampling - EMR Data"

ggplot(rf_emr_up)


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


## Slide 74 "SMOTE - EMR Data"

ggplot(rf_emr_smote)


## Slide 75 "SMOTE - EMR Data"

emr_test_pred <- data.frame(Class = emr_test$Class)
emr_test_pred$normal <- predict(rf_emr_mod, emr_test, type = "prob")[, "event"]
emr_test_pred$down <- predict(rf_emr_down, emr_test, type = "prob")[, "event"]
emr_test_pred$down_int <- predict(rf_emr_down_int, emr_test, type = "prob")[, "event"]
emr_test_pred$up <- predict(rf_emr_up, emr_test, type = "prob")[, "event"]
emr_test_pred$smote <- predict(rf_emr_smote, emr_test, type = "prob")[, "event"]

get_auc <- function(pred, ref){
    auc(roc(ref, pred, levels = rev(levels(ref))))
}

apply(emr_test_pred[, -1], 2, get_auc, ref = emr_test_pred$Class)

###################################################################

## Slide 25 "Example Data - OKCupid"

load("okc.RData") ## create this using the file "okc_data.R"
str(okc, list.len = 20, vec.len = 2)


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
## this will preserve the factor encoding instead of using dummy 
## variables. 

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

cost_grid <- expand.grid(trials = c(1:10, 20, 30),
                         winnow = FALSE, model = "tree",
                         cost = c(1, 5, 10, 15))
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