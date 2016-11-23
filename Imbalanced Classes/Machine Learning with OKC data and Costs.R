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
