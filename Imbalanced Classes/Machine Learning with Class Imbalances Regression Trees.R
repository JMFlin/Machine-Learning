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