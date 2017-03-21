Summary
-------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

As described in the website, six young health participants were asked to
perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl
in five different fashions: exactly according to the specification
(Class A), throwing the elbows to the front (Class B), lifting the
dumbbell only halfway (Class C), lowering the dumbbell only halfway
(Class D) and throwing the hips to the front (Class E).

The goal of this project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set.

    library(ggplot2)
    library(caret)
    library(reshape2)

    training <- read.csv("training.csv", na.string = c("NA","Na", "NaN", "", " ", "#DIV/0!"))
    validation <- read.csv("validation.csv", na.string = c("NA","Na", "NaN", "", " ", "#DIV/0!"))

Remove those columns that have 70% of their values missing. Check for
zero and near zero variance predictors and remove highly correlated
predictors.

    train <- training[, -which(colMeans(is.na(training)) > 0.7)]

    train <- train[, !grepl("^X|timestamp|window", names(train))]

    nearZeroVar(train, saveMetrics = TRUE)

    ##                      freqRatio percentUnique zeroVar   nzv
    ## user_name             1.100679    0.03057792   FALSE FALSE
    ## roll_belt             1.101904    6.77810621   FALSE FALSE
    ## pitch_belt            1.036082    9.37722964   FALSE FALSE
    ## yaw_belt              1.058480    9.97349913   FALSE FALSE
    ## total_accel_belt      1.063160    0.14779329   FALSE FALSE
    ## gyros_belt_x          1.058651    0.71348486   FALSE FALSE
    ## gyros_belt_y          1.144000    0.35164611   FALSE FALSE
    ## gyros_belt_z          1.066214    0.86127816   FALSE FALSE
    ## accel_belt_x          1.055412    0.83579655   FALSE FALSE
    ## accel_belt_y          1.113725    0.72877383   FALSE FALSE
    ## accel_belt_z          1.078767    1.52379982   FALSE FALSE
    ## magnet_belt_x         1.090141    1.66649679   FALSE FALSE
    ## magnet_belt_y         1.099688    1.51870350   FALSE FALSE
    ## magnet_belt_z         1.006369    2.32901845   FALSE FALSE
    ## roll_arm             52.338462   13.52563449   FALSE FALSE
    ## pitch_arm            87.256410   15.73234125   FALSE FALSE
    ## yaw_arm              33.029126   14.65701763   FALSE FALSE
    ## total_accel_arm       1.024526    0.33635715   FALSE FALSE
    ## gyros_arm_x           1.015504    3.27693405   FALSE FALSE
    ## gyros_arm_y           1.454369    1.91621649   FALSE FALSE
    ## gyros_arm_z           1.110687    1.26388747   FALSE FALSE
    ## accel_arm_x           1.017341    3.95984099   FALSE FALSE
    ## accel_arm_y           1.140187    2.73672409   FALSE FALSE
    ## accel_arm_z           1.128000    4.03628580   FALSE FALSE
    ## magnet_arm_x          1.000000    6.82397309   FALSE FALSE
    ## magnet_arm_y          1.056818    4.44399144   FALSE FALSE
    ## magnet_arm_z          1.036364    6.44684538   FALSE FALSE
    ## roll_dumbbell         1.022388   84.20650290   FALSE FALSE
    ## pitch_dumbbell        2.277372   81.74498012   FALSE FALSE
    ## yaw_dumbbell          1.132231   83.48282540   FALSE FALSE
    ## total_accel_dumbbell  1.072634    0.21914178   FALSE FALSE
    ## gyros_dumbbell_x      1.003268    1.22821323   FALSE FALSE
    ## gyros_dumbbell_y      1.264957    1.41677709   FALSE FALSE
    ## gyros_dumbbell_z      1.060100    1.04984201   FALSE FALSE
    ## accel_dumbbell_x      1.018018    2.16593619   FALSE FALSE
    ## accel_dumbbell_y      1.053061    2.37488533   FALSE FALSE
    ## accel_dumbbell_z      1.133333    2.08949139   FALSE FALSE
    ## magnet_dumbbell_x     1.098266    5.74864948   FALSE FALSE
    ## magnet_dumbbell_y     1.197740    4.30129447   FALSE FALSE
    ## magnet_dumbbell_z     1.020833    3.44511263   FALSE FALSE
    ## roll_forearm         11.589286   11.08959331   FALSE FALSE
    ## pitch_forearm        65.983051   14.85577413   FALSE FALSE
    ## yaw_forearm          15.322835   10.14677403   FALSE FALSE
    ## total_accel_forearm   1.128928    0.35674243   FALSE FALSE
    ## gyros_forearm_x       1.059273    1.51870350   FALSE FALSE
    ## gyros_forearm_y       1.036554    3.77637346   FALSE FALSE
    ## gyros_forearm_z       1.122917    1.56457038   FALSE FALSE
    ## accel_forearm_x       1.126437    4.04647844   FALSE FALSE
    ## accel_forearm_y       1.059406    5.11160942   FALSE FALSE
    ## accel_forearm_z       1.006250    2.95586586   FALSE FALSE
    ## magnet_forearm_x      1.012346    7.76679238   FALSE FALSE
    ## magnet_forearm_y      1.246914    9.54031189   FALSE FALSE
    ## magnet_forearm_z      1.000000    8.57710733   FALSE FALSE
    ## classe                1.469581    0.02548160   FALSE FALSE

    it <- findCorrelation(cor(train[,2:(ncol(train)-1)]), .70) 
    it <- it+1
    train <- train[, -it]

Do some visual checking of the data. We can see from gyros\_forearm\_x
that there is an outlier in the data. This outlier is in the row
numbered 5373. I will remove it from the data.

    temp <- train
    temp <- temp[,2:17]
    temp <- cbind(temp, classe = train[,"classe"])
    longtemp <- melt(temp, measure.vars = (1:ncol(temp)-1))
    ggplot(longtemp, aes(x=classe, y=value))+
      xlab("")+
      ylab("")+
      stat_boxplot(geom ='errorbar')+ 
      geom_boxplot() +
      theme(axis.line = element_line(), axis.text=element_text(color='black'), axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text(), legend.key = element_rect(colour = "black"))+
      facet_wrap(~ variable, ncol = 4, scales = "free")

![](Practical_Machine_Learning_Course_Project_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    temp <- train
    temp <- temp[,18:ncol(train)]
    longtemp <- melt(temp, measure.vars = (1:ncol(temp)-1))
    ggplot(longtemp, aes(x=classe, y=value))+
      xlab("")+
      ylab("")+
      stat_boxplot(geom ='errorbar')+ 
      geom_boxplot() +
      theme(axis.line = element_line(), axis.text=element_text(color='black'), axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text(), legend.key = element_rect(colour = "black"))+
      facet_wrap(~ variable, ncol = 4, scales = "free")

![](Practical_Machine_Learning_Course_Project_files/figure-markdown_strict/unnamed-chunk-4-2.png)

    row.names(train[train$gyros_forearm_x < -10,])

    ## [1] "5373"

    train <- train[-as.numeric(row.names(train[train$gyros_forearm_x < -10,])),]

First we split the data into training and testing sets. Then we set the
seed so that the analysis can be reproduced exactly. Then we use the
train function to set up 5-fold cross validation and train a random
forest algorithm on the training data using 250 trees.

    validation <- validation[,names(validation) %in% names(train)]

    index <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
    train_split <- train[index, ]
    train_test_split <- train[-index, ]

    ctrl <- trainControl(method = "cv",
                         number = 5)

    set.seed(1234)

    rf_mod <- train(classe ~ ., 
                    data = train_split,
                    method = "rf",
                    ntree = 250,
                    trControl = ctrl)

    rfClasses <- predict(rf_mod, train_test_split)

The accuracy of the model is about 0.99. We can be fairly confident that
we will predict the class of all 20 observations with perfectly.

    confusionMatrix(data = rfClasses, train_test_split$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668   23    1    0    0
    ##          B    2 1106   15    0    0
    ##          C    3   10 1004   19    0
    ##          D    0    0    6  941    2
    ##          E    0    0    0    4 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9856          
    ##                  95% CI : (0.9822, 0.9884)
    ##     No Information Rate : 0.2843          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9817          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9970   0.9710   0.9786   0.9761   0.9982
    ## Specificity            0.9943   0.9964   0.9934   0.9984   0.9992
    ## Pos Pred Value         0.9858   0.9849   0.9691   0.9916   0.9963
    ## Neg Pred Value         0.9988   0.9931   0.9955   0.9953   0.9996
    ## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1839
    ## Detection Rate         0.2835   0.1880   0.1706   0.1599   0.1835
    ## Detection Prevalence   0.2876   0.1909   0.1761   0.1613   0.1842
    ## Balanced Accuracy      0.9957   0.9837   0.9860   0.9873   0.9987

The predicted classes of these 20 unknown observations are shown here:

    result <- predict(rf_mod, validation)
    result

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
