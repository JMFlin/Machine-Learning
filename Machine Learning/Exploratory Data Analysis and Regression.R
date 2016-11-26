#https://www.r-bloggers.com/part-3b-eda-with-ggplot2/

library(psych)
library(ggplot2)
library(caret)
library(reshape2)
library(gridExtra)
library(GGally)
library(ggcorrplot)
library(caret)
library(rpart)
library(tidyr)
library(FactoMineR)
source("my_functions.R")

#Exaploratory Data Analysis

#Analyse the (continuous) dependent variable - at least, make an histogram of the distribution; when applicable, plot the variable against time to see the trend. 
#Can the continuous variable be transformed to a binary one (i.e., did it rain or not on a particular day?), or to a multicategorical one (i.e., was the rain none, light, moderate, or heavy on a particular day?).

#Search for correlations between the dependent variable and the continuous independent variables - are there any strong correlations? Are they linear or non-linear?

#Do the same as above but try to control for other variables (faceting in ggplot2 is very useful  to do this), in order to assess for confounding and effect modification. 
#Does the association between two continuous variables hold for different levels of a third variable, or is modified by them? 
#(e.g., if there is a strong positive correlation between the rain amount and the wind gust maximum speed, does that hold regardless of the season of the year, or does it happen only in the winter?)

#Search for associations between the dependent variable and the categorical independent variables (factors) 
#- does the mean or median of the dependent variable change depending on the level of the factor? What about the outliers, are they evenly distributed across all levels, or seem to be present in only a few of them? 

#day.count - number of days passed since the beginning of the year
#day - day of the month
#month - month of the year
#season - season of the year
#l.temp, h.temp, ave.temp - lowest, highest and average temperature for the day (in ºC)
#l.temp.time, h.temp.time - hour of the day when l.temp and h.temp occurred
#rain - amount of precipitation (in mm)
#ave.wind - average wind speed for the day (in km/h)
#gust.wind - maximum wind speed for the day (in km/h)
#gust.wind.time - hour of the day when gust.wind occurred
#dir.wind - dominant wind direction for the day

weather <- read.csv("weather_2014.csv",sep=";",stringsAsFactors=FALSE)
describe(weather)
str(weather)
table(is.na(weather))

weather$season <- factor(weather$season, levels = c("Spring","Summer","Autumn","Winter"))

weather$day <- as.factor(weather$day)
weather$month <- as.factor(weather$month)
weather$dir.wind <- as.factor(weather$dir.wind)


my_fn <- function(data, mapping, pts=list(), smt=list(), ...){
  ggplot(data = data, mapping = mapping, ...) + 
    do.call(geom_point, pts) +
    do.call(geom_smooth, smt) 
}

ggpairs(weather[,sapply(weather,is.numeric)], 
        lower = list(continuous = 
                       wrap(my_fn,
                            pts=list(colour="black"), 
                            smt=list(method="loess", se=F, colour="blue"))))


hist_func_cont(weather)
factor_table(weather)
sort(round(prop.table(table(weather$dir.wind))*100,1),decreasing = TRUE)# Making it relative (prop.table function)

#As long as the analyst knows to explain why and how some new variable was created, and bears in mind it may lack accuracy, 
#it is perfectly fine to add it to the data set. 
#It may be useful or useless, and that is something he will try to figure out during the stages of visualisation or modelling.


# Create a copy from the original variable...
weather$dir.wind.8 <- weather$dir.wind 

# ...and then simply recode some of the variables
weather$dir.wind.8 <- ifelse(weather$dir.wind %in%  c("NNE","ENE"),
                                 "NE",as.character(weather$dir.wind.8)) 

weather$dir.wind.8 <- ifelse(weather$dir.wind %in% c("NNW","WNW"),
                               "NW",as.character(weather$dir.wind.8)) 

weather$dir.wind.8 <- ifelse(weather$dir.wind %in% c("WSW","SSW"),
                               "SW",as.character(weather$dir.wind.8)) 

weather$dir.wind.8 <- ifelse(weather$dir.wind %in% c("ESE","SSE"),
                               "SE",as.character(weather$dir.wind.8)) 

# create factors, ordered by "levels" 
weather$dir.wind.8 <- factor(weather$dir.wind.8,
                                 levels = c("N","NE","E","SE","S","SW","W","NW"))

## A 2-way table (direction vs season), with relative frequencies calculated over column
round(prop.table(table(weather$dir.wind.8,weather$season),margin = 2)*100,1)

weather$day.count

#make date from dat variable

first.day <- "2014-01-01"
first.day <- as.Date(first.day)
weather$date  <- first.day + weather$day.count - 1 

#The last thing we need to do is to round (to the nearest hour) the time at which a certain event occurred  (lower temperature, higher temperature, and wind gust). 

l.temp.time.date <- as.POSIXlt(paste(weather$date, weather$l.temp.time), tz = "GMT")
head(l.temp.time.date)

# Round to the nearest hour
l.temp.time.date <- round(l.temp.time.date,"hours")

attributes(l.temp.time.date)

# Extract the value of the hour attribute as a number and add it to the data set
weather$l.temp.hour <- l.temp.time.date[["hour"]]

# Lastly, the integer is converted to factor
weather$l.temp.hour <- as.factor(weather$l.temp.hour)


h.temp.time.date <- as.POSIXlt(paste(weather$date, weather$h.temp.time), tz = "GMT")
head(h.temp.time.date)

h.temp.time.date <- round(h.temp.time.date,"hours")

weather$h.temp.hour <- h.temp.time.date[["hour"]]

weather$h.temp.hour <- as.factor(weather$h.temp.hour)


gust.wind.time.date <- as.POSIXlt(paste(weather$date, weather$gust.wind.time), tz = "GMT")
head(gust.wind.time.date)

gust.wind.time.date <- round(gust.wind.time.date,"hours")

weather$gust.wind.time.hour <- gust.wind.time.date[["hour"]]

weather$gust.wind.time.hour <- as.factor(weather$gust.wind.time.hour)

str(weather)

# This draws a scatter plot where season controls the colour of the point
ggplot(data = weather, aes(x = l.temp, y= h.temp, colour = season)) + geom_point()+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                                                          axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


# This draws a scatter plot but colour is now controlled by dir.wind (overwrites initial definition)
ggplot(data = weather, aes(x = l.temp, y= h.temp, colour = season)) + geom_point(aes(colour=dir.wind))+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                                                                              axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 



# Time series of average daily temperature, with smoother curve
ggplot(weather,aes(x = date,y = ave.temp)) +
  geom_point(colour = "blue") +
  geom_smooth(colour = "red",size = 1, method = "loess") +
  scale_y_continuous(limits = c(5,30), breaks = seq(5,30,5)) +
  ggtitle ("Daily average temperature") +
  xlab("Date") +  ylab ("Average Temperature ( ºC )")+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                             axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


# Same but with colour varying
ggplot(weather,aes(x = date,y = ave.temp)) + 
  geom_point(aes(colour = ave.temp)) +
  scale_colour_gradient2(low = "blue", mid = "green" , high = "red", midpoint = 16) + 
  geom_smooth(color = "red",size = 1, method = "loess") +
  scale_y_continuous(limits = c(5,30), breaks = seq(5,30,5)) +
  ggtitle ("Daily average temperature") +
  xlab("Date") +  ylab ("Average Temperature ( ºC )")+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                             axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 




# Distribution of the average temperature by season - density plot
ggplot(weather,aes(x = ave.temp, colour = season)) +
  geom_density() +
  scale_x_continuous(limits = c(5,30), breaks = seq(5,30,5)) +
  ggtitle ("Temperature distribution by season") +
  xlab("Average temperature ( ºC )") +  ylab ("Probability")+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                                    axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 


#Spring and autumn seasons are often seen as the transition from cold to warm and warm to cold days, respectively. 
#The spread of their distributions reflect the high thermal amplitude of these seasons. 
#On the other hand, winter and summer average temperatures are much more concentrated around a few values, and hence the peaks shown on the graph.


# Label the months - Jan...Dec is better than 1...12

weather$month = factor(weather$month,
                       labels = c("Jan","Feb","Mar","Apr",
                                  "May","Jun","Jul","Aug","Sep",
                                  "Oct","Nov","Dec"))

# Distribution of the average temperature by month - violin plot,
# with a jittered point layer on top, and with size mapped to amount of rain

ggplot(weather,aes(x = month, y = ave.temp)) +
  geom_violin(fill = "orange") +
  geom_point(aes(size = rain), colour = "blue", position = "jitter") +
  ggtitle ("Temperature distribution by month") +
  xlab("Month") +  ylab ("Average temperature ( ºC )")+ theme(axis.line = element_line(), axis.text=element_text(color='black'), 
                                                              axis.title = element_text(colour = 'black'), legend.text=element_text(), legend.title=element_text()) 

#A violin plot is sort of a mixture of a box plot with a histogram (or even better, a rotated density plot). 
#I also added a point layer on top (with jitter for better visualisation), in which the size of the circle is mapped to the continuous variable representing the amount of rain.


# Scatter plot of low vs high daily temperatures, with a smoother curve for each season
my_fn <- function(data, mapping, ...){
  p <- ggplot(data = weather, mapping = mapping) + #NOTE THE WEATHER
    geom_point(colour = "firebrick", alpha = 0.3) + 
    geom_smooth(aes(colour = season),se= F, method = 'loess')#NOTE THE SEASON
  p
}

ggpairs(weather[,sapply(weather,is.numeric)], lower = list(continuous = my_fn))

# Time series of the daily rain amount, with smoother curve

ggplot(weather, aes(date,rain)) +
  geom_point(aes(colour = rain)) +
  geom_smooth(colour = "blue", size = 1, method = 'loess') +
  scale_colour_gradient2(low = "green", mid = "orange",high = "red", midpoint = 20) +
  scale_y_continuous(breaks = seq(0,80,20)) +
  xlab("Month") +
  ylab("Rain (mm)") +
  ggtitle("Daily rain amount")


# Histogram of the daily rain amount

ggplot(weather,aes(rain)) + 
  geom_histogram(binwidth = 1,colour = "blue", fill = "darkgrey") +
  xlab("Rain (mm)") +
  ylab ("Frequency (days)") +
  ggtitle("Daily rain amount distribution")


#The time series plot shows that the daily rain amount varies wildly throughout the year. 
#There are many dry days interspersed with stretches of consecutive wet ones, often severe, especially in the autumn and winter seasons. 
#The histogram not only confirms what was said above, but also shows that the distribution is extremely right-skewed. 
#As shown below, both informally (comparing the mean to the median), and formally (calculating the actual value), the positive skewness remains even after removing all the days where it did not rain.

ggplot(weather,aes(log(rain+1))) + 
  geom_histogram(binwidth = 1,colour = "blue", fill = "darkgrey") +
  xlab("Rain (mm)") +
  ylab ("Frequency (days)") +
  ggtitle("Daily rain amount distribution")

# Heavily right-skewed distribution
describe(weather$rain)

# Right-skewness is still there after removing all the dry days
describe(subset(weather, rain > 0)$rain)


#It should be clear at this point that one possible approach would be to dichotomise the dependent variable (rain vs. no rain).  
#Note that it is common to consider days with rain only those where the total amount was at least 1mm (to allow for measurement errors), and that's the criterion we will adopt here. 
#Here is the code to do it and a few interesting summaries.

weather$rained <- ifelse(weather$rain >= 1, "Yes", "No")
table(weather$rained)
prop.table(table(weather$rained)) 

##Looking at the association between rain and season of the year
#The time series plot seems to indicate that season of the year might be a good predictor for the occurrence of rain. 
#Let's start by investigating this relation, with both the continuous rain variable and the binary one.

# Jitter plot - Rain amount by season 

ggplot(weather, aes(season,rain)) +
  geom_jitter(aes(colour=rain), position = position_jitter(width = 0.2)) +
  scale_colour_gradient2(low = "blue", mid = "red",high = "black", midpoint = 30) +
  xlab("Season") +
  ylab ("Rain (mm)") +
  ggtitle("Daily rain amount by season")

#We can see that most of the extreme values (outliers) are in the winter and autumn. 
#There are still, however, many dry days in both these seasons, and hence the means are too close from each other. 
#Fitting a model to predict the actual amount of rain (not the same as probability of occurrence) based on the season alone might not give the greatest results.

# Bar plot - dry and wet days by season (relative)
ggplot(weather,aes(season)) +
  geom_bar(aes(fill = rained), position = "fill") +
  geom_hline(aes(yintercept = prop.table(table(weather$rained))["Yes"]),
             linetype = "dashed") +
  annotate("text", x = 1, y = 0.45, label = "Yearly avg") +
  xlab("Season") +
  ylab ("Proportion") +
  ggtitle("Proportion of days without and with rain, by season")

#It appears that, when it comes to calculate the likelihood of raining on a particular day, the season of the year may have some predictive power. 
#This is especially true for the winter season, where the rainy days (63%) are well above the yearly average (40%).


#Looking at the correlations between rain and all numeric variables

#We are now going to calculate the linear (Pearson) correlations between the continuous outcome variable (daily rain amount) and all the numeric variables - 
#both the actual numeric ones (such as the temperature and wind speed), and those that are actually factor variables, 
#but still make sense if we want to see them as numeric (for instance, the hour of the day in which some event occurred).
str(weather)
weather$l.temp.hour <- as.numeric(weather$l.temp.hour)
weather$h.temp.hour <- as.numeric(weather$h.temp.hour)
weather$gust.wind.time.hour <- as.numeric(weather$gust.wind.time.hour)
corr_num_both(weather, 3)
#Note that we are not modelling at this stage, just trying to gain some insight from the data, 
#and therefore we are neither concerned about whether the correlations are significant (p-values and/or confidence intervals) 
#nor if the the relation between any two variables is in fact linear (we will be able to check this later, anyway, when we create the scatter plots with the lowess curves on top).

#Always bear in mind that correlation does not imply causation, therefore while it is true that the wind correlates with rain, this does not necessarily mean that the wind itself is causing the rain. 
#It could actually be the other way around or, what is very common, both of the variables are caused by a third one that we are not even considering.
#There are also some negative correlations with the temperatures (especially the daily high) that, even though not as strong as the wind ones, are still worth looking at. 
#It appears higher amounts of rain are correlated with lower high temperatures. 
#But let's think about it for a minute: we saw that it is in the winter when it rains the most, and in Part 3a we also saw that the temperatures were lower in the winter. 
#Here is an example of a potential (yet to be confirmed) interaction with a third variable: 
#it may not be the lower temperature by itself that causes more rain, but the fact the both the precipitation and 
#lower temperatures tend to occur during a specific period of the year.
#Since the season seems to have an impact on these variables, I would like to explore it a bit further, calculating all these correlations by season and check whether the values hold. 
#If the correlation between rain and some other variable is very dissimilar across all seasons, then there is the proof for an interaction.

weather.num.season <- split(weather[,sapply(weather,is.numeric)],weather$season)
summary(weather.num.season)
attributes(weather.num.season)

# {sapply(x,z) applies function z to each element of x}
# First go over the elements of the list and calculate the correlation matrix (all against all)
# For each season, return only the correlation between "rain" and everything else
sapply(weather.num.season, function (x) round(cor(x)["rain",],2))

#What do you conclude from the table above? 
#The correlation between the rain and wind varies, but keeps moderately strong, regardless of the season of the year, making this the most promising variable in out data set; 
#the correlation between the rain and daily high temperature does not confirm what I had hypothesised above. 
#In fact, the correlation is even stronger in the spring than in the winter, 
#and we would have to go even deeper if we really needed to understand what is going on (keep in mind we are just analysing one of the possible interactions - the season - 
#but in practice there can be multiple ones). 
#For the purpose of what we are doing now, it is enough to be aware that this correlation is not stable throughout the year, and actually goes to zero during the autumn. 
#Lastly, the correlations between rain and the hour of the events (low and high temperatures, and wind gust) are rather weak but show some stability (see the l.temp.hour and h.temp.hour). 
#They might have some predictive power, at least in a linear model.
#Now that we know that the wind has the strongest correlation with the rain, and that it holds true across all seasons, 
#it's time to plot these variables, because we want to learn something we still don't know: what is the shape of this relation? 
#Linear, piecewise linear, curvilinear?

#Plot the high correlations

# Amount of rain vs high temp, by season

ggplot(weather,aes(h.temp, rain)) +
  geom_point(colour = "firebrick") +
  geom_smooth(size = 0.75, se = F, method = 'loess') +
  facet_wrap(~season) +
  xlab("Maximum wind speed (km/h)") +
  ylab ("Rain (mm)") +
  ggtitle("Amount of rain vs. Highest Temperature, by season")

# Amount of rain vs. wind, by season

ggplot(weather,aes(gust.wind,rain)) +
  geom_point(colour = "firebrick") +
  geom_smooth(size = 0.75, se = F, method = 'loess') +
  facet_wrap(~season) +
  xlab("Maximum wind speed (km/h)") +
  ylab ("Rain (mm)") +
  ggtitle("Amount of rain vs. maximum wind speed, by season")

#This plot confirms what we had already discovered: there is a positive correlation between rain and wind, and the association holds regardless of the season. 
#But now we know more: this correlation is non-linear. In fact, if we were to generalise, we could say there is no correlation at all when the maximum wind speed is below 25 km/h. 
#For values higher than that, there seems to be a linear association in the autumn and winter, not so linear in the spring, and definitely non-linear during the summer. 
#If we wanted to model this relation, we would either fit a non-linear model (such as a regression tree) 
#or we could try to force a piecewise linear model (linear spline), where the equation relating the outcome to the predictors would, itself, be different depending on the value of the wind.



# let's check whether the variable that seemed to have some predictive power for the amount of rain (maximum wind speed), is also good in the case of a binary outcome (occurrence of rain), 
#but now we will not only control for the season, but also for the daily high temperature (because, as we have seen before, this variable was interacting with both the rain and the season). 
#We will do this simultaneously, using the faceting technique on two variables. 
#But, since the daily high temperature variable is continuous, we need to transform it to categorical first. 
#A common strategy is to split the continuous variables in four groups of (roughly) the same size, i.e., the quartiles.

quantile(weather$h.temp)
weather$h.temp.quant <- cut(weather$h.temp, breaks = quantile(weather$h.temp),
                            labels = c("Cool","Mild","Warm","Hot"),include.lowest = T)
table(weather$h.temp.quant)

ggplot(weather,aes(rained,gust.wind)) +
  stat_boxplot(geom ='errorbar')+geom_boxplot(aes(colour=rained)) +
  facet_grid(h.temp.quant~season) +
  xlab("Occurrence of rain") +
  ylab ("Maximum wind speed (km/h)") +
  ggtitle("Occurrence of rain, by season and daily high temperature")


#The graph reveals a clear pattern: the median of the maximum wind speed is always higher when it rains, 
#and this is not affected by the range the of daily high temperature, 
#even after controlling for the temperature variation within each season of the year.
#I think we now have a much better understanding of the data. We know which variables matter the most, 
#and which ones seem to be useless, when it comes to predict the rain, either the actual amount or the probability of its occurrence. 
#Please note that it would be impossible to write about the analysis of every single variable and show every plot; 
#behind the scenes, much more work than what I've shown here has been done.

factor_boxplot(weather, weather$rain, weather$season, xlab = "Season", ylab = "Rain (mm)")
factor_boxplot(weather, weather$rain,weather$h.temp.quant, xlab = "Temperature", ylab = "Rain (mm)")
factor_boxplot(weather, weather$rain,weather$dir.wind.8, xlab = "Wind Direction", ylab = "Rain (mm)")

pcp_func(weather, Ycol = c(20), notXcols = c(10))
#This confirms that wind is a good predictor for rain

pca_num(weather, -c(1))
pca_num2(weather, cumu = TRUE, -c(1))


#We will build several predictive models and evaluate their accuracies. Now our dependent value will be continuous, and we will be predicting the daily amount of rain. 
#Later we will deal with the case of a binary outcome, which means we will assign probabilities to the occurrence of rain on a given day. 
#In both the continuous and binary cases, we will try to fit the following models:
#Baseline model - usually, this means we assume there are no predictors (i.e., independent variables). 
#Thus, we have to make an educated guess (not a random one), based on the value of the dependent value alone. 
#Majority class in classification and mean in regression.

#A model from inferential statistics - this will be a (generalised) linear model. 
#In the case of a continuous outcome, we will fit a multiple linear regression; for the binary outcome, the model will be a multiple logistic regression;
#Two models from machine learning - we will first build a decision tree (regression tree for the continuous outcome, and classification tree for the binary case); 
#these models usually offer high interpretability and decent accuracy; 
#then, we will build random forests, a very popular method, where there is often a gain in accuracy, at the expense of interpretability.
#This model is important because it will allow us to determine how good, or how bad, are the other ones.


# For reproducibility; 123 has no particular meaning
set.seed(123)

index <- createDataPartition(weather$rain, p = 2/3, list = FALSE)
train <- weather[ index,]
test  <- weather[-index,]

#Here is a plot showing which points belong to which set (train or test).

# Create a dataframe with train and test indicator...
group <- rep(NA,nrow(train) + nrow(test))
group <- ifelse(seq(1,nrow(train) + nrow(test)) %in% index,"Train","Test")
df <- data.frame(date=weather$date,rain=weather$rain, group)

# ...and plot it
ggplot(df,aes(x = date,y = rain, color = group)) + geom_point() +
  scale_color_discrete(name="") + theme(legend.position="top")

#For the continuous outcome, the main error metric we will use to evaluate our models is the RMSE (root mean squared error).
#This error measure gives more weight to larger residuals than smaller ones
#We will use the MAE (mean absolute error) as a secondary error metric. It gives equal weight to the residuals

# Baseline model - predict the mean of the training data
best.guess <- mean(train$rain) 

# Evaluate RMSE and MAE on the testing data
RMSE.baseline <- sqrt(mean((best.guess-test$rain)^2))
RMSE.baseline

MAE.baseline <- mean(abs(best.guess-test$rain))
MAE.baseline

#We will now fit a (multiple) linear regression, which is probably the best known statistical model.

#Linear models do not require variables to have a Gaussian distribution (only the errors / residuals must be normally distributed); 
#they do require, however, a linear relation between the dependent and independent variables. 
#Like we saw, the distribution of the amount of rain is right-skewed, and the relation with some other variables is highly non-linear. 
#For this reason of linearity, and also to help fixing the problem with residuals having non-constant variance across the range of predictions (called heteroscedasticity), 
#we will do the usual log transformation to the dependent variable. 
#Since we have zeros (days without rain), we can't do a simple ln(x) transformation, but we can do ln(x+1), where x is the rain amount. 
#Why do we choose to apply a logarithmic function? Simply because the regression coefficients can still be interpreted, although in a different way when compared with a pure linear regression. 
#This model we will fit is often called log-linear;

#I started with all the variables as potential predictors and then eliminated from the model, one by one, those that were not statistically significant (p < 0.05). 
#We need to do it one by one because of multicollinearity (i.e., correlation between independent variables). 
#Some of the variables in our data are highly correlated (for instance, the minimum, average, and maximum temperature on a given day), 
#which means that sometimes when we eliminate a non-significant variable from the model, another one that was previously non-significant becomes statistically significant. 
#This iterative process of backward elimination stops when all the variables in the model are significant (in the case of factors, here we consider that at least one level must be significant);


#Our dependent variable has lots of zeros and can only take positive values; if you're an expert statistician, perhaps you would like to fit very specific models that can deal better with count data,
#such as negative binomial, zero-inflated, Poisson regression, and hurdle models and mixture models. There are several packages to do it in R.


# Create a multiple (log)linear regression model using the training data
#lin.reg <- lm(log(rain+1) ~ season +  h.temp + ave.temp + ave.wind + gust.wind +
#                  dir.wind + as.numeric(gust.wind.time.hour), data = train)
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 10, #  number = 10, repeats = 10 is 10 fold cv
                     savePredictions = TRUE)

set.seed(1537)
lin.reg <- train(log(rain+1) ~ season +  h.temp + ave.temp + ave.wind + gust.wind +
                      dir.wind + as.numeric(gust.wind.time.hour), 
                    data = train,
                    method = "lm", #preProcess=c("BoxCox") or preProcess=c("center","scale")
                    metric = "RMSE",
                    trControl = ctrl)


# ...and evaluate the accuracy
RMSE.lin.reg <- sqrt(mean((test.pred.lin-test$rain)^2))
RMSE.lin.reg

# Inspect the model
summary(lin.reg)

# What is the multiplicative effect of the wind variable?
exp(coef(lin.reg$finalModel)["gust.wind"])

# Apply the model to the testing data (i.e., make predictions) ...
# (Don't forget to exponentiate the results to revert the log transformation)
test.pred.lin <- exp(predict(lin.reg,test))-1
test.pred.lin <- ifelse(test.pred.lin < 0 , 0, test.pred.lin)

# ...and evaluate the accuracy
RMSE.lin.reg <- sqrt(mean((test.pred.lin-test$rain)^2))
RMSE.lin.reg

MAE.lin.reg <- mean(abs(test.pred.lin-test$rain))
MAE.lin.reg


lm.pred <- data.frame(date = test$date, actual = test$rain,
                       predicted = test.pred.lin)
lm.pred <- melt(lm.pred, c("actual", "predicted"), id = "date")

ggplot(lm.pred, aes(x = date, y = value, color = variable))+
   geom_point() + scale_color_discrete(name="") + theme(legend.position="top") + ylab("Rain (mm)") + xlab("Date")

#he R-squared is 0.66, which means that 66% of the variance in our dependent variable can be explained by the set of predictors in the model; 
#at the same time, the adjusted R-squared is not far from that number, meaning that the original R-squared has not been artificially increased by adding variables to the model. 
#Note that the R-squared can only increase or stay the same by adding variables, whereas the adjusted R-squared can even decrease if the variable added doesn't help the model more than what is expected by chance;
#All the variables are statistically significant (p < 0.05), as expected from the way the model was built, and the most significant predictor is the wind gust (p = 7.44e-12). 
#The advantage of doing a log transformation is that, if the regression coefficient is small (i.e. -0.1 to 0.1), a unit increase in the independent variable yields an increase of approximately coeff*100% in the dependent variable.
#To be clear, the coefficient of the wind gust is 0.062181. It means that a unit increase in the gust wind (i.e., increasing the wind by 1 km/h), increases the predicted amount of rain by approximately 6.22%. 
#You can always exponentiate to get the exact value (as I did), and the result is 6.42%. 
#By the same token, for each degree (ºC) the daily high temperature increases, the predicted rain increases by exp(-0.197772) = 0.82 (i.e., it decreases by 18%);


#How do we grow trees, then? 
#In very simple terms, we start with a root node, which contains all the training data, and split it into two new nodes based on the most important variable (i.e., the variable that better separates the outcome into two groups).
#The disadvantage of a decision tree is that they tend to overfit

set.seed(1537)
# rpart function applied to a numeric variable => regression tree
rt <- rpart(rain ~ month + season + l.temp + h.temp + ave.temp + ave.wind +
              gust.wind + dir.wind + dir.wind.8 + as.numeric(h.temp.hour)+
              as.numeric(l.temp.hour)+ as.numeric(gust.wind.time.hour), data=train)

plot(as.party(rt))

test.pred.rtree <- predict(rt,test) 

RMSE.rtree <- sqrt(mean((test.pred.rtree-test$rain)^2))
RMSE.rtree


MAE.rtree <- mean(abs(test.pred.rtree-test$rain))
MAE.rtree

#Now that we have a full-grown tree, let's see if it's possible to prune it.

# Check cross-validation results (xerror column)
# It corresponds to 2 splits and cp = 0.088147
printcp(rt)

# Get the optimal CP programmatically...
min.xerror <- rt$cptable[which.min(rt$cptable[,"xerror"]),"CP"]
min.xerror

# ...and use it to prune the tree
rt.pruned <- prune(rt,cp = min.xerror) 

# Evaluate the new pruned tree on the test set
test.pred.rtree.p <- predict(rt.pruned,test)
RMSE.rtree.pruned <- sqrt(mean((test.pred.rtree.p-test$rain)^2))
RMSE.rtree.pruned

MAE.rtree.pruned <- mean(abs(test.pred.rtree.p-test$rain))
MAE.rtree.pruned

tree.pred <- data.frame(date = test$date, actual = test$rain,
                      predicted = test.pred.rtree.p)

tree.pred <- melt(tree.pred, c("actual", "predicted"), id = "date")

ggplot(tree.pred, aes(x = date, y = value, color = variable))+
  geom_point() + scale_color_discrete(name="") + theme(legend.position="top") + ylab("Rain (mm)") + xlab("Date")


#As you can see, we were able to prune our tree, from the initial 8 splits on six variables, to only 2 splits on one variable (the maximum wind speed), 
#gaining simplicity without losing performance (RMSE and MAE are about equivalent in both cases). 
#In the final tree, only the wind gust speed is considered relevant to predict the amount of rain on a given day, and the generated rules are as follows (using natural language):
#If the daily maximum wind speed exceeds 52 km/h (4% of the days), predict a very wet day (37 mm);
#If the daily maximum wind is between 36 and 52 km/h (23% of the days), predict a wet day (10mm);
#If the daily maximum wind stays below 36 km/h (73% of the days), predict a dry day (1.8 mm); 
#The accuracy of this extremely simple model is only a bit worse than the much more complicated linear regression.


train$h.temp.hour <- as.numeric(train$h.temp.hour)
train$l.temp.hour <- as.numeric(train$l.temp.hour)
train$gust.wind.time.hour <- as.numeric(train$gust.wind.time.hour)
test$h.temp.hour <- as.numeric(test$h.temp.hour)
test$l.temp.hour <- as.numeric(test$l.temp.hour)
test$gust.wind.time.hour <- as.numeric(test$gust.wind.time.hour)

set.seed(123)

grid <- data.frame(mtry = seq(1, 21, 4))

rf <- train(rain ~ month + season + l.temp + h.temp + ave.temp + ave.wind +
                     gust.wind + dir.wind + dir.wind.8 + h.temp.hour + l.temp.hour +
                     gust.wind.time.hour, 
                   ntree=1000,
                   data = train,
                   method = "rf",
                   trControl = ctrl,
                   metric="RMSE",
                   tuneGrid = grid,
                   importance = TRUE)

ggplot(rf)
rfFitTime
imp <- varImp(rf)
ggplot(imp)

test.pred.forest <- predict(rf,test)
RMSE.forest <- sqrt(mean((test.pred.forest-test$rain)^2))
RMSE.forest

MAE.forest <- mean(abs(test.pred.forest-test$rain))
MAE.forest

rf.pred <- data.frame(date = test$date, actual = test$rain,
                        predicted = test.pred.forest)
rf.pred <- melt(rf.pred, c("actual", "predicted"), id = "date")

ggplot(rf.pred, aes(x = date, y = value, color = variable))+
  geom_point() + scale_color_discrete(name="") + theme(legend.position="top") + ylab("Rain (mm)") + xlab("Date")


#We have just built and evaluated the accuracy of five different models: baseline, linear regression, fully-grown decision tree, pruned decision tree, and random forest. 
#Let's create a data frame with the RMSE and MAE for each of these methods.

# Create a data frame with the error metrics for each method
accuracy <- data.frame(Method = c("Baseline","Linear Regression","Full tree","Pruned tree","Random forest"),
                           RMSE   = c(RMSE.baseline,RMSE.lin.reg,RMSE.rtree,RMSE.rtree.pruned,RMSE.forest),
                           MAE    = c(MAE.baseline,MAE.lin.reg,MAE.rtree,MAE.rtree.pruned,MAE.forest)) 

# Round the values and print the table
accuracy$RMSE <- round(accuracy$RMSE,2)
accuracy$MAE <- round(accuracy$MAE,2) 

accuracy


#All methods beat the baseline, regardless of the error metric, with the random forest and linear regression offering the best performance. 
#It would be interesting, still, to compare the fitted vs. actual values for each model.
# Create a data frame with the predictions for each method
all.predictions <- data.frame(actual = test$rain,
                              baseline = best.guess,
                              linear.regression = test.pred.lin,
                              full.tree = test.pred.rtree,
                              pruned.tree = test.pred.rtree.p,
                              random.forest = test.pred.forest)

# First six observations 
head(all.predictions)


# Needed to melt the columns with the gather() function 
# tidyr is an alternative to the reshape2 package.
# Gather the prediction variables (columns) into a single row (i.e., wide to long)
# Recall the ggplot2 prefers the long data format
all.predictions <- gather(all.predictions,key = model,value = predictions,2:6)
head(all.predictions)

# Predicted vs. actual for each model
ggplot(data = all.predictions,aes(x = actual, y = predictions)) + 
  geom_point(colour = "blue") + 
  geom_abline(intercept = 0, slope = 1, colour = "red") +
  geom_vline(xintercept = 23, colour = "green", linetype = "dashed") +
  facet_wrap(~ model,ncol = 2) + 
  coord_cartesian(xlim = c(0,70),ylim = c(0,70)) +
  ggtitle("Predicted vs. Actual, by model")


#The graph shows that none of the models can predict accurately values over 25 mm of daily rain. 
#In the range from 0 to approximately 22 mm (vertical dashed line for reference), 
#the random forest seems to be the method that better approximates the diagonal line (i.e., smaller prediction error), followed closely by the linear regression. 
#Even though these models can not be considered more than fair, they still do a much better job when compared with the baseline prediction.

#we will continue building models, this time considering the rain as a binary outcome. This means all models will assign probabilities to the occurrence of rain, for each day in the test set.
#Also time series dimension so using the lagged value of rain as an additional predictor
#Different types of regressions such as poisson, quasi-poisson, negative binomial, zero-inflated, and hurdle models and mixture models and ordinal regression on Y binned into 5 categories.

#http://www.ats.ucla.edu/stat/r/dae/nbreg.htm
#http://www.ats.ucla.edu/stat/r/dae/zipoisson.htm