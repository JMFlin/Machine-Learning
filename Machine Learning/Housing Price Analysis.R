#http://ucanalytics.com/blogs/regression-analysis-pricing-case-study-example-part-1/
#log transform is used to make a screwed distribution normal and boxcox
#In the case of categorical variables, a simple cross table, 
#and chi-square test can reveal a lot about a significant large collinearity.
source("my_functions.R")
source("my_packages.R")

my_data <- read.csv("Regression-Analysis-Data.csv", sep = ",", header = T)

str(my_data)
summary(my_data)
head(my_data)
describe(my_data)
table(is.na(my_data))

pairs(my_data, gap=0, pch=19, cex=0.4, col="darkblue", panel = "panel.smooth")
#outlier
fit <- lm(House_Price~Rainfall , data= my_data)
par(mfrow = c(2,2))
plot(fit)
par(mfrow = c(1,1))
#remove it!
my_data <- my_data[-361,]
pairs(my_data, gap=0, pch=19, cex=0.4, col="darkblue", panel = "panel.smooth")

new_data <- na_test_data(my_data)
hist_func_cont(new_data)

imp_data <- impute_numint(my_data, type = "bagImpute")#bagged trees can also be used to impute. 
#For each predictor in the data, a bagged tree is created using all of the other predictors in the training set. 
#When a new sample has a missing predictor value, the bagged model is used to predict the value.
factorize(imp_data)
pairs(imp_data, gap=0, pch=19, cex=0.4, col="darkblue", panel = "panel.smooth")
hist_func_cont(imp_data)
hist_func_fact(imp_data)
factor_table(imp_data)
#I think we should impute first

factor_boxplot(imp_data$House_Price, imp_data$City_Category, t.test = TRUE)
#You have 2 categorical variables in our data-set: city category and parking availability. 
#A good way to analyse categorical predictor variables and numeric response variable is through a box plot.

#We can clearly see that there is a significant difference between the average price of houses based on the category of cities. 
#The average prices are shown as in the middle of the boxes.
#Moreover to validate what you see in the box plot, you have performed pair-wise t test for each category. 
#The results for pair-wise t-test shown at the top of the box plot in red. 
#You noticed that P(A=B)~0 means that there are almost 0% chances that average price of houses in cat A city is equal to cat B city. 


corr_num_both(imp_data, 3)#cross-table or chi-square test for categorical variables
dev.off()


#I first briefly examine the data. I notice that the variables have vastly different means.
round(apply(imp_data[,sapply(imp_data,is.numeric)], 2, mean),3)
round(apply(imp_data[,sapply(imp_data,is.numeric)], 2, var),3)
#Not surprisingly, the variables also have vastly different variances
#Thus, it is important to standardize the
#variables to have mean zero and standard deviation one before performing PCA.


pca_num(imp_data,-c(1,8))
#Here, distance to taxi, market, and hospital have formed a composite variable (comp 1) which explains 37.7% information in data. 
#Another, orthogonal axis (comp 2) explains the remaining 33.4% of variation through the composite of carpet and built-up area. 
#Rainfall is not a part of comp 1 or comp 2 but is a 3rd orthogonal component. 

#We have got the percentage of data (variance) explained by each component.

#This correlation matrix tells us that 88% of the distance to the hospital is loaded on comp 1. 

#100% of both carpet and built-up area is loaded on comp 2.

#If we remove component 4 to 6 from our data we will lose a little over 10% of the information. 
#This also means that ~39% of the information available in 'distance to market' will be lost with component 4 & 5.

#Now, you are feeling much more confident that you have addressed the issues of multicollinearity in the numeric predictor variables. 

pca_num2(imp_data, cumu = TRUE, -c(1,8))

metric_mds(imp_data)

new_imp_data <- decile_func(imp_data$House_Price, imp_data, n=2)

data_list <- split_binaryY(new_imp_data$decile, new_imp_data, prob = c(0.7, 0.3), is.same.size = TRUE)
table(data_list$training$decile)
table(data_list$testing$decile)

pcp_func(new_imp_data, Ycol = c(11), notXcols = c(1,10))
