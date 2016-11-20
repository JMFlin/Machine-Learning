library(ggplot2)
iris.2 <- iris[iris$Species != "setosa", ]

qplot(x = Sepal.Width, y = Petal.Length, color = Species, data = iris.2)

GLM.1 <- glm(Species ~ Sepal.Width + Petal.Length, family = "binomial", data = iris.2)
summary(GLM.1)

slope <- coef(GLM.1)[2]/(-coef(GLM.1)[3])

intercept <- coef(GLM.1)[1]/(-coef(GLM.1)[3])

qplot(x = Sepal.Width, y = Petal.Length, 
      color = Species, data = iris.2) + geom_abline(intercept = intercept, slope = slope)
#The line is the models decision boundry

#GLM is a good starting point for most binary problems
#It provides easily interpretable results