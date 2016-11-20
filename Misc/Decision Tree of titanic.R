library(reshape2)
library(plyr)
library(rpart)

titanic <- melt(Titanic)
titanic <- titanic[titanic$value > 0, ]

titanic <- ddply(.data = titanic,
                 .variables = c("Class", "Sex", "Age", "Survived"),
                 .fun = function(x){
                   n <- x$value[1]
                   df1 <- data.frame(Class = rep(x$Class[1], n),
                                     Sex = rep(x$Sex[1], n),
                                     Age = rep(x$Age[1], n),
                                     Survived = rep(x$Survived[1], n))
                   return(df1)
                 })
  
tree.1 <- rpart(Survived ~ Class + Sex + Age, method = "class", data = titanic)

plot(tree.1, uniform = TRUE, main = "Prbability of Survival on Titanic")
text(tree.1, use.n = TRUE, all = TRUE, cex = 0.6)
