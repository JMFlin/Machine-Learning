install.packages("neuralnet")
library(neuralnet)

iris$Species.Class <- 0
iris$Species.Class[iris$Species == "versicolor"] <- 1
iris$Species.Class[iris$Species == "virginica"] <- 2
nn.1 <- neuralnet(Species.Class ~ Sepal.Width + Sepal.Length + Petal.Width + Petal.Length,
                  data = iris, hidden = 2)
plot(nn.1)
nn.1
