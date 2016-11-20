#Attribute Information:
# 1. BI-RADS assessment: 1 to 5 (ordinal)  
# 2. Age: patient's age in years (integer)
# 3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
# 4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
# 5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
# 6. Severity: benign=0 or malignant=1 (binominal)


file <- "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
cancer.data <- read.csv(file, header=FALSE, na.strings = "?")

colnames(cancer.data) <- c("BI.RADS", "Age", "Shape", "Margin", "Density", "Diagnosis")

#Remove rows with missing data
cancer.data <- na.omit(cancer.data)

#Factorize nominal features
class(cancer.data$Shape) <- factor(cancer.data$Shape)
class(cancer.data$Margin) <- factor(cancer.data$Shape)

#Derive train and test sets
cancer.data <- cancer.data[sample(nrow(cancer.data)), ]
cut.off <- floor(.7*nrow(cancer.data))
train.set <- cancer.data[1:cut.off, ]
test.set <- cancer.data[(cut.off+1):nrow(cancer.data), ]

#Train model on train set
glm.1 <- glm(Diagnosis ~ Age + Shape + Margin + Density, family = binomial, data = train.set)

#Derive predicted probabilities from test set
pr.1 <- predict(glm.1, newdata = test.set, type = "response")

pr.1 <- cbind(pr.1, test.set)

#Derive discrete class prediction
pr.1$Prediction <- ifelse(pr.1$pr.1 > .5, 1, 0)

#Calculate simple performance metrics
tp <- nrow(pr.1[pr.1$Diagnosis==1 & pr.1$Prediction==1, ])
tn <- nrow(pr.1[pr.1$Diagnosis==0 & pr.1$Prediction==0, ])
fp <- nrow(pr.1[pr.1$Diagnosis==0 & pr.1$Prediction==1, ])
fn <- nrow(pr.1[pr.1$Diagnosis==1 & pr.1$Prediction==0, ])

accuracy <- (tp+tn)/nrow(pr.1)
precision <- tp/(tp+fp)
recall <- tp/(tp+fn)
f1 <- 2*precision*recall/(precision+recall)

#See image for explanations
accuracy
precision
recall
f1
