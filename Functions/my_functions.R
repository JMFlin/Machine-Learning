#to do
#auto.arima
#multivariate methods
#robust stats
#TRADING stuff - Big Mikes HW thread. Range stuff etc
#Weekday returns into wide format and check calculations. Diff?
#Drawdown from peak see performance analytics

#FUNCTIONS


#Missing value coulmns to data frame for hists and na.omit
na_test_data <- function(data){
  
  a <- names
  for(i in 1:ncol(data)){
    if(class(data[,i]) == "integer" || class(data) == "numeric"){
      if(sum(is.na(data[,i])) > 0){
        a <- append(a, colnames(data)[i])
      }
    }
  }
  
  datalist <- list()
  for(i in 2:length(a)){
    dat <- data.frame(data[,a[[i]]])
    datalist[[i-1]] <- dat # add it to your list
  }
  
  new_data <- do.call(cbind, datalist)
  for(i in 1:length(new_data)){
    names(new_data)[i] <- a[[i+1]]
  }
  new_data <- na.omit(new_data)
  return(new_data)
}

#Histograms
hist_func_cont <- function(data){

  for(i in 1:ncol(data)){
    if(class(data[,i]) != "factor"){
      breaks <- pretty(range(data[,i]), n = nclass.Sturges(data[,i]), min.n = 1)
      bwidth <- breaks[2]-breaks[1]
      df <- data.frame(data[,i])
      a <- ggplot(df,aes(data[,i]))+geom_histogram(binwidth=bwidth,fill="white",colour="black")+
        labs(title = paste(names(data)[i],"- Sturges' Method"))+
        theme(plot.title = element_text(size = rel(1.5)))
    
    
      breaks <- pretty(range(data[,i]), n = nclass.scott(data[,i]), min.n = 1)
      bwidth <- breaks[2]-breaks[1]
      df <- data.frame(data[,i])
      b <- ggplot(df,aes(data[,i]))+geom_histogram(binwidth=bwidth,fill="white",colour="black")+
        labs(title = paste(names(data)[i],"- Scott's Method"))+
        theme(plot.title = element_text(size = rel(1.5)))
    
    
      breaks <- pretty(range(data[,i]), n = nclass.FD(data[,i]), min.n = 1)
      bwidth <- breaks[2]-breaks[1]
      df <- data.frame(data[,i])
      c <- ggplot(df,aes(data[,i]))+geom_histogram(binwidth=bwidth,fill="white",colour="black")+
        labs(title = paste(names(data)[i],"- Freedman-Diaconis' Method"))+
        theme(plot.title = element_text(size = rel(1.5)))
    
      grid.arrange(a, b, c, ncol=1)
    
    }
  }
  print("Remember to na.omit or impute missing values!")
}



factor_table <- function(data){
  for(i in 1:ncol(data)){
    if(class(data[,i]) == "factor"){
      print(table(data[,i]))
      print("---------------------------")
    }
  }
}

#CHECK THIS GEAM_BAR
hist_func_fact <- function(data, list = TRUE){
  for(i in 1:ncol(data)){
    if(class(data[,i]) == "factor"){
      data[,i] <- as.numeric(data[,i])
      breaks <- pretty(range(data[,i]), n = nclass.Sturges(data[,i]), min.n = 1)
      bwidth <- breaks[2]-breaks[1]
      df <- data.frame(data[,i])
      a <- ggplot(df,aes(data[,i]))+geom_bar(binwidth=bwidth,fill="white",colour="black")+
        labs(title = paste(names(data)[i]))+
        theme(plot.title = element_text(size = rel(1.5)))
      
      grid.arrange(a, ncol=1)
    }
  }
  print("Remember to na.omit or impute missing values!")
}


#To factors
factorize <- function(data){
  for(i in 1:ncol(data)){
    if(class(data[,i]) == "character"){
      data[,i] <- as.factor(data[,i])
    }
  }
}



#Creating data sets for binary Y cross-validation
split_binaryY <- function(dataY, data, prob, is.same.size = TRUE){
  print("Only for binary Y (0,1)")
  won <- subset(data, dataY == 1)
  lost <- subset(data, dataY == 0)
  #Select churners for training (70%) and testing (30%)
  ind <- sample(2, nrow(lost), replace=TRUE, prob=prob)
  trainData.lost <- lost[ind==1,]
  testData.lost <- lost[ind==2,]
  
  #Randomly select won cases for training and testing, same size with correponding lost sets
  if(is.same.size == TRUE){
    ind2 <- sample(1:nrow(won), nrow(lost), replace = TRUE)
    trainData.won <- won[ind2[ind==1],]
    testData.won <- won[ind2[ind==2],]
  }else{
    ind <- sample(2, nrow(won), replace=TRUE, prob=prob)
    trainData.won <- won[ind==1,]
    testData.won <- won[ind==2,]
  }
  trainData <- rbind(trainData.lost, trainData.won)
  testData <- rbind(testData.lost, testData.won)
  
  return(list(training = trainData, testing=testData))
}


#Imputation for numeric variables only
impute_numint <- function(data, type = "medianImpute"){
  preProcValues <- preProcess(data[,sapply(data,is.numeric)], method = c(type))
  # or *bagImpute* / *medianImpute* / knnImpute
  #knnImpute centers and scales all variables
  #bagged trees can also be used to impute. 
  #For each predictor in the data, a bagged tree is created using all of the other predictors in the training set. 
  #When a new sample has a missing predictor value, the bagged model is used to predict the value.
  imp <- data.frame(predict(preProcValues, data))
  
  preProcValues1 <- preProcess(data[,sapply(imp,is.integer)], method = c(type))
  imp1 <- data.frame(predict(preProcValues, data))
  return(imp1)
}

#Imputation
#with(DF, impute(age, mean))

#Imputation for categorical:
#http://stackoverflow.com/questions/17212502/impute-missing-values-with-caret?rq=1

#dummies <- dummyVars(House_Price ~ . -1, data = my_data, na.action = na.pass)
#x <- predict(dummies, my_data)

#preProcValues <- preProcess(x, method = c("medianImpute"))## or *bagImpute* / *medianImpute* / knnImpute
#knnImpute centers and scales all variables
#imp <- data.frame(predict(preProcValues, x))
#table(is.na(imp))

#data <- cbind(my_data$House_Price, imp)#get rid of "Observation" column



factor_boxplot <- function(dataY, dataX, t.test = TRUE, main = "Y by X", ylab = "Y", xlab = "X"){
  boxplot(dataY~dataX,
          xlab=xlab,ylab=ylab,
          main=main,col=c("Orange","cornflowerblue","grey"))
  
  if(t.test == TRUE){
  
    if(length(levels(dataX)) == 2){
      text(1, 13, paste("P(A = B) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[1],2)),col = "red")
      text(2, 13, paste("P(B = A) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[1],2)),col = "red")
    }
  
    if(length(levels(dataX)) == 3){
      text(1, 13, paste("P(A = B) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[1],2)),col = "red")
    
      text(2, 13, paste("P(B = C) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[2],2)),col = "red")
    
      text(3, 13, paste("P(A = C) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[4],2)),col = "red")
    }
  
    if(length(levels(dataX)) == 4){
      text(1, 13, paste("P(A = B) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[1],2)),col = "red")
    
      text(2, 13, paste("P(B = C) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[2,2],2)),col = "red")
    
      text(3, 13, paste("P(C = D) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[3,3],2)),col = "red")
    
      text(4, 13, paste("P(D = A) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[3,1],2)),col = "red")
    }
  
    if(length(levels(dataX)) == 5){
      text(1, 13, paste("P(A = B) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[1],2)),col = "red")
    
      text(2, 13, paste("P(B = C) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[2,2],2)),col = "red")
    
      text(3, 13, paste("P(C = D) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[3,3],2)),col = "red")
    
      text(4, 13, paste("P(D = E) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[3,1],2)),col = "red")
    
      text(5, 13, paste("P(E = A) = ",
                      round(pairwise.t.test(dataY,dataX)$p.value[4,4],2)),col = "red")
    }
  }
  
  return(round(pairwise.t.test(dataY,dataX)$p.value,2))
}


#Correlations
corr_num_both <- function(data, n = 5){
  a <- round(cor(data[,sapply(data,is.numeric)], use="complete.obs", method="kendall"), n) 
  b <- round(cor(data[,sapply(data,is.numeric)], use="complete.obs", method="pearson"), n)
  datalist <- list(a, b)
  
  #See Rcolor pdf
  #Remember to widen your plot screen before exporting
  corrplot(cor(data[,sapply(data,is.numeric)], use="complete.obs", method="kendall"), method = "number", 
           col = colorRampPalette(c("red","lightgray","blue"))(100), tl.col = "black", 
           mar=c(0,0,1,0), title = "Kendall's Tau Correlation", tl.cex=0.8)
  
  corrplot(cor(data[,sapply(data,is.numeric)], use="complete.obs", method="pearson"), method = "number", 
           col = colorRampPalette(c("red","lightgray","blue"))(100), tl.col = "black", 
           mar=c(0,0,1,0), title = "Pearson Correlation", tl.cex=0.8)
  
  print("First is Kendall's Tau and then Pearson's Linear Correlation")
  return(datalist)
}

#PCA
pca_num <- function(data, ...){
  
  numeric <- data[,sapply(data,is.numeric)]
  numeric <- numeric[,...]
  pca <- PCA(data[,names(numeric)])
  a <- round(pca$eig,3)
  
  correlation_matrix <- as.data.frame(round(cor(data[,names(numeric)],pca$ind$coord)^2*100,0))
  b <- correlation_matrix[with(correlation_matrix, order(-correlation_matrix[,1])),]
  
  datalist <- list(a, b)
  return(datalist)
}

pca_num2 <- function(data, cumu = TRUE, ...){
  
  print("Y has to be numeric!")
  
  #scale=T standardizes the variables to the same relative scale, 
  #avoiding some variables to become dominant just because of their large measurement units.
  pca2 <- prcomp(data[,sapply(data,is.numeric)], scale = TRUE, center = TRUE)
  
  pr.var2 <- pca2$sdev^2
  pve2 <- pr.var2/sum(pr.var2)
  
  b <- summary(pca2)
  
  #the summary indicates that four PCs where created: the number 
  #of possible PCs always equals the number of original variables.
  
  #PC1 and PC2 explain respectively ~30% and ~14% of the data's 
  #total variability, summing up to a 44% of the total variability. 
  
  #a "scree plot" allows a graphical assessment of the relative 
  #contribution of the PCs in explaining the variability of the data.
  
  #pca2$roation # the "Rotation" matrix contains the "loadings" of each 
  #of the original variables on the newly created PCs.
  
  
  variances <- data.frame(variances=pca2$sdev**2, pcomp=1:length(pca2$sdev))
  varPlot <- ggplot(variances, aes(pcomp, variances))+ geom_bar(stat="identity", fill="gray") + geom_line() + 
    labs(title = "Scree plot with normalization")+
    theme(plot.title = element_text(size = rel(1.5)))
  plot(varPlot)#Ylab is Eigenvalues! Xlab is Eigenvectors
  
  par(mfrow = c(1,1))
  plot(cumsum(pve2), main = "Normalized", xlab="Principal  Component", 
       ylab="Cumulative  Proportion  of Variance  Explained", ylim=c(0,1),type="b")#Xlab is Eigenvectors
  
  
  PCbiplot <- function(PC, x="PC1", y="PC2", mm) {
    # PC being a prcomp object
    # data <- data.frame(obsnames=row.names(PC$x), PC$x)
    data <- data.frame(obsnames="", PC$x)#data.frame(obsnames=my_data[,12], PC$x)
    plot <- ggplot(data, aes_string(x=x, y=y))# + geom_text(alpha=.4, size=3, aes(label=obsnames))
    #geom_text is redundant here because we have obsnames as blank
    plot <- plot + geom_hline(aes(0), size=.2, yintercept = 0) + geom_vline(aes(0), size=.2, xintercept = 0) + geom_point(alpha = 1,size = 2)
    datapc <- data.frame(varnames=rownames(PC$rotation), PC$rotation)
    mult <- min(
      (max(data[,y]) - min(data[,y])/(max(datapc[,y])-min(datapc[,y]))),
      (max(data[,x]) - min(data[,x])/(max(datapc[,x])-min(datapc[,x])))
    )
    datapc <- transform(datapc,
                        v1 = .7 * mult * (get(x)),
                        v2 = .7 * mult * (get(y))
    )
    plot <- plot + coord_equal() + geom_text(data=datapc, aes(x=v1, y=v2, label=varnames), size = 6.5, vjust=1, color="red")
    plot <- plot + geom_segment(data=datapc, aes(x=0, y=0, xend=v1, yend=v2), arrow=arrow(length=unit(0.2,"cm")), alpha=0.75, color="red")+
      labs(title = mm)+
      theme(plot.title = element_text(size = rel(1.5)))
    plot
  }
  
  numeric <- data[,sapply(data,is.numeric)]
  numeric <- numeric[,...]
  fit.1 <- prcomp(numeric, scale=TRUE, center = TRUE)#Zero mean and unit variance
  
  plot(PCbiplot(fit.1, mm = "PCA with normalization"))
  
  if(cumu == TRUE){
    return(b)
  }
}

metric_mds <- function(data){
  #http://rpubs.com/sinhrks/plot_mds
  #Standardize the data first
  for(i in 1:ncol(data)){
    if(class(data[,i]) == "integer"){
      data[,i] <- as.numeric(data[,i])
    }
  }
  a <- scale(data[,sapply(data,is.numeric)], scale = TRUE, center = TRUE)

  d <- dist(a, method = "euclidean") # euclidean distances between the rows
  fit <- cmdscale(d, eig=TRUE, k=2) # k is the number of dim
  plot <- autoplot(fit, xlab="Coordinate 1", ylab="Coordinate 2", main = "Metric MDS")
  #plot <- plot +theme(plot.title = element_text(size = rel(1.5))) +geom_hline(aes(0), size=.2) + geom_vline(aes(0), size=.2)
  plot
}

decile_func <- function(dataY, data, n, binary = FALSE){
  data$decile <- ntile(dataY, n=n)#n is the number of groups to split into
  data[,"decile"] <- as.factor(data[,"decile"])
  
  if(n == 2){
    data$decile <- ifelse(data[,"decile"] == 1, 0, 1)
  }
  
  return(data)
}


pcp_func <- function(data, Ycol, notXcols){
  
  if(missing(notXcols) == TRUE){
    a <- match(names(data[,sapply(data,is.numeric)]), names(data))
    remove <- c(Ycol)
    b <- a[!a %in% remove]
  
    data[,Ycol] <- as.factor(data[,Ycol])
    pcp <- ggparcoord(data = data, columns = b, groupColumn = Ycol,#groupColumn should be decile for cont.
                      showPoints = TRUE, title = "Parallel Coordinate Plot", scale = "std")
    pcp <- pcp + theme(plot.title = element_text(size = rel(1.5)))
    pcp #Can't separate the classes but I see some outliers.
    
  }else{
    
    a <- match(names(data[,sapply(data,is.numeric)]), names(data))
    remove <- c(notXcols, Ycol)
    b <- a[!a %in% remove]
    
    data[,Ycol] <- as.factor(data[,Ycol])
    pcp <- ggparcoord(data = data, columns = b, groupColumn = Ycol,#groupColumn should be decile for cont.
                      showPoints = TRUE, title = "Parallel Coordinate Plot", scale = "std")
    pcp <- pcp + theme(plot.title = element_text(size = rel(1.5)))
    pcp #Can't separate the classes but I see some outliers.
  }
}



long_to_wide_ts <- function(data, Xname = "VariableX", date = Sys.Date()){
  names(data) <- c('Year', '01', '02', '03', '04', '05', '06', '07', '08',
                   '09', '10', '11', '12')
  data <- data %>% gather(MonthNum, name, -Year) %>%
    unite_('Month', c('Year', 'MonthNum'), sep = '-') %>% arrange(Month)
  data$Month <- str_c(data$Month, '-01')
  
  # limit date range
  data <- data[data$Month <= date, ]
  data$Month <- ymd(data$Month)  #Month to posix
  names(data) <- c("Date", Xname)
  return(data)
}


#Acquiring data from the Bureau of Labor Statistics website
get_BLStats <- function(url, from_year = 1976, to_year = c(format(Sys.Date(), "%Y"))){
  #example unemployment: 'http://data.bls.gov/timeseries/LNS14000000/pdq/SurveyOutputServlet'
  bls.url <- url
  bls.query <- list(request_action = 'get_data',
                    reformat = 'true',
                    from_results_page = 'false',
                    years_option = 'specific_years',
                    delimiter = 'comma',
                    output_type = 'default',
                    output_view = 'data',
                    to_year = to_year,
                    from_year = from_year,
                    output_format = 'text',
                    original_output_type = 'default',
                    annualAveragesRequested = 'false',
                    include_graphs = 'false'
  )
  bls.response <- GET(bls.url, query = bls.query, encode = 'form')
  
  #This time, however, the data is served within the response HTML and will need to be extracted using the XML package. 
  #The extracted data will be passed as a text connection, this time to read.csv, to create the data frame.
  
  bls.html <- htmlTreeParse(readBin(bls.response$content,
                                    'text'), 
                            useInternalNodes = TRUE)
  bls.raw <- xpathApply(bls.html, "//div/pre", xmlValue)
  bls.con <- textConnection(str_replace_all(as.character(bls.raw),
                                            'Â', ''))
  bls.data <- read.csv(bls.con, skip = 11)
  #bls.data <- bls.data[,c(-length(names(bls.data)))]  # get rid of 'X' col
  
  #rm(list = c('bls.query', 'bls.response', 'bls.html', 'bls.raw', 'bls.con'))
  return(bls.data)
}

#google about embed function
do_lag12 <- function(data, variables){
  
  num_vars <- length(variables)
  num_rows <- nrow(data)
  
  
  for(j in 1:num_vars){
    for(i in 1){
      data[[paste0(variables[j], "_lag")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  for(j in 1:num_vars){
    for(i in 2){
      data[[paste0(variables[j], "_lag2")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  for(j in 1:num_vars){
    for(i in 3){
      data[[paste0(variables[j], "_lag3")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 4){
      data[[paste0(variables[j], "_lag4")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  for(j in 1:num_vars){
    for(i in 5){
      data[[paste0(variables[j], "_lag5")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 6){
      data[[paste0(variables[j], "_lag6")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  for(j in 1:num_vars){
    for(i in 7){
      data[[paste0(variables[j], "_lag7")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 8){
      data[[paste0(variables[j], "_lag8")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 9){
      data[[paste0(variables[j], "_lag9")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 10){
      data[[paste0(variables[j], "_lag10")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 11){
      data[[paste0(variables[j], "_lag11")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  for(j in 1:num_vars){
    for(i in 12){
      data[[paste0(variables[j], "_lag12")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  
  return(data)
}

do_lag <- function(data, variables, num_periods = 1){
  
  num_vars <- length(variables)
  num_rows <- nrow(data)
  

  for(j in 1:num_vars){
    for(i in 1){
      data[[paste0(variables[j], "_lag")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
    }
  }
  
  
  if(num_periods == 2){
    for(j in 1:num_vars){
      for(i in 1:num_periods){
        data[[paste0(variables[j], "_lag2")]] <- c(rep(NA, i), head(data[[variables[j]]], num_rows - i))
      }
    }
  }
  
  return(data)
}

MSE <- function(testY, fit){
  error <- testY - fit
  return(sum(error**2) / length(error))
}

rownames_date <- function(data){
  
  for(i in 1:ncol(data)){
    if(names(data)[i] == "Date" || names(data)[i] == "date" || names(data)[i] == "DATE"){
      rownames(data) <- data[,i]
      data[,i] <- NULL
      return(data)
    }
  } 
}

logdiff_func_one <- function(data, log = FALSE, diff = 1){
  data <- na.omit(data)
  if(diff == 1){
    if(log == TRUE){
      for(i in 1){
        data <- data.frame(log(data))
      }
    }
    for(i in 1){
      data <- diff(data)
    }
    data <- append(data, NA)
    data <- data.frame(data)
    return(data)
  }
  
  if(diff == 2){
    if(log == TRUE){
      for(i in 1){
        data <- data.frame(log(data))
      }
    }
    for(i in 1){
      data <- diff(data)
    }
    data <- append(data, NA)
    for(i in 1){
      data <- diff(data)
    }
    data <- append(data, NA)
    data <- data.frame(data)
    return(data)
  }
}

logdiff_func_all <- function(data, log = FALSE, diff = 1){
  data <- na.omit(data)
  
  if(diff == 0){
    if(log == TRUE){
      for(i in 1:length(data)){
        data[, i][1:(nrow(data))] <- log(data[, i])
      }
      return(data)
    }
  }
  
  if(diff == 1){
    if(log == TRUE){
      for(i in 1:length(data)){
        data[, i][1:(nrow(data))] <- log(data[, i])
      }
    }
    for(i in 1:length(data)){
      data[, i][1:(nrow(data)-1)] <- diff(data[, i])
    }
    data[nrow(data),] <- NA
    return(data)
  }
  
  if(diff == 2){
    if(log == TRUE){
      for(i in 1:length(data)){
        data[, i][1:(nrow(data))] <- log(data[, i])
      }
    }
    
    for(i in 1:length(data)){
      data[, i][1:(nrow(data)-1)] <- diff(data[, i])
    }
    data[nrow(data),] <- NA
    
    for(i in 1:length(data)){
      data[, i][1:(nrow(data)-1)] <- diff(data[, i])
    }
    data[nrow(data),] <- NA
    return(data)
  }
}



growth_rate_all <- function(data, n=12, type = "log"){
  
  data <- na.omit(data)
  a <- ncol(data)
  
  i <- 1
  names <- names(data)
  for(i in 1:length(data)){
    data <- data.frame(data, Delt(tail(data[,i], nrow(data)), k = n, type = type))
  }
  data1 <- data[,(a+1):ncol(data)]
  names(data1) <- names    
  return(data1)
}

to_stationary_all <- function(data){
  
  data <- na.omit(data)
  #dat <- rep(NA, 549)

  for(i in 1:ncol(data)){
    n1 <- ndiffs(data[,i], test=c("kpss"), max.d = 2)
    n2 <- ndiffs(data[,i], test=c("adf"), max.d = 2)
    n3 <- ndiffs(data[,i], test=c("pp"), max.d = 2)
    
    k <- round((n1+n2+n3)/3,0)
    
    if(k > 0){
      data[,i] <- logdiff_func_one(data[,i], diff = k)
    }else{
      data[,i] <- data[,i]
    }
  }
  data <- na.omit(data)
  return(data)
}

plot_all_ts <- function(data, list = TRUE, ncol = 1){
  
  for(i in 1:ncol(d)){
    a <- ggplot(d, aes(x=Date, y=d[,i], group = 1)) + geom_line(alpha = 1)+
      labs(title = paste(names(d)[i]), x = 'Year', y = 'Value')
    
    grid.arrange(a, ncol=ncol)
  }
}

#nsdiffs uses seasonal unit root tests to determine the number of seasonal differences required for time series x to be made stationary (possibly with some lag-one differencing as well).
seasonal_stationary <- function(data){
  ns <- nsdiffs(data)
  if(ns > 0){
    xstar <- diff(data,lag=frequency(data),differences=ns)
  }else{
    xstar <- data
  }
  return(xstar)
}


read_files_to_one <- function(type = "csv"){
  if(type == "csv"){
    files <- list.files(pattern="*.csv")
    data <- do.call(data.frame, lapply(files, function(x) read.csv(x, sep = ";", header = T, stringsAsFactors = FALSE)))
    return(data)
  }
  if(type == "txt"){
    files <- list.files(pattern="*.txt")
    data <- do.call(data.frame, lapply(files, function(x) read.csv(x, sep = ";", header = T, stringsAsFactors = FALSE)))
    return(data)
  }
}

#temp <- list.files(pattern="*.csv")
#for(i in 1:length(temp)){
#  assign(temp[i], read.csv(temp[i], sep = ";", skip = 1, header = T))
#}
