library(ISLR)
library(MASS)
library(e1071)

#Chapter 10: Unsupervised Learning

#----------------------------Principal Components Analysis

states <- row.names(USArrests)

#The columns of the data set contain the four variables.
names(USArrests)

#We first briefly examine the data. We notice that the variables have vastly
#different means.
apply(USArrests, 2, mean)
#The second input here denotes whether we wish to compute the mean of the rows, 1,
#or the columns, 2. We see that there are on average three times as many
#rapes as murders, and more than eight times as many assaults as rapes.
#We can also examine the variances of the four variables using the apply()function.

apply(USArrests, 2, var)#Not surprisingly, the variables also have vastly different variances

#Thus, it is important to standardize the
#variables to have mean zero and standard deviation one before performing PCA.
#We now perform principal components analysis using the prcomp() function

#scale=T standardizes the variables to the same relative scale, 
#avoiding some variables to become dominant just because of their 
#large measurement units.

pr.out <- prcomp(USArrests, scale=TRUE)

#By default, the prcomp() function centers the variables to have mean zero.
#By using the optionscale=TRUE, we scale the variables to have standard deviation one.

names(pr.out)

#The center and scale components correspond to the means and standard
#deviations of the variables that were used for scaling prior to implementing PCA.

pr.out$center
pr.out$scale

#The rotation matrix provides the principal component loadings; each column of pr.out$rotation
#contains the corresponding principal componentloading vector

pr.out$rotation

#We see that there are four distinct principal components
#This is to be expected because there are in general min(n-1,p) 
#informative principal components in a data set with n observations and p variables.

#Using the prcomp()function, we do not need to explicitly multiply the
#data by the principal component loading vectors in order to obtain the
#principal component score vectors. Rather the 50 × 4 matrix x has as its 
#columns the principal component score vectors. That is, the
#kth column is the kth principal component score vector.

dim(pr.out$x)

#We can plot the first two principal components as follows:
biplot(pr.out, scale=0)
#The scale=0 argument to biplot()
#ensures that the arrows are scaled to
#represent the loadings; other values for scale give slightly different biplots
#with different interpretations.

pr.out$rotation <- -pr.out$rotation
pr.out$x <- -pr.out$x
biplot(pr.out, scale=0)

#standard deviations:
pr.out$sdev

#The variance explained by each principal component is obtained by squaring these:
pr.var <- pr.out$sdev^2
pr.var

#To compute the proportion of variance explained by each principal compo-
#nent, we simply divide the variance explained by each principal component
#by the total variance explained by all four principal components:

pve <- pr.var/sum(pr.var)
pve

#We see that the first principal component explains 62.0% of the variance
#in the data, the next principal component explains 24.7% of the variance,
#and so forth. We can plot the PVE explained by each component, as well
#as the cumulative PVE, as follows:
plot(pve, xlab="Principal  Component", ylab="Proportion  of Variance  Explained", ylim=c(0,1),type="b")
plot(cumsum(pve), xlab="Principal  Component", ylab="Cumulative  Proportion  of Variance  Explained", ylim=c(0,1),type="b")

#----------------------------------------K-Means Clustering

set.seed(2)
x <- matrix(rnorm(50*2), ncol=2)
x[1:25,1] <- x[1:25,1]+3
x[1:25,2] <- x[1:25,2]-4

#We now perform K-means clustering with K=2
km.out <- kmeans(x,2,nstart=20)

#The  cluster  assignments  of  the  50  observations  are  contained  in km.out$cluster
km.out$cluster

plot(x, col=(km.out$cluster+1), main="K-Means ClusteringResults with K=2", xlab="", ylab="", pch=20, cex=2)

#Here the observations can be easily plotted because they are two-dimensional.
#If there were more than two variables then we could instead perform PCA and plot the first two principal components score vectors.

#K=3

set.seed(4)
km.out <- kmeans(x,3,nstart=20)
km.out
plot(x, col=(km.out$cluster+1), main="K-Means ClusteringResults with K=3", xlab="", ylab="", pch=20, cex=2)

#To run the kmeans() function in R with multiple initial cluster assignments, we use the
#nstart argument. If a value of nstart greater than one is used, then
#K-means clustering will be performed using multiple random assignments in Step 1 of Algorithm 10.1, and the kmeans()
#function will report only the best results. Here we compare using nstart=1 to nstart=20.

set.seed(3)
km.out <- kmeans(x,3,nstart=1)
km.out$tot.withinss
km.out <- kmeans(x,3,nstart=20)
km.out$tot.withinss

#Note that km.out$tot.withinss is the total within-cluster sum of squares, 
#which we seek to minimize by performing K-means clustering
#The individual within-cluster sum-of-squares are contained in the vector km.out$withinss.

#We strongly recommend always running K-means clustering with a large value of nstart
#such as 20 or 50, since otherwise an undesirable localoptimum may be obtained.

#--------------------------------------Hierarchical Clustering

#The hclust() function implements hierarchical clustering in R. 
# plot the hierarchical
#clustering dendrogram using complete, single, and average linkage clustering, 
#with Euclidean distance as the dissimilarity measure. We begin by
#clustering observations using complete linkage.

hc.complete <- hclust(dist(x), method="complete")

#We could just as easily perform hierarchical clustering with average or single linkage instead:
hc.average <- hclust(dist(x), method="average")
hc.single <- hclust(dist(x), method="single")

#We can now plot the dendrograms obtained using the usual plot() function.
#The numbers at the bottom of the plot identify each observation.

par(mfrow=c(1,3))
plot(hc.complete,main="Complete  Linkage", xlab="", sub="",cex=.9)
plot(hc.average ,main="Average Linkage", xlab="", sub="",cex=.9)
plot(hc.single,main="Single Linkage", xlab="", sub="",cex=.9)
#To determine the cluster labels for each observation associated with a
#given cut of the dendrogram, we can use the cutree() function:

cutree(hc.complete, 2)
cutree(hc.average , 2)
cutree(hc.single, 2)

#For this data, complete and average linkage generally separate the observa-
#tions into their correct groups. However, single linkage identifies one point
#as belonging to its own cluster. A more sensible answer is obtained when
#four clusters are selected, although there are still two singletons.

cutree(hc.single, 4)

#To scale the variables before performing hierarchical clustering of the
#observations, we use the scale() function:
xsc <- scale(x)
par(mfrow = c(1,1))
plot(hclust(dist(xsc), method="complete"), main="HierarchicalClustering  with Scaled Features")

#Correlation-based distance can be computed using the as.dist() function
#which converts an arbitrary square symmetric matrix into a form that
#the hclust() function recognizes as a distance matrix. However, this only
#makes sense for data with at least three features since the absolute corre-
#lation between any two observations with measurements on two features is
#always 1. Hence, we will cluster a three-dimensional data set.

x <- matrix(rnorm(30*3), ncol=3)
dd <- as.dist(1-cor(t(x)))
plot(hclust(dd, method="complete"), main="Complete  Linkagewith Correlation-Based Distance", xlab="", sub="")


#---------------------------------------NCI60 Data Example: PCA and Hierarchical Clustering

nci.labs <- NCI60$labs
nci.data <- NCI60$data
#Each cell line is labeled with a cancer type. We do not make use of the
#cancer types in performing PCA and clu
#stering, as these are unsupervised
#techniques. But after performing PCA and clustering, we will check to
#see the extent to which these cancer types agree with the results of these
#unsupervised techniques

#We begin by examining the cancer types for the cell lines.
nci.labs[1:4]
table(nci.labs)

#PCA

#We first perform PCA on the data after scaling the variables (genes) to
#have standard deviation one, although one could reasonably argue that it
#is better not to scale the genes.

pr.out <- prcomp(nci.data, scale=TRUE)

#We now plot the first few principal component score vectors, in order to
#visualize the data. The observations (cell lines) corresponding to a given
#cancer type will be plotted in the sa
#me color, so that we can see to what
#extent the observations within a cancer type are similar to each other. We
#first create a simple function that assigns a distinct color to each element
#of a numeric vector. The function will be used to assign a color to each of
#the 64 cell lines, based on the cancer type to which it corresponds.

Cols <- function(vec){
    cols=rainbow(length(unique(vec)))
    return(cols[as.numeric(as.factor(vec))])
}
#Note that therainbow()function takes as its argument a positive integer,
#and returns a vector containing that number of distinct colors. We now can
#plot the principal component score vectors.

par(mfrow=c(1,2))
plot(pr.out$x[,1:2], col=Cols(nci.labs), pch=19,
       xlab="Z1",ylab="Z2")
plot(pr.out$x[,c(1,3)], col=Cols(nci.labs), pch=19,
       xlab="Z1",ylab="Z3")

#On the whole, cell lines corresponding to a single cancer type do tend to have similar values on the
#first few principal component score vectors. This indicates that cell lines
#from the same cancer type tend to have pretty similar gene expression levels.

summary(pr.out)
par(mfrow = c(1,1))
plot(pr.out)

#Note that the height of each bar in the bar plot is given by squaring the
#corresponding element of pr.out$sdev. However, it is more informative to
#plot the PVE of each principal component (i.e. a scree plot) and the cu-
#mulative PVE of each principal componen

pve=100*pr.out$sdev^2/sum(pr.out$sdev^2)
par(mfrow=c(1,2))
plot(pve,  type="o", ylab="PVE", xlab="Principal  Component",
       col="blue")
plot(cumsum(pve), type="o", ylab="Cumulative  PVE", xlab="
Principal  Component", col="brown3")

#(Note that the elements of
#pve can also be computed directly from the summary,
#summary(pr.out)$importance[2,], and the elements of
#cumsum(pve) are given by summary(pr.out)$importance[3,].) 
#The resulting plots are shown in Figure 10.16. We see that together, the first seven principal components
#explain around 40% of the variance in the data. This is not a huge amount
#of the variance. However, looking at the scree plot, we see that while each
#of the first seven principal components explain a substantial amount of
#variance, there is a marked decrease in the variance explained by further
#principal components. That is, there is an elbow
#in the plot after approximately the seventh principal component. This suggests that there may
#be little benefit to examining more than seven or so principal components
#(though even examining seven principal components may be difficult)

#--------------------------------Clustering the Observations of the NCI60 Data

#We now proceed to hierarchically cluster the cell lines in the NCI60data,
#with the goal of finding out whether or not the observations cluster into
#distinct types of cancer. To begin, we standardize the variables to have
#mean zero and standard deviation one. As mentioned earlier, this step is
#optional and should be performed only if we want each gene to be on the same scale.
sd.data=scale(nci.data)
#We now perform hierarchical clustering of the observations using complete,
#single, and average linkage. Euclidean distance is used as the dissimilarity measure.
par(mfrow=c(1,3))
data.dist=dist(sd.data)
plot(hclust(data.dist), labels=nci.labs, main="Complete Linkage", xlab="", sub="",ylab="")
plot(hclust(data.dist, method="average"), labels=nci.labs, main="Average Linkage", xlab="", sub="",ylab="")
plot(hclust(data.dist, method="single"), labels=nci.labs, main="Single Linkage", xlab="", sub="",ylab="")

#We see that the choice of linkage
#certainly does affect the results obtained. Typically, single linkage will tend to yield trailing
#clusters: very large clusters onto which individual observations attach one-by-one. On the other hand, complete and average linkage
#tend to yield more balanced, attractive clusters. For this reason, complete
#and average linkage are generally preferred to single linkage. Clearly cell
#lines within a single cancer type do tend to cluster together, although the
#clustering is not perfect. We will use complete linkage hierarchical clustering for the analysis that follows.
#We can cut the dendrogram at the height that will yield a particular number of clusters, say four:
hc.out=hclust(dist(sd.data))
hc.clusters=cutree(hc.out,4)
table(hc.clusters,nci.labs)
#There are some clear patterns. All the leukemia cell lines fall in cluster 3, 
#while the breast cancer cell lines are spread out over three different clusters.
#We can plot the cut on the dendrogram that produces these four clusters:
par(mfrow=c(1,1))
plot(hc.out, labels=nci.labs)
abline(h=139, col="red")
#horizontal line at height 139 on the dendrogram; this is the height that results in four distinct clusters. 
#It is easy to verify that the resulting clusters are the same as the ones we obtained using
cutree(hc.out,4)

#quick summary:
hc.out

#K-means clustering
set.seed(2)
km.out=kmeans(sd.data, 4, nstart=20)
km.clusters=km.out$cluster
table(km.clusters,hc.clusters)

#We see that the four clusters obtained using hierarchical clustering and
#K-means clustering are somewhat different. Cluster 2 in K-means clustering is identical to cluster 3 in hierarchical clustering. 
#However, the other clusters differ: for instance, cluster 4 in K-means clustering contains a portion of
#the observations assigned to cluster 1 by hierarchical clustering, as well as
#all of the observations assigned to cluster 2 by hierarchical clustering.
#Rather than performing hierarchical clustering on the entire data matrix,
#we can simply perform hierarchical clustering on the first few principal component score vectors, as follows:

hc.out=hclust(dist(pr.out$x[,1:5]))
plot(hc.out, labels=nci.labs,main="Hier. Clust. on First Five Score Vectors")
table(cutree(hc.out,4), nci.labs)

#Not surprisingly, these results are different from the ones that we obtained when we performed 
#hierarchical clustering on the full data set. Sometimes performing clustering on the first few principal component score vectors
#can give better results than performing clustering on the full data. 
#In this situation, we might view the principal component step as one of denoising the data. 
#We could also performK-means clustering on the first few principal component score vectors rather than the full data set.

#---------------------------APPLIED

#In the chapter, we mentioned the use of correlation-based distance and Euclidean distance as dissimilarity measures for hierarchical clustering. 
#It turns out that these two measures are almost equivalent: 
#if each observation has been centered to have mean zero and standard deviation one
#On the USArrests data, show that this proportionality holds.

dsc = scale(USArrests)
a = dist(dsc)^2
b = as.dist(1 - cor(t(dsc)))
summary(b/a)


#Consider the USArrests data. We will now perform hierarchical clustering on the states.

#(a) Using  hierarchical  clustering  with  complete  linkage  and Euclidean distance, cluster the states.
hc.complete = hclust(dist(USArrests), method="complete")
plot(hc.complete)

#(b) Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?
cutree(hc.complete, 3)
table(cutree(hc.complete, 3))

#(c) Hierarchically cluster the states using complete linkage and Euclidean distance, 
#after scaling the variables to have standard deviation one.
dsc = scale(USArrests)
hc.s.complete = hclust(dist(dsc), method="complete")
plot(hc.s.complete)

#(d) What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be
#scaled before the inter-observati on dissimilarities are computed?
#Provide a justification for your answer.
cutree(hc.s.complete, 3)
table(cutree(hc.s.complete, 3))
table(cutree(hc.s.complete, 3), cutree(hc.complete, 3))
#Scaling the variables effects the max height of the dendogram obtained from
#hierarchical clustering. From a cursory glance, it doesn't effect the bushiness
#of the tree obtained. However, it does affect the clusters obtained from cutting the dendogram into 3 clusters. 
#In my opinion, for this data set the data should be standardized because the data measured has different units ($UrbanPop$ compared to other three columns).


#In this problem, you will generate simulated data, and then perform PCA and K-means clustering on the data.
#(a) Generate a simulated data set with 20 observations in each of three classes (i.e. 60 observations total), and 50 variables.
#Hint: There are a number of functions in R that you can use to generate data. One example is the rnorm() function;
#runif() is another option. Be sure to add a mean shift to the observations in each class so that there are three distinct classes.
x = matrix(rnorm(20*3*50, mean=0, sd=0.001), ncol=50)
x[1:20, 2] = 1
x[21:40, 1] = 2
x[21:40, 2] = 2
x[41:60, 1] = 1
#The concept here is to separate the three classes amongst two dimensions.

#Perform PCA on the 60 observations and plot the first two principal component score vectors. Use a different color to indicate
#the observations in each of the three classes. If the three classes
#appear separated in this plot, then continue on to part (c). 
#If not, then return to part (a) and modify the simulation so that there is greater separation between the three classes. 
#Do not continue to part (c) until the three classes show at least some
#separation in the first two principal component score vectors.
pca.out = prcomp(x)
summary(pca.out)
pca.out$x[,1:2]
plot(pca.out$x[,1:2], col=2:4, xlab="Z1", ylab="Z2", pch=19) 

#Perform K -means clustering of the observations with K=3.
#How well do the clusters that you obtained in K-means clustering compare to the true class labels?
#Hint: You can use the table() function in R to compare the true class labels to the class labels obtained by clustering. 
#Be careful how you interpret the results: K-means clustering will arbitrarily
#number the clusters, so you cannot simply check whether the true class labels and clustering labels are the same.
km.out = kmeans(x, 3, nstart=20)
table(km.out$cluster, c(rep(1,20), rep(2,20), rep(3,20)))
#Perfect match.

#Perform K-means clustering with K= 2. Describe your results.
km.out = kmeans(x, 2, nstart=20)
km.out$cluster
#All of one previous class absorbed into a single class.

#Now perform K-means clustering with K= 4, and describe your results.
km.out = kmeans(x, 4, nstart=20)
km.out$cluster
#All of one previous cluster split into two clusters.

    
#Now perform K-means clustering with K=3on the first two principal component score vectors, rather than on the raw data.
#That is, perform K-means clustering on the 60×2 matrix of which the first column is the first principal component score
#vector, and the second column is the second principal component score vector.
#Comment on the results.
km.out = kmeans(pca.out$x[,1:2], 3, nstart=20)
table(km.out$cluster, c(rep(1,20), rep(2,20), rep(3,20)))
#Perfect match, once again.

#Using the scale() function, perform K-means clustering with
#K= 3 on the dataafter scaling each variable to have standarddeviation one. 
#How do these results compare to those obtained in (b)? Explain
km.out = kmeans(scale(x), 3, nstart=20)
km.out$cluster

#Poorer results than (b): the scaling of the observations effects the distance between them.