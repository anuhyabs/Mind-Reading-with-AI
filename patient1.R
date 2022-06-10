#Importing libraries
library('R.matlab')
library(caTools)
library(e1071)
library(class)
library(tree)
# library(rpart)
# library(rpart.plot)
library(randomForest)

#Loading Data
p1 <- readMat("data-science-P1.mat")

info <- as.data.frame(p1[2])
info <- t(info)
info <- as.data.frame(info)
lab.grp <-as.data.frame(matrix(nrow=0,ncol=1))
lab.wrd <-as.data.frame(matrix(nrow=0,ncol=1))
for (i in 1:360){
  lab.grp <-  rbind(lab.grp,info$cond[[i]])
  lab.wrd <-  rbind(lab.wrd,info$word[[i]])
}

p1.data <- p1$data
voxels <-as.data.frame(matrix(nrow=0,ncol=21764))
for (i in 1:360){
  voxels <-  rbind(voxels,p1.data[[i]][[1]])
}

# Principal Component Analysis for Feature Reduction
pr.out <- prcomp(voxels)
pcs <- as.data.frame(pr.out$x)
pcs$grp <- lab.grp$V1
#pcs$wrd <- lab.wrd$V1

# Splitting data into training and test data
set.seed(100) 
sample <- sample.split(pcs[,1], SplitRatio = .834)
pcs.train <- subset(pcs, sample == TRUE)
pcs.test  <- subset(pcs, sample == FALSE)
pcs.train.x <- subset(pcs.train, select = -c(grp))
pcs.train.labs <- pcs.train$grp
pcs.test.x <- subset(pcs.test, select = -c(grp))
pcs.test.labs <- pcs.test$grp

# Classification Algorithms

# Naive Bayes Classifier

nb.fit <- naiveBayes(grp ~ . , data = pcs.train)
nb.class <- predict(nb.fit,pcs.test.x)
nb.class

confusion_mat.nb = as.matrix(table(Actual_Values = pcs.test.labs, Predicted_Values = nb.class)) 
print(confusion_mat.nb)

print(mean(nb.class != pcs.test$grp))

# KNN

knn.pred <- knn(pcs.train.x, pcs.test.x, pcs.train.labs, k=3)

confusion_mat.knn = as.matrix(table(pcs.test.labs, knn.pred)) 
print(confusion_mat.knn)

print(mean(knn.pred!= pcs.test$grp))

# Decision Trees

tree.fit <- tree(as.factor(grp) ~ ., data = pcs.train)
summary(tree.fit)
plot(tree.fit)
text(tree.fit, pretty = 0)

tree.pred <- predict(tree.fit, newdata = pcs.test)

# Random Forest

rf.fit <- randomForest(as.factor(grp) ~ ., data = pcs.train,, mtry = 80, importance = TRUE)
summary(rf.fit)

# Plotting PCA
Cols <- function(vec){
  cols <- rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}
par(mfrow = c(1,2))
plot(pr.out.train$x[,1:2], col = Cols(voxels.train.labs), pch = 19, xlab = "Z1", ylab = "Z2")
plot(pr.out.train$x[,c(1,3)], col = Cols(voxels.train.labs), pch = 19, xlab = "Z1", ylab = "Z3")
summary(pr.out.train)
plot(pr.out.train)
pve <- 100 * pr.out.train$sdev^2 / sum(pr.out.train$sdev^2)
plot(pve, type = "o", ylab = "PVE", xlab = "Principal Component", col = "blue")
plot(cumsum(pve), type = "o", ylab = "Cumulative PVE", 
     xlab = "Principal Component", col = "brown3")
pr.out.test <- prcomp(voxels.test.x)
