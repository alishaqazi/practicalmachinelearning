# Practical Machine Learning Course Project
Alisha Qazi  
12/17/2017  

Practical Machine Learning Course Project
=========================================

## Predicting the Manner in which Exercise was Done

## Analysis

In the code below, we create a model that involves splitting the data into test and cross validation data. This is done because it will help us to ensure that the variables of interest are put into the final model. The expected out of sample error is 0.0056. Early on, I removed a few strings (NA, #DIV/0!, etc.) because these are error codes that will hinder the results. 

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'zone/tz/2017c.
## 1.0/zoneinfo/America/Chicago'
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
mytrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
mytest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url = mytrain, destfile = "data_train.csv")
download.file(url = mytest, destfile = "data_test.csv")
train1 <- read.csv(file = "data_train.csv", na.strings = c("NA","#DIV/0!", ""))
test1 <- read.csv(file = "data_test.csv", na.strings = c("NA","#DIV/0!", ""))
for(i in c(8:ncol(train1)-1)) {
    train1[,i] = as.numeric(as.character(train1[,i]))
    test1[,i] = as.numeric(as.character(test1[,i]))
}
index <- colnames(train1)
index <- colnames(train1[colSums(is.na(train1)) == 0])
index <- index[-c(1:7)]
set.seed(2017)
trainIndex <- createDataPartition(y=train1$classe, p=0.6, list=FALSE)
trainData <- train1[trainIndex, index]
crossvalidation <- train1[-trainIndex, index]
dim(trainData)
```

```
## [1] 11776    53
```

```r
dim(crossvalidation)
```

```
## [1] 7846   53
```

```r
rfmodel <- train(classe ~ ., data = trainData, method = "rf", tr_control = trainControl(method="cv", number = 4, allowParallel = TRUE, verboseIter = TRUE))
rfpredict <- predict(rfmodel, crossvalidation)
matrixmodel <- confusionMatrix(rfpredict,crossvalidation$classe)
matrixmodel
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    8    0    0    0
##          B    3 1502   13    1    0
##          C    2    8 1350   13    5
##          D    0    0    5 1272    4
##          E    1    0    0    0 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9897, 0.9938)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9898          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9895   0.9868   0.9891   0.9938
## Specificity            0.9986   0.9973   0.9957   0.9986   0.9998
## Pos Pred Value         0.9964   0.9888   0.9797   0.9930   0.9993
## Neg Pred Value         0.9989   0.9975   0.9972   0.9979   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1914   0.1721   0.1621   0.1826
## Detection Prevalence   0.2847   0.1936   0.1756   0.1633   0.1828
## Balanced Accuracy      0.9979   0.9934   0.9913   0.9939   0.9968
```

The model's accuracy is 0.9944, and since the sample size is 20, this accuracy rate is sufficient for us to assume the answers will be correct.

## Model - applying to test set

```r
lastcolumn <- length(colnames(test1[]))
colnames(test1)[lastcolumn] <- "classe"
rfquiz <- predict(rfmodel, test1[,index])
rfquiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

