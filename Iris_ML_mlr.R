#------------------------------------------------------------------                                                            #
#   H0   --- Model could not predict the correct class of Species #
#   H1   --- Model can predict the Species class correctly        #
#   Target  -- Species  (classification problem)                  #
#-----------------------------------------------------------------
setwd('G:/DATASCIENCE/DS-PRACTICE-PROJECTS/3_Iris/')
#-----------------------------------------------------------------
library(dplyr)
#library(data.table)
library(ggplot2)
library(gridExtra)
library(corrplot)
#library(GGally)
library(caret)
library(mlr)
#----------------------------------------------------------------
#------------------ 1. Data Loading and Task Preparation -----------
data(iris)
set.seed(3)
train <- sample_frac(iris, 0.8, replace=T)
rid <- as.numeric(rownames(train))
test <- iris[-rid,]

trainTask <- makeClassifTask(data = train, target = 'Species')
testTask <- makeClassifTask(data = test, target = 'Species')

#---------------------------------------------------------------
#------------------ 2. Data Summarization -------------------------
head(train)
str(train)
summary(train)
summarizeColumns(train)
prop.table(table(train$Species)) * 100
#---------------------------------------------------------------
#------------------ 3. Data Visualization -------------------------
# Univariate Visualizations
par(mfrow=c(1,4))
for (i in (1:4)){
  hist(train[,i], main=names(train)[i])
}
for (i in (1:4)){
  plot(density(train[,i]), main=names(train)[i])
}
for (i in (1:4)){
  boxplot(train[,i], main=names(train)[i])
}
for (i in (1:4)) {
  qqnorm(train[,i])
  qqline(train[,i], main=names(train)[i])
}
# Bi/Multi Variate Visualizations
#plot(train[sapply(train,is.numeric)], main = 'Iris_train Data', pch = 21, bg = c('red','yellow', 'blue'))
featurePlot(train[,1:4], train[,5], plot='ellipse')
cor <- cor(train[,1:4], method='pearson')
cor
corrplot(cor, method='circle', type='lower')

SL_b <- ggplot(iris,aes(Species, Sepal.Length,fill=Species)) + geom_boxplot() + labs(title='S.L Vs Species')
SW_b <- ggplot(iris,aes(Species, Sepal.Width,fill=Species)) + geom_boxplot() + labs(title='S.W Vs Species')
PL_b <- ggplot(iris,aes(Species, Petal.Length,fill=Species)) + geom_boxplot() + labs(title='P.L Vs Species')
PW_b <- ggplot(iris,aes(Species, Petal.Width,fill=Species)) + geom_boxplot() + labs(title='P.L Vs Species')

grid.arrange(SL_b,SW_b,PL_b,PW_b, nrow=2)

SL_d <- ggplot(iris,aes(Sepal.Length, ..density.., fill=Species)) + geom_density() + labs(title='S.L by Species')
SW_d <- ggplot(iris,aes(Sepal.Width, ..density.., fill=Species)) + geom_density() + labs(title='S.W by Species')
PL_d <- ggplot(iris,aes(Petal.Length, ..density.., fill=Species)) + geom_density() + labs(title='P.L by Species')
PW_d <- ggplot(iris,aes(Petal.Width, ..density.., fill=Species)) + geom_density() + labs(title='P.L by Species')

grid.arrange(SL_d,SW_d,PL_d,PW_d, nrow=2)
#--------------------------------------------------------------
#------------------ 4. Data Preparation of ML -----------------
#      Here the data can be used as such

#------------------ 5. Resample and ML Model evaluation -------
#listLearners()

lrns <- list(
  makeLearner('classif.lda', id='lda'), 
  makeLearner('classif.rpart',id='rpart'), 
  makeLearner('classif.randomForest', id='rf'),
  makeLearner('classif.ksvm',id='svm'),
  makeLearner('classif.knn', id='knn'),
  makeLearner('classif.naiveBayes', id='nb'),
#  makeLearner('classif.nnnet', id='nb'),
  makeLearner('classif.gbm', id='gbm')
)

set.seed(3)
rdesc <- makeResampleDesc(method='CV', iter=10, stratify=TRUE)
bmr <- benchmark(lrns,trainTask,rdesc,measures = acc)
bmr

sum <- plotBMRSummary(bmr, measure=acc)
bar <- plotBMRRanksAsBarChart(bmr,measure=acc)
box <- plotBMRBoxplots(bmr, measure=acc) + aes(color=learner.id)
viol <- plotBMRBoxplots(bmr, measure=acc, style='violin') + aes(color=learner.id)

grid.arrange(sum,bar,box,viol, nrow=2)

#-------- Finalizing and Summarizing the model --------
set.seed(3)
lda <- resample(learner='classif.lda', task=trainTask, resampling=rdesc, 
               measures=list(acc,mmce))
lda
names(lda)
ggplot(lda$measures.test, aes(iter,acc)) + geom_line()
#---------------------------------------------------
#---------------- 6. Training -------------------------
set.seed(3)
lda_model <- train(learner='classif.lda', task=trainTask)
lda_model
names(lda_model)
lda_model$learner.model
# --------------- 7. Prediction -----------------------
lda_pred <- predict(lda_model, newdata=test)
names(lda_pred)
getConfMatrix(lda_pred)

performance(lda_pred, measures=list(acc,mmce), task=lda_model)

#------------------------- END ---------------------