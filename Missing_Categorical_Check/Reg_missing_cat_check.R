# -------------------------------------------------------------
# This is a demonstration to check whether 'missing values'   |
# and 'categorical variables' are needed to treat or not.     |
# The target is a 'continuous variable'                       |
#                                                             |
#         ** Following are the observations **                |
#                                                             |
# 1. for DT's, RF, and XGboost - hot encoding is not required |
#       for categorical var's. They automatically does it     | 
# 2. RF and XGboost complaints about 'missingness' both in    |
#               train and test data sets and stops            |
# 3. DT's works fine with missingness                         |
# 4. For categorical variables, I created a new category 'UNK'|
#                for missing values                           |
# 5. For continuous vars, DT simply works, but randomForest   |
#   and xgboost fails to run and I tested them by replacing   |
#   the missing values with '-9999' and they run fine         | 
#-------------------------------------------------------------
library(car)
library(mlr)
library(dplyr)
library(tibble)
rm(list=ls())
#-----------------------------------------------------
data(Prestige)
head(Prestige)
colSums(is.na(Prestige))
head(Prestige)
#-----------------------------------------------------
Prestige <- rownames_to_column(Prestige)
set.seed(3)
train <- sample_frac(Prestige,0.7,replace=T) 
rid <- as.numeric(rownames(train))
test <- Prestige[-rid,]
head(test)
train <- train[,-1]
test <- test[,-c(1)]
#-----------------------------------------------------
trainTask <- makeRegrTask(data=train, target='prestige')
dt_lrn <- makeLearner('regr.rpart',predict.type = 'response')
rf_lrn <- makeLearner('regr.randomForest',predict.type = 'response')
xgb_lrn <- makeLearner('regr.xgboost', predict.type='response')
#-----------------------------------------------------
# data as such with "--missing values--"
dt <- train(dt_lrn,trainTask)
dt_pred <- predict(dt,newdata=test)
dt_pred$data

dt_mse <- mean((dt_pred$data$truth - dt_pred$data$response)^2)
dt_rmse <- sqrt(dt_mse) # 9.92
#------------------------------------------------------------

rf <- train(rf_lrn,trainTask)
xgb <- train(xgb_lrn,trainTask)
# randomForest and xgboost -- throws an error saying there are 
#                             missing values
#------------------------------------------------------------
train_unk <- train
test_unk <- test
# Lets replace the missingvalues with '-9999' or 'UNK'
which(!complete.cases(train_unk))
which(!complete.cases(test_unk)) 
# it complained the missingness even for test data while predicting
# hence the missingness has to be fixed for both the test and train 
# data sets
train_unk$type <- as.character(train_unk$type)
test_unk$type <- as.character(test_unk$type)

train_unk$type[4] <- 'UNK'
train_unk$type[11] <- 'UNK'
test_unk$type[26] <- 'UNK'
test_unk$type[29] <- 'UNK'

train_unk$type <- as.factor(train_unk$type)
test_unk$type <- as.factor(test_unk$type)
colSums(is.na(train_unk))
colSums(is.na(test_unk))
#------------------------------------------------------------
trainTask_unk <- makeRegrTask(data=train_unk, target='prestige')

# checking how the result will change with the replacement of 
# missing values in DT's

dt_unk <- train(dt_lrn,trainTask_unk)
dt_pred_unk <- predict(dt_unk,newdata=test_unk)
dt_mse_unk <- mean((dt_pred_unk$data$truth - dt_pred_unk$data$response)^2)
dt_rmse_unk <- sqrt(dt_mse_unk) # 9.92 --- here the result did not changed !!!

rf <- train(rf_lrn,trainTask_unk)
rf_pred <- predict(rf,newdata=test_unk)
rf_pred$data
rf_mse <- mean((rf_pred$data$truth - rf_pred$data$response)^2)
rf_rmse <- sqrt(rf_mse) # 7.56

xgb <- train(xgb_lrn,trainTask_unk)
xgb_pred <- predict(xgb,newdata=test_unk)
xgb_pred$data
xgb_mse <- mean((xgb_pred$data$truth - xgb_pred$data$response)^2)
xgb_rmse <- sqrt(xgb_mse) # 33.62

#------------------------------------------------------------
# In the following I am introducing some random missing values in 
# numerical attributes and lets check them
train_unk2 <- train_unk
test_unk2 <- test_unk

list <- c(10,32,47,53,69)
for (i in list) {
    train_unk2$education[i] <- NA
}

list2 <- c(5, 17, 29)
for (i in list2) {
    test_unk2$education[i] <- NA
}
#--------------------------------------------------------------
# data as such with "--missing values in education--"
trainTask2 <- makeRegrTask(data=train_unk2, target='prestige')
dt2_lrn <- makeLearner('regr.rpart',predict.type = 'response')
rf2_lrn <- makeLearner('regr.randomForest',predict.type = 'response')
xgb2_lrn <- makeLearner('regr.xgboost', predict.type='response')
#--------------------------------------------------------------
dt2 <- train(dt2_lrn,trainTask2)
dt2_pred <- predict(dt2,newdata=test_unk2)
dt2_pred$data

dt2_mse <- mean((dt2_pred$data$truth - dt2_pred$data$response)^2)
dt2_rmse <- sqrt(dt2_mse) # 9.08
#-------------------------------------------------------------
rf2 <- train(rf2_lrn,trainTask2)
xgb2 <- train(xgb2_lrn,trainTask2)
# randomForest and xgboost -- throws an error saying there are 
#                             missing values in education
#-------------------------------------------------------------
# lets replace these with '-9999'
train_unk3 <- train_unk2
test_unk3 <- test_unk2
for (i in list) {
  train_unk3$education[i] <- -9999
}
for (i in list2) {
  test_unk3$education[i] <- -9999
}
#------------------------------------------------------------
# data as such with "--missing values in education--"
trainTask3 <- makeRegrTask(data=train_unk3, target='prestige')
dt3_lrn <- makeLearner('regr.rpart',predict.type = 'response')
rf3_lrn <- makeLearner('regr.randomForest',predict.type = 'response')
xgb3_lrn <- makeLearner('regr.xgboost', predict.type='response')
#--------------------------------------------------------------
dt3 <- train(dt3_lrn,trainTask3)
dt3_pred <- predict(dt3,newdata=test_unk3)
dt3_pred$data

dt3_mse <- mean((dt3_pred$data$truth - dt3_pred$data$response)^2)
dt3_rmse <- sqrt(dt3_mse) # 9.70 -- this changed with replacement
#-------------------------------------------------------------
rf3 <- train(rf3_lrn,trainTask3)
rf3_pred <- predict(rf3,newdata=test_unk3)
rf3_pred$data
rf3_mse <- mean((rf3_pred$data$truth - rf3_pred$data$response)^2)
rf3_rmse <- sqrt(rf3_mse) # 7.72

xgb3 <- train(xgb3_lrn,trainTask3)
xgb3_pred <- predict(xgb3,newdata=test_unk3)
xgb3_pred$data
xgb3_mse <- mean((xgb3_pred$data$truth - xgb3_pred$data$response)^2)
xgb3_rmse <- sqrt(xgb3_mse) # 34.02

#-------------------------------------------------------------



