#-----------------------------------------------------------------
# This is based on Analytics Vidhya's Knocktober-Machine Learning |
#  Competition                                                    |
#-----------------------------------------------------------------
setwd('G:/DATASCIENCE/DS-PROJECTS/Analytics_Vidhya/Knoctober/')
rm(list=ls())
#-----------------------------------------------------------------
library(dplyr)
library(data.table)
library(ggplot2)
library(GGally)
library(plotly)
library(caret)
library(mlr) 
library(gridExtra)
library(lubridate)
#----------------------------------------------------------------
train <- read.csv('Train/Train.csv',na.strings = c('',' ','?',NA),stringsAsFactors = FALSE)
hcd <- read.csv('Train/Health_Camp_Detail.csv', na.strings=c('',' ','?',NA), stringsAsFactors = FALSE)
fhc <- read.csv('Train/First_Health_Camp_Attended.csv', na.strings=c('',' ','?',NA), stringsAsFactors = FALSE)
shc <- read.csv('Train/Second_Health_Camp_Attended.csv', na.strings=c('',' ','?',NA), stringsAsFactors = FALSE)
thc <- read.csv('Train/Third_Health_Camp_Attended.csv', na.strings=c('',' ','?',NA), stringsAsFactors = FALSE)
patient <- read.csv('Train/Patient_Profile.csv', na.strings=c('',' ','?',NA), stringsAsFactors = FALSE)

test <- read.csv('Test.csv',na.strings = c('',' ','?',NA),stringsAsFactors = FALSE)
#----------------------------------------------------------------
train$isTrain <- T
test$isTest <- F

df_all <- bind_rows(train,test)

df_all <- left_join(df_all, fhc, by=c("Patient_ID","Health_Camp_ID"))
df_all <- left_join(df_all, shc, by=c("Patient_ID","Health_Camp_ID"))
df_all <- left_join(df_all, thc, by=c("Patient_ID","Health_Camp_ID"))

df_all <- left_join(df_all, patient, by=c("Patient_ID"))
df_all <- left_join(df_all, hcd, by=c("Health_Camp_ID"))

df_all <- df_all[, c(1,2,24,3,27,28,4:23,25,26,29:31)]

df_all$Outcome <- ifelse(!is.na(df_all$Health_Score) | !is.na(df_all$Health.Score) | (df_all$Number_of_stall_visited > 0), 1, 0)
df_all$Outcome <- ifelse(is.na(df_all$Outcome),0,df_all$Outcome)

summarizeColumns(df_all)

df_all$First_Interaction <- dmy(df_all$First_Interaction)
df_all$Registration_Date <- dmy(df_all$Registration_Date)
df_all$Camp_Start_Date   <- dmy(df_all$Camp_Start_Date)
df_all$Camp_End_Date     <- dmy(df_all$Camp_End_Date)

df_all$Education_Score   <- as.numeric(df_all$Education_Score)
df_all$City_Type         <- as.numeric(as.factor(df_all$City_Type))
df_all$Employer_Category <- as.numeric(as.factor(df_all$Employer_Category))
df_all$Income            <- as.numeric(df_all$Income)
df_all$Age               <- as.numeric(df_all$Age)
df_all$Category1         <- as.numeric(as.factor(df_all$Category1))
df_all$Category2         <- as.numeric(as.factor(df_all$Category2))
df_all$Outcome           <- as.factor(df_all$Outcome)

head(df_all)

df_all$camp_duration <- as.numeric(difftime(df_all$Camp_End_Date, df_all$Camp_Start_Date, units='days'))
df_all$register_End <- as.numeric(difftime(df_all$Camp_End_Date, df_all$Registration_Date, units='days'))
df_all$register_Start <- ifelse(df_all$Registration_Date < df_all$Camp_Start_Date, 1, 0)
df_all$First_int_Register <- as.numeric(difftime(df_all$Registration_Date, df_all$First_Interaction, units='days'))
df_all$First_int_Start <- as.numeric(difftime(df_all$Camp_Start_Date, df_all$First_Interaction, units='days'))

train <- filter(df_all,isTrain %in% c(TRUE))
test <- filter(df_all,isTest %in% c(FALSE))

delete_features <- c('Patient_ID',
                     'Health_Camp_ID',
                     'First_Interaction',
                     'Registration_Date',
                     'Camp_Start_Date',
                     'Camp_End_Date',
                     'isTrain',
                     'isTest',
                     'X',
                     'Number_of_stall_visited',
                     'Last_Stall_Visited_Number',
                     'Online_Follower',
                     'LinkedIn_Shared',
                     'Twitter_Shared',
                     'Facebook_Shared')

new_train <- train
new_test <- test
for (i in delete_features){
  new_train <- select(new_train, -get(i))
  new_test <- select(new_test, -get(i))
}
colnames(new_test)
new_train <- subset(train, select=-c(delete_features))
new_test <- select(test,-delete_features)

new_train[is.na(new_train)] <- -9999
new_test[is.na(new_test)] <- -9999

trainTask <- makeClassifTask(data=new_train, target='Outcome')
testTask <- makeClassifTask(data=new_test,target='Outcome')
xgb.learner <- makeLearner('classif.xgboost', predict.type = 'prob')
xgb.learner$par.vals <- list(objective = "binary:logistic",nrounds = 100)
#------------------------------------------------------------
set.seed(3)
rdesc <- makeResampleDesc(method='CV', iter=3, stratify=TRUE)
xgb <- resample(learner=xgb.learner, task=trainTask, resampling=rdesc, measures=list(acc,auc))

#--------------- Training------------------------------------
set.seed(3)
xgb_model <- train(learner=xgb.learner, task=trainTask)

xgb_model$learner.model
# --------------- 7. Prediction -----------------------
xgb_pred <- predict(xgb_model, task=testTask)
names(xgb_pred)
getConfMatrix(xgb_pred)
#-----------------------------------------------------------------
# XGBoost tuning
#define parameters for tuning
getParamSet('classif.xgboost')
xgb_ps <- makeParamSet(
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("subsample", lower = 0.10, upper = 1.00),
  makeNumericParam("colsample_bytree",lower = 0.1,upper = 1.0),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeIntegerParam("nrounds",lower=200,upper=600)
  #makeIntegerParam('early.stop.round', lower=50,upper=100)
  )
#define search function
rancontrol <- makeTuneControlRandom(maxit = 10L) #do 100 iterations

#tune parameters
xgb_tune <- tuneParams(learner = xgb.learner, task = trainTask, resampling = rdesc,measures =
                        list(acc,auc),par.set = xgb_ps, control = rancontrol)
xgb_tune$x
#set parameters
xgb_model_tuned <- setHyperPars(learner = xgb.learner, par.vals = xgb_tune$x)
#train model
xgb_model_trained <- train(xgb_model_tuned, trainTask)
#test model
xgb_predicted <- predict(xgb_model_trained, testTask)
names(xgb_predicted)
head(xgb_predicted$data$response)
# Feature Importance
top_task <- filterFeatures(trainTask, method= 'rf.importance',abs=6)

# Submission
submit <- data.frame(Patient_ID = test$Patient_ID, Health_Camp_ID = test$Health_Camp_ID,
                     Outcome = xgb_predicted$data$response)
write.csv(submit, "xgboost_health.csv",row.names = F)
#---------------------------------------------------------------




