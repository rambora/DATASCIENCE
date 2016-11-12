#-----------------------------------------------------------------
# Problem : Build a predictive model to determine the income      |
# level for people in US.                                         |
#                   Data Source:                                  |
# http://archive.ics.uci.edu/ml/machine-learning-databases/       |
#  census-income-mld/census-income.names                          |
#-----------------------------------------------------------------|
# This is a nice example for imbalanced data. However, due to the |
# memory limitations on my machine, I could not use the complete  |
# data set, but used a reduced data set and the balancing method  |
# employed is'smote'
#-----------------------------------------------------------------
setwd('D:/DATASCIENCE/DS-PROJECTS/1_Imbalanced/')
rm(list=ls())
#-----------------------------------------------------------------
library(dplyr)
library(data.table)
library(ggplot2)
library(gridExtra)
#library(plotly)
library(caret)
library(mlr)               
#-----------------------------------------------------------------
train <- fread("train.csv",na.strings = c(""," ","?",'NA',NA))
test  <- fread("test.csv",na.strings = c(""," ","?",'NA', NA))

train <- sample_frac(train,0.4,replace=FALSE)
test <- sample_frac(test,0.4,replace=FALSE)

train <- as.data.frame(train)
test <- as.data.frame(test)

# since the test data also constains the income_level (target)
# I will basically build the model and check its accuracy

dim(train) #79809 obs. of  41 variables
dim(test)  #39905 obs. of  41 variables
str(train)
# Even in the reduced data set (I only sampled the observations),
# total number of featrues are - 41. Cleraly this require some
# feature engineering / preprocessing

# For the sake of convinience, later, I will separating the numerical and categorical
# variablse from the data sets
numeric <- names(which(sapply(train,is.numeric)))
factor <- names(which(!sapply(train,is.numeric)))

fea <- c('industry_code','occupation_code','business_or_self_employed',
         'veterans_benefits','year')
factor <- c(factor,fea)
numeric <- setdiff(numeric,fea)

# OVerall, we have 34 categoric and 7 numeric variables

# Here, I am converting all the integer features also numeric as I intended to 
# apply xgboost, which only accepts numeric variables


train[,factor] <- lapply(train[,factor], function(x) as.factor(as.character(x)))
test[,factor] <- lapply(test[,factor], function(x) as.factor(as.character(x)))
train[,numeric] <- lapply(train[,numeric], function(x) as.numeric(as.integer(x)))
test[,numeric] <- lapply(test[,numeric], function(x) as.numeric(as.integer(x)))

str(train)      
str(test)       

# target : Income_level

table(is.na(train))  
table(is.na(test))   
# Data sets contains large number of missing values
#-----------------------------------------------------------------
# Let's work on 'target'

unique(train$income_level)
unique(test$income_level)


train$income_level <- ifelse(train$income_level == '-50000', 0,1) 
test$income_level <- ifelse(test$income_level == '-50000', 0,1) 
train$income_level <- as.factor(train$income_level)
test$income_level <- as.factor(test$income_level)

table(train$income_level)
prop.table(table(train$income_level))   # clearly indicates that its an inbalanced data
prop.table(table(test$income_level))
#-----------------------------------------------------------------

train_numeric <- train[,numeric]
train_factor <- train[,factor]
test_numeric <- test[,numeric]
test_factor <- test[,factor]

str(train_numeric)
str(train_factor)
str(test_numeric)
str(test_factor)
rm(train,test) 

summary(train_numeric)
summary(test_numeric)
summary(train_factor)
summary(test_factor)
# numeric variables - wage_per_hour, capital_gains, capital_losses, and 
# dividend_from_Stocks appear to have extreme values (though except in wage_per_hour
# case, in others these may be feasible)

# --- I might cap these large values in the later stage ---

#----------------------------------------------------------------
# Explore Numerical Variables
tr <- function(x) {
  ggplot(train_numeric, aes(x)) + geom_histogram(fill='blue',color='red',alpha=0.7,bins=20)
}

tr_age       <- tr(train_numeric$age) 
tr_wph       <- tr(train_numeric$wage_per_hour)
tr_cap_gains <- tr(train_numeric$capital_gains)
tr_cap_loses <- tr(train_numeric$capital_losses)
tr_div_sto   <- tr(train_numeric$dividend_from_Stocks)
tr_num_per   <- tr(train_numeric$num_person_Worked_employer)
tr_weeks     <- tr(train_numeric$weeks_worked_in_year)


grid.arrange(tr_age,tr_wph,tr_cap_gains,tr_cap_loses,nrow=2, ncol=2)
grid.arrange(tr_div_sto,tr_num_per,tr_weeks,nrow=2, ncol=2)

# Age is somewhat right skewed, clearly wage_per_hour, cpaital_gails,
# capitsl_loses are not well distributed -- might require some transofrmation
#------------------------------------------------------------------
tr2 <- function(x) {
  ggplot(test_numeric, aes(x)) + geom_histogram(fill='red',color='blue',alpha=0.7,bins=20)
}

ts_age       <- tr2(test_numeric$age) 
ts_wph       <- tr2(test_numeric$wage_per_hour)
ts_cap_gains <- tr2(test_numeric$capital_gains)
ts_cap_loses <- tr2(test_numeric$capital_losses)
ts_div_sto   <- tr2(test_numeric$dividend_from_Stocks)
ts_num_per   <- tr2(test_numeric$num_person_Worked_employer)
ts_weeks     <- tr2(test_numeric$weeks_worked_in_year)


grid.arrange(ts_age,ts_wph,ts_cap_gains,ts_cap_loses,nrow=2, ncol=2)
grid.arrange(ts_div_sto,ts_num_per,ts_weeks,nrow=2, ncol=2)

#-------------------------------------------------------------------------------------
#------------------- Data Cleaning --------------------------------------------------
table(is.na(train_numeric))
table(is.na(test_numeric))
# no missing values for numeric features

# Check the correlation in train_numeric
train_corr <- cor(train_numeric)
train_corr <-findCorrelation(train_corr, cutoff = 0.7)
train_numeric <- train_numeric[,-train_corr] # weeks_worked_in_year is removed
test_numeric$weeks_worked_in_year <- NULL
str(train_factor)
#-------------------------------------------------------------------------------------
table(is.na(train_factor)) # has large number of missing values
table(is.na(test_factor))  # has large number of missing values
sapply(train_factor, function(x) (sum(is.na(x))*100)/length(x))
sapply(test_factor, function(x) (sum(is.na(x))*100)/length(x))

# for features migration_msa, migration_reg, migration_within_reg, migration_sunbelt
# almost 50% data is missing !!! Its not good to impute or delete these observations
# interestingly, the data for these attributes is missing largely both in train 
# as well as in test data sets => probably it was hard to collect the data
# lets subset the train and test data sets from these features

delete <- c('migration_msa','migration_reg','migration_within_reg','migration_sunbelt')

for (i in delete){
  train_factor <- select(train_factor, -get(i))
  test_factor <- select(test_factor, -get(i))
}

summary(train_factor)
colSums(is.na(train_factor))
summary(test_factor)
colSums(is.na(test_factor))

# in the remaining features with NA's, classify them as a different category -----------------
#  --------------- Lots of cleaning required -- perhaps define a function and clean it---

mis <- c('hispanic_origin','state_of_previous_residence','country_father','country_mother','country_self')

for (c1 in mis) {
  train_factor[,c1] <- as.character(train_factor[,c1])
}

for (c2 in mis){
  c3 <- which(!complete.cases(train_factor[,c2]))
  for (j1 in c3){
    train_factor[j1,c2] <- 'UNA'
  }
}

for (c4 in mis) {
  train_factor[,c4] <- as.factor(train_factor[,c4])
}
#----------------------------------------------------------------------------------
for (c11 in mis) {
  test_factor[,c11] <- as.character(test_factor[,c11])
}

for (c12 in mis){
  c13 <- which(!complete.cases(test_factor[,c12]))
  for (j11 in c13){
    test_factor[j11,c12] <- 'UNA'
  }
}

for (c14 in mis) {
  test_factor[,c14] <- as.factor(test_factor[,c14])
}
#------------------------------------------------------------------------------------
summary(train_factor)
colSums(is.na(train_factor))
colSums(is.na(test_factor))

# most of the records for features 'wage_per_hour','capital_gains','capital_lossses','dividend_from_Stocks'
# are Zero's. So, lets convert them to categorical variables

dim(train_numeric)

train_numeric %>% select(wage_per_hour) %>% 
  filter(wage_per_hour==0) %>% summarise(n=n())/length(train_numeric$wage_per_hour)  # 94% are '0's
train_numeric %>% select(capital_gains) %>% 
  filter(capital_gains==0) %>% summarise(n=n())/length(train_numeric$capital_gains)  # 96% are '0's
train_numeric %>% select(capital_losses) %>% 
  filter(capital_losses==0) %>% summarise(n=n())/length(train_numeric$capital_losses)  # 98% are '0's
train_numeric %>% select(dividend_from_Stocks) %>% 
  filter(dividend_from_Stocks==0) %>% summarise(n=n())/length(train_numeric$dividend_from_Stocks)  # 89% are '0's

# Hence, I converted these as factors

train_numeric$wage_per_hour <- as.factor(ifelse(train_numeric$wage_per_hour==0,'zero','>Zero'))
train_numeric$capital_gains <- as.factor(ifelse(train_numeric$capital_gains==0,'zero','>Zero'))
train_numeric$capital_losses <- as.factor(ifelse(train_numeric$capital_losses==0,'zero','>Zero'))
train_numeric$dividend_from_Stocks <- as.factor(ifelse(train_numeric$dividend_from_Stocks==0,'zero','>Zero'))

test_numeric$wage_per_hour <- as.factor(ifelse(test_numeric$wage_per_hour==0,'zero','>Zero'))
test_numeric$capital_gains <- as.factor(ifelse(test_numeric$capital_gains==0,'zero','>Zero'))
test_numeric$capital_losses <- as.factor(ifelse(test_numeric$capital_losses==0,'zero','>Zero'))
test_numeric$dividend_from_Stocks <- as.factor(ifelse(test_numeric$dividend_from_Stocks==0,'zero','>Zero'))


train_numeric$age <- cut(train_numeric$age, br=seq(0,90,30),include.lowest=TRUE,labels=c('Young','adult','old'))
test_numeric$age <- cut(test_numeric$age, br=seq(0,90,30),include.lowest=TRUE,labels=c('Young','adult','old'))
#------------------------------------------------------------------------------------
# ---------------- Combining the data -----------------------
train <- cbind(train_numeric,train_factor)
test <- cbind(test_numeric, test_factor)
dim(train)
dim(test)
str(train)
rm(train_numeric,train_factor,test_numeric,test_factor)
#------------------------------------------------------------
#-------------- Task creation -------------------------------
trainTask <- makeClassifTask(data=train, target='income_level')
testTask <- makeClassifTask(data=test,target='income_level')

xgb_lrn <- makeLearner('classif.xgboost', predict.type='response')
xgb_lrn$par.vals <- list(objective = "binary:logistic",nrounds = 100)

# remove zero variance features
trainTask <- removeConstantFeatures(trainTask)
tetsTask <- removeConstantFeatures(testTask)

# get variable importance
var_imp <- generateFilterValuesData(trainTask, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)
#-----------------------------------------------------------
#------------- Handling the unbalnced data -----------------
# can try 'Undersampling','oversampling' but here I am using 'smote'
train.smote <- smote(trainTask,rate = 2,nn = 3) # 74814  9990 
# due to the memory restrictions, I restricted to these parameters
table(getTaskTargets(train.smote))

#xgboost
set.seed(3)
getParamSet('classif.xgboost')
xgb_ps <- makeParamSet( 
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("subsample", lower = 0.10, upper = 1.00),
  makeNumericParam("colsample_bytree",lower = 0.1,upper = 1.0),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeIntegerParam("nrounds",lower=200,upper=600)
)
ranctrl <- makeTuneControlRandom(maxit = 5L) 
rdesc <- makeResampleDesc("CV",iters = 5L,stratify = TRUE)
xgb_tune <- tuneParams(learner = xgb_lrn, task = train.smote, resampling = rdesc, measures = list(acc,tpr,tnr,fpr,fp,fn), 
                       par.set = xgb_ps, control = ranctrl)

# Tune Results :
# [Tune-y] 5: acc.test.mean=0.952,tpr.test.mean=0.985,tnr.test.mean=0.705,
# fpr.test.mean=0.295,fp.test.mean= 589,fn.test.mean= 218; time: 4.8 min; 
# memory: 221Mb use, 1439Mb max
# [Tune] Result: eta=0.144; max_depth=10; min_child_weight=3.93; subsample=0.878; 
# colsample_bytree=0.345; lambda=0.593; nrounds=267 : acc.test.mean=0.952,
# tpr.test.mean=0.985,tnr.test.mean=0.705,fpr.test.mean=0.295,fp.test.mean= 589,
# fn.test.mean= 218

xgb_tuned <- setHyperPars(xgb_lrn, par.vals = xgb_tune$x)

#train model
xgb_model <- train(xgb_tuned, train.smote)

#Prediction
predict.xgb <- predict(xgb_model, testTask)

#Prediction results
xgb_predicted <- predict.xgb$data$response

#make confusion matrix
xgb_confused <- confusionMatrix(test$income_level,xgb_predicted)
# Accuracy : 0.951
# Sensitivity : 0.9595
# Specificity : 0.6898

precision <- xgb_confused$byClass['Pos Pred Value']
recall <- xgb_confused$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure
#0.9744 

#top 20 features
filtered.data <- filterFeatures(train.smote,method = "information.gain",abs = 20)
#------------------------------------------------------------------------------

