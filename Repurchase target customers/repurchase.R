library(ggplot2) ##
library(tidyverse) ##
library(caret)
library(dplyr) ##
library(scales)
library(broom)##

library(lattice) #for caret
library(rpart)
library(modelr)#
library(purrr)
#for random forest
library(parallel) #
library(doParallel)#
library(iterators)
library(ROCR)##
library(DMwR)
library(gbm)#
library(mlbench)
library(plotROC)
library(pROC)
library(forecast)
library(readr)


## Importing data
train_set <- read_csv("repurchase_training.csv")

str(train_set)#NOt important

#converting all to factor?

#Age band
train_set$age_band <- as.factor(train_set$age_band)
levels(train_set$age_band)

#Gender
train_set$gender <- as.factor(train_set$gender)
levels(train_set$gender)

#car model
train_set$car_model <- as.factor(train_set$car_model)
train_set$car_model <- factor(train_set$car_model,
                              levels = c("model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8", "model_9", "model_10", "model_11", "model_12", "model_13", "model_14", "model_15", "model_16", "model_17", "model_18"))
levels(train_set$car_model)

#car segment
train_set$car_segment <- as.factor(train_set$car_segment)
train_set$car_segment <- factor(train_set$car_segment,
                                levels = c("Small/Medium", "Large/SUV", "LCV", "Other"))
levels(train_set$car_segment)


summary(train_set)

#Lets look for missing values
which(is.na(train_set), arr.ind=TRUE)

#deleting the rows with NA in car model
train_set <- train_set[-c(26319, 85668), ]

train_set %>%
  filter(gender == "NULL") %>%
  head()

train_set %>%
  filter(age_band == "NULL") %>%
  tail()

#Taking a look at the dataset
nrow(train_set)

summary(train_set)

#installing library modelr
library(modelr)

#Splitting the data in test and train
set.seed(42)
folds <- crossv_kfold(train_set, k = 6)
folds

#Used crossv_kfold divides the dataset into different folds. here we have set k as 6. so data was divided into 6 parts
#

folds$test[[1]]
folds$test[[2]]
folds$test[[3]]
folds$test[[4]]
folds$test[[5]]
folds$test[[6]]

#EDA
round(prop.table(table(train_set$Target)), 3)
#97.3% of the customers chose to not repurchase from the manufacturer



ggplot(train_set, aes(x=car_model, fill = car_segment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90)) +
  xlab("Model of car") +
  ylab("Number of cars solds") +
  ggtitle ("Number of Car Models Sold")

ggplot(train_set, aes(x=age_band, y= ..count.. / sum(..count..), fill = car_segment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90)) +
  xlab("Age Bands") +
  ylab("Percent of sales") +
  ggtitle ("Car_segment Analysis(percent of sales)")

folds.lm <- folds %>% mutate(model = map(train, ~ glm(data = ., Target ~.)))
folds.lm

folds.lm$model[[1]] %>% summary()
folds.lm$model[[2]] %>% summary() #Performs the best with highest AIC score of -104364
folds.lm$model[[3]] %>% summary()
folds.lm$model[[4]] %>% summary()
folds.lm$model[[5]] %>% summary()
folds.lm$model[[6]] %>% summary()

#Removing car_segment, age_of_vehicle_years & non_sched_serv_paid because they dont seem to be statistically significant after seeing the above summary
folds.lm <- folds %>% mutate(model = map(train, ~ glm(Target ~ age_band + gender + car_model + sched_serv_warr + non_sched_serv_warr + sched_serv_paid + total_paid_services + total_services + mth_since_last_serv + annualised_mileage + num_dealers_visited + num_serv_dealer_purchased, 
                                                      data = .)))
folds.lm

folds.lm$model[[1]] %>% summary()
folds.lm$model[[2]] %>% summary() #Performs the best with the highest AIC score of -96037
folds.lm$model[[3]] %>% summary()
folds.lm$model[[4]] %>% summary()
folds.lm$model[[5]] %>% summary()
folds.lm$model[[6]] %>% summary()

#The AIC value of -96,037 is good as we have a data of 131,337 observations.
#Upon running the model by removing all age and genders as a lot of the values in that variable was NULL, the AIC score improved marginally

#Plotting the best LM model with the highest AIC Score
plot(folds.lm$model[[2]])

###############
folds.lm %>% mutate(predicted = map2(model, test, ~ augment(.x, newdata = .y)))

#NOt important to do
folds.lm %>%
  mutate(predicted = map2(model, test, ~ augment(.x, newdata = .y))) %>% 
  unnest(predicted) 

#Nai chal raha hai yeh
predicted.lm <- folds.lm %>% 
  unnest(map2(model, test, ~ augment(.x, newdata = .y)))
predicted.lm %>%
  filter(Target == TRUE)
###############

#Testing RMSE
test.rmse <- map2_dbl(folds.lm$model, folds.lm$test, rmse)
test.rmse # Model 2 has lowest RMSE(root mean square error), Indicates a better fit


summary(train_set$Target)

summary(test.rmse)

#Standard deviation of test.rmse
sd(test.rmse)

as.data.frame(test.rmse) %>%
  ggplot(aes(y=test.rmse)) +
  geom_boxplot()
#as see in the plot, an oulier lies at 0.150

train.rmse <- map2_dbl(folds.lm$model, folds.lm$train, rmse)
train.rmse

#convert test rmse of train and test to vectors to run the test
test.rmse2 <- as.numeric(test.rmse) 
train.rmse2 <- as.numeric(train.rmse)

test.rmse2
train.rmse2


######
#Running wilcox test #not important
wilcox.test(test.rmse2, train.rmse2, paried=T)
#p>0.05
######


#computing the residuals has been skipped


test.predictions <- folds.lm %>%
  unnest((pred = map2(model, test, ~predict( .x, .y, type= "response"))))

test.predictions


test.predictions <- folds.lm %>%
  unnest (fitted = map2(model, test, ~augment (.x, newdata = .y)),
          pred = map2(model, test, ~predict(.x, .y, type = "response")) )

test.predictions %>%
  select(.id, Target, pred) %>%
  filter(Target ==1)

#we are creating a dataframe with fold


#AUC
test.predictions %>%
  group_by(.id) %>%
  summarize(auc = roc(Target, .fitted)$auc) %>%
  select(auc)
#High auc value indicate model fits

#COnfusion matrix
test.predictions %>%
  filter(Target == 1)

test.predictions %>%
  select(.id, Target, pred) %>%
  mutate(pred = ifelse(pred >= 0.070, "Likely", "Unlikely"))

#we need to tally up by folds.
which(is.na(test.predictions), arr.ind=TRUE)

#these NA values are derived from this table, as we can see these people have bought a car however are purely customers that are male and don't aid the predicted results. lets remove these NAs.

test.predictions <- test.predictions[-c(36170, 48181), ]

test.predictions %>%
  select(.id, Target, pred) %>%
  mutate(pred = ifelse(pred >=0.075, "Likely", "Unlikely")) %>%
  group_by(.id, Target, pred) %>%
  tally()

#accuracy
mean(predicted==folds$test$`1`$data$Target)

#confusion matrix (i know this is manual for now but i couldn't figure out how to do it within the test.predictions and spred() created an error everytime)
cfm1 <- matrix(c(18543,2761,175,411), ncol=2, byrow=TRUE)
colnames(cfm1) <- c("Unlikely", "Likely")
rownames(cfm1) <- c("Unlikely", "Likely")
cfm1 <- as.table(cfm1)
cfm1

#precision
precision.lm <- cfm1[1,1]/(cfm1[1,1]+cfm1[1,2])
precision.lm

#Recall = TP/(TP+FN)
recall.lm <- cfm1[1,1]/(cfm1[1,1]+cfm1[2,1])
recall.lm


#F1
f1.lm <- 2*(precision.lm*recall.lm/(precision.lm+recall.lm))
f1.lm



#Rpart
#get index of predicted variable
typeColNum <- grep("Target",names(train_set))

#create training and test sets
## 80% of the sample size, use floor to round down to nearest integer
trainset_size <- floor(0.80 * nrow(train_set))


# first step is to set a random seed to ensurre we get the same result each time
#All random number generators use a seed 

set.seed(53) 

#get indices of observations to be assigned to training set...
#this is via randomly picking observations using the sample function

trainset_indices <- sample(seq_len(nrow(train_set)), size = trainset_size)
trainset_indices

#assign observations to training and testing sets

trainset <- train_set[trainset_indices, ]
testset <- train_set[-trainset_indices, ]

#rowcounts to check
nrow(trainset)
nrow(testset)
nrow(train_set)

#build tree
#default params. This is a classification problem so set method="class"
rpart_model <- rpart(Target~ age_band + gender + car_model + sched_serv_warr + non_sched_serv_warr + sched_serv_paid + total_paid_services + total_services + mth_since_last_serv + annualised_mileage + num_dealers_visited + num_serv_dealer_purchased, data = trainset, method="class")
#plot tree - SAVE PLOT for comparison later
#plot(rpart_model);text(rpart_model)
#prp from rpart.plot produces nicer plots
library(rpart.plot)
rpart.plot(rpart_model)


#summary
summary(rpart_model)


#predict on test data
rpart_predict <- predict(rpart_model,testset[-typeColNum],type="class")

#accuracy
mean(rpart_predict==testset$Target)
#0.9832


#Something about confusion matrix
testset$Target <- as.factor(testset$Target)
rpart.cfm <- confusionMatrix(rpart_predict, testset$Target)
rpart.cfm

#Precision = TP/(TP+FP)
rpart.precision <- rpart.cfm$table[1,1]/(rpart.cfm$table[1,1]+rpart.cfm$table[1,2])
rpart.precision

#Recall = TP/(TP+FN)
rpart.recall <- rpart.cfm$table[1,1]/(rpart.cfm$table[1,1]+rpart.cfm$table[2,1])
rpart.recall

#F1
rpart.f1 <- 2*(rpart.precision*rpart.recall/(rpart.precision+rpart.recall))
rpart.f1



multiple_runs_rpart <- function(df,class_variable_name,train_fraction,nruns){
  
  #Purpose:
  #Builds rpart model for nrun data partitions
  
  #Return value:
  #Vector containing nrun accuracies
  
  #Arguments:
  #df: variable containing dataframe
  #class_variable_name: class name as a quoted string. e.g. "Class"
  #train_fraction: fraction of data to be assigned to training set (0<train_fraction<1)
  #nruns: number of data partitions
  
  #find column index of class variable
  typeColNum <- grep(class_variable_name,names(df))
  #initialize accuracy vector
  accuracies <- rep(NA,nruns)
  #set seed (can be any integer)
  set.seed(1)
  for (i in 1:nruns){
    #partition data
    trainset_size <- floor(train_fraction * nrow(df))
    trainset_indices <- sample(seq_len(nrow(df)), size = trainset_size)
    trainset <- df[trainset_indices, ]
    testset <- df[-trainset_indices, ]
    #build model 
    #paste builds formula string and as.formula interprets it as an R formula
    rpart_model <- rpart(as.formula(paste(class_variable_name,"~.")),data = trainset, method="class")
    #predict on test data
    rpart_predict <- predict(rpart_model,testset[,-typeColNum],type="class")
    #accuracy
    accuracies[i] <- mean(rpart_predict==testset[[class_variable_name]])
  }
  return(accuracies)
}

#calculate average accuracy and std dev over 30 random partitions
accuracy_results <- multiple_runs_rpart(train_set,"Target",0.8,30)
mean(accuracy_results) #0.9999962


sd(accuracy_results)

accuracy_results


#GBM
#as our target is unbalanced. GBM builds trees, where new tree corrects errors made by previous tree
#Its very good at detecting anamoly, especially when data is unbalanced
#REMOVE: GBM tends to over fit if data has lot of noise. it takes longer to train, difficult to tune

set.seed(42)
train_binary = createDataPartition(y = train_set$Target, p = 0.8, list = F)
training_binary = train_set[train_binary, -1]
testing_binary = train_set[-train_binary, -1]

#simple table

tbl_train <- table(training_binary$Target)
tbl_train
#0=102267, 1=2801

tbl_test <- table(testing_binary$Target)
tbl_test
#0=25547, 1=720

#checking the percentages of data that is 0 & 1
tbl_train_prop <- prop.table(tbl_train)
tbl_train_prop
#0=0.97334107, 1=0.02665893

tbl_test_prop <- prop.table(tbl_test)
tbl_test_prop
#0=0.97258918 , 1= 0.02741082

#Evidently, both test and train split had same % of 0 & 1

#comparing it to the set overall
vehicle_prop <- prop.table(table(testing_binary$Target))
vehicle_prop
#0=0.97258, 1=0.02741

#Training the Model
#pull out the dependent variable
trainX <-training_binary[,-1]        # Pull out the dependent variable
testX <- testing_binary[,-1]
sapply(trainX,summary)

gbm_grid =  expand.grid(
  interaction.depth = c(3,4,5),
  n.trees = c(350,450), 
  shrinkage = c(0.01, 0.001),
  n.minobsinnode = c(10,15)
)

control_binary <- trainControl(method = "repeatedcv", #10 fold cross validation
                               number = 5, #do 5 repetitions of CV
                               summaryFunction = twoClassSummary, #Use AUC to pick the best model
                               classProbs = TRUE,
                               allowParallel = TRUE,
                               savePredictions = TRUE)

training_binary$Target <- as.factor(training_binary$Target)
levels(training_binary$Target) <- c("Unlikely", "Likely")
levels(training_binary$Target)

testing_binary$Target <- as.factor(testing_binary$Target)
levels(testing_binary$Target) <- c("Unlikely", "Likely")
levels(testing_binary$Target)

"should put the expanded grid here"

# Setting up parallel processing   
registerDoParallel(2)       # Register a parallel backend for train
getDoParWorkers()


start <- proc.time()

#fitting our gbm object
gbm_fit = train(x = trainX, y = training_binary$Target, 
                method = "gbm", 
                trControl = control_binary, 
                verbose = FALSE,
                metric = "ROC",
                tuneGrid = gbm_grid)

# Look at the tuning results
# Note that ROC was the performance criterion used to select the optimal model.   

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time #

print(gbm_fit)

gbm_fit$bestTune

plot(gbm_fit)

res <- gbm_fit$results
res

varImp(gbm_fit)

### GBM Model Predictions and Performance
# Make predictions using the test data set
gbm_pred <- predict(gbm_fit, testX)
gbm_pred

#age band and gender has high relative influence
#confusion matrix  
gbm.cfm <- confusionMatrix(gbm_pred, testing_binary$Target)
gbm.cfm

#Precision = TP/(TP+FP)
gbm.precision <- gbm.cfm$table[1,1]/(gbm.cfm$table[1,1]+gbm.cfm$table[1,2])
gbm.precision
#0.9865878

#Recall = TP/(TP+FN)
gbm.recall <- gbm.cfm$table[1,1]/(gbm.cfm$table[1,1]+gbm.cfm$table[2,1])
gbm.recall
#0.9991388

#F1
gbm.f1 <- 2*(gbm.precision*gbm.recall/(gbm.precision+gbm.recall))
gbm.f1
#0.9928237

#ROC CURVE
gbm.probs <- predict(gbm_fit,testX, type ="prob")
head(gbm.probs)

gbm.ROC <- roc(predictor=gbm.probs$Likely,
               response=testing_binary$Target,
               levels=rev(levels(testing_binary$Target)))
gbm.ROC$auc

#Area under the curve: 0.9818
#A high AUC score shows that models prediction are correct

plot(gbm.ROC,main="GBM ROC")



validate_set <- read_csv("repurchase_validation.csv")
head(validate_set)
str(validate_set)

#Converting to factor
#Converting Age Band
validate_set$age_band <- as.factor(validate_set$age_band)
levels(validate_set$age_band)

#Converting Gender
validate_set$gender <- as.factor(validate_set$gender)
levels(validate_set$gender)

#Converting Car_model
validate_set$car_model <- as.factor(validate_set$car_model)
validate_set$car_model <- factor(validate_set$car_model,
                                 levels = c("model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8", "model_9", "model_10", "model_11", "model_12", "model_13", "model_14", "model_15", "model_16", "model_17", "model_18"))
levels(validate_set$car_model)

#Converting Car_Segment
validate_set$car_segment <- as.factor(validate_set$car_segment)
validate_set$car_segment <- factor(validate_set$car_segment,
                                   levels = c("Small/Medium", "Large/SUV", "LCV", "Other"))
levels(validate_set$car_segment)

summary(validate_set)
#Gender has 26375 NUlls, age_band has 42734 Nulls

val.prob <- predict(gbm_fit, validate_set, type="prob")
val.prob[,2]


val.pred <- predict(gbm_fit,validate_set)
val.pred <- as.numeric(factor(val.pred))
val.pred


validate_set$target_class <- val.pred
validate_set$target_probability <- val.prob[,2]

export <- validate_set %>%
  select(ID, target_probability, target_class)

export <- export %>%
  mutate(target_class = replace(target_class, target_class == 1, 0))
export <- export %>%
  mutate(target_class = replace(target_class, target_class == 2, 1))

write.csv(export, file="repurchase_validation_14217754.csv", row.names=FALSE)

