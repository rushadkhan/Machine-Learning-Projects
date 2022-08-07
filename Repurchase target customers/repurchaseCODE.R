#Installing packages
library(ggplot2)
library(tidyverse)
library(dplyr)
library(modelr) #to create test and train splits(fold)
library(gbm)
library(broom)
library(rpart)
library(rpart.plot)
library(caret)
library(parallel)
library(doParallel)
library(ROCR)
library(plotROC)
library(pROC)


## Importing the data, reading the csv file
trainingdata <- read_csv("repurchase_training.csv")

#Looking at the structure of the dataset
str(trainingdata)

#converting the following varibales from charachter to factors/categorical variable

#Converting Gender
trainingdata$gender <- as.factor(trainingdata$gender)
levels(trainingdata$gender)

#Converting Age band
trainingdata$age_band <- as.factor(trainingdata$age_band)
levels(trainingdata$age_band)

#Converting car model and setting levels manually
trainingdata$car_model <- as.factor(trainingdata$car_model)
trainingdata$car_model <- factor(trainingdata$car_model,
                                 levels = c("model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8", "model_9", "model_10", "model_11", "model_12", "model_13", "model_14", "model_15", "model_16", "model_17", "model_18"))
levels(trainingdata$car_model)

#Converting car segment and setting levels manually
trainingdata$car_segment <- as.factor(trainingdata$car_segment)
trainingdata$car_segment <- factor(trainingdata$car_segment,
                                   levels = c("Small/Medium", "Large/SUV", "LCV", "Other"))
levels(trainingdata$car_segment)

#Taking a closer look at the dataset and examining for NULL values
summary(trainingdata)

#using which(is.na) function to look for NA values
which(is.na(trainingdata), arr.ind=TRUE)

#deleting the rows with NA in car model & using filter() on Gender and age
trainingdata <- trainingdata[-c(26319, 85668), ]

trainingdata %>%
  filter(gender == "NULL") %>%
  head()

trainingdata %>%
  filter(age_band == "NULL") %>%
  tail()



#Generating K-fold test-training pairs
set.seed(42)
fold <- crossv_kfold(trainingdata, k = 6)
fold

#Used crossv_kfold divides the dataset into different folds. here we have set k as 6. so data was divided into 6 parts

#Taking a look the folds
fold$test[[1]]
fold$test[[2]]
fold$test[[3]]
fold$test[[4]]
fold$test[[5]]
fold$test[[6]]

#EDA
#evaluating how many people chose to repurchase
round(prop.table(table(trainingdata$Target)), 3)
#97.3% of the customers chose to not repurchase from the manufacturer
#and only 2.7% of the customers chose to purchase the data


#Visualising the most sold car model
ggplot(trainingdata, aes(x=car_model, fill = car_segment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90)) +
  xlab("Model of car") +
  ylab("Number of cars solds") +
  ggtitle ("Number of Car Models Sold")


#GLM
#Running the glm model on the created fold. Id can be removed as it serves us no purpose
#dataset being changed using mutate()
foldmodel <- fold %>% mutate(model = map(train, ~ glm(data = ., Target ~.)))
foldmodel

#Taking a look at the summary of all the 6 models and noting their AIC score
foldmodel$model[[1]] %>% summary()
foldmodel$model[[2]] %>% summary() #Has high AIC score -104363
foldmodel$model[[3]] %>% summary()
foldmodel$model[[4]] %>% summary()
foldmodel$model[[5]] %>% summary()
foldmodel$model[[6]] %>% summary()



#Removing car_segment, age_of_vehicle_years & non_sched_serv_paid because they dont seem to be statistically significant after seeing the above summary
foldmodel <- fold %>% mutate(model = map(train, ~ glm(Target ~ age_band + gender + car_model + sched_serv_warr + non_sched_serv_warr + sched_serv_paid + total_paid_services + total_services + mth_since_last_serv + annualised_mileage + num_dealers_visited + num_serv_dealer_purchased, 
                                                      data = .)))
foldmodel

#Taking a look at the summary of all the 6 models and noting their AIC score again
foldmodel$model[[1]] %>% summary()
foldmodel$model[[2]] %>% summary() #AIC = -96037
foldmodel$model[[3]] %>% summary()
foldmodel$model[[4]] %>% summary() 
foldmodel$model[[5]] %>% summary()
foldmodel$model[[6]] %>% summary()

#The AIC value of -96,037 is good as we have a data of 131,337 observations.
#Upon running the model by removing all age and genders as a lot of the values in that variable was NULL, the AIC score improved marginally

#Plotting the best LM model with the highest AIC Score
plot(foldmodel$model[[2]])


foldmodel %>% mutate(predicted = map2(model, test, ~ augment(.x, newdata = .y)))

foldmodel %>%
  mutate(predicted = map2(model, test, ~ augment(.x, newdata = .y))) %>% 
  unnest(predicted) 


#Testing the RMSE value of all 6 models
rmse_test <- map2_dbl(foldmodel$model, foldmodel$test, rmse)
rmse_test # Model 2 has lowest RMSE(root mean square error), Indicates a better fit
summary(rmse_test)

#summarising the target variable from the trainingdata
summary(trainingdata$Target)


#Checking boxplot
as.data.frame(rmse_test) %>%
  ggplot(aes(x="", y=rmse_test)) +
  geom_boxplot()
#as seen in the plot, There is an oulier lies at 0.150



rmse_train <- map2_dbl(foldmodel$model, foldmodel$train, rmse)
rmse_train

#convert test rmse of train and test to vectors to run the test
rmse_test2 <- as.numeric(rmse_test) 
rmse_train2 <- as.numeric(rmse_train)

rmse_test2
rmse_train2


#we are now creating a dataframe with only 3 variables that is fold(ID), target and prediction
predtest <- foldmodel %>%
  unnest (fitted = map2(model, test, ~augment (.x, newdata = .y)),
          pred = map2(model, test, ~predict(.x, .y, type = "response")) )

predtest %>%
  select(.id, Target, pred) %>%
  filter(Target ==1)


#AUC (Area under the curve)
predtest %>%
  group_by(.id) %>%
  summarize(auc = roc(Target, .fitted)$auc) %>%
  select(auc)
#High Auc score of 0.901 indicates good fit

#Summarising the predictions in all the 6 folds
predtest %>%
  filter(Target == 1)

predtest %>%
  select(.id, Target, pred) %>%
  mutate(pred = ifelse(pred >= 0.070, "Likely", "Unlikely"))

#we need to tally up by fold.
which(is.na(predtest), arr.ind=TRUE)

#Removin the NA values as they dont help us

predtest <- predtest[-c(36170, 48181), ]

predtest %>%
  select(.id, Target, pred) %>%
  mutate(pred = ifelse(pred >=0.075, "Likely", "Unlikely")) %>%
  group_by(.id, Target, pred) %>%
  tally()


#manually making a confusion matrix
#A confusion matrix in R is a table that will categorize the predictions against the actual values.
confusionmatrix1 <- matrix(c(18543,2761,175,411), ncol=2, byrow=TRUE)
colnames(confusionmatrix1) <- c("Unlikely", "Likely")
rownames(confusionmatrix1) <- c("Unlikely", "Likely")
confusionmatrix1 <- as.table(confusionmatrix1)
confusionmatrix1

#Calculating precision - True positive/ (Truepositive + False positive)
precision_lm <- confusionmatrix1[1,1]/(confusionmatrix1[1,1]+confusionmatrix1[1,2])
precision_lm
#0.8703999

#Recall - True positive/ (Truepositive + False Negative)
recall_lm <- confusionmatrix1[1,1]/(confusionmatrix1[1,1]+confusionmatrix1[2,1])
recall_lm
#0.9906507
#An ideal system with high precision and high recall will return many results, with all results labeled correctly.


#F1 - It combines precison and recall into asingle by taking mean
f1_lm <- 2*(precision_lm*recall_lm/(precision_lm+recall_lm))
f1_lm
#0.9266403
#Indicates good precision and recall


#Rpart
#getting the index of the predicted variable
typeColNum <- grep("Target",names(trainingdata))

#creating the training and test sets
## taking 80% of the sample size, using the floor() to round to closest integer
trainingdatasize <- floor(0.80 * nrow(trainingdata))


# Setting random seed
set.seed(53) 

#randomly picking the observations to get indices

trainingdataindices <- sample(seq_len(nrow(trainingdata)), size = trainingdatasize)
trainingdataindices

#assigning the observations to trainingset and testingset

trainingset <- trainingdata[trainingdataindices, ]
testingset <- trainingdata[-trainingdataindices, ]

#counting rows of trainingset & testingset and the main dataset
nrow(trainingset)
nrow(testingset)
nrow(trainingdata)

#building trees as this is a classification assignment, we set methoad as class
rpartmodel <- rpart(Target~ age_band + gender + car_model + sched_serv_warr + non_sched_serv_warr + sched_serv_paid + total_paid_services + total_services + mth_since_last_serv + annualised_mileage + num_dealers_visited + num_serv_dealer_purchased, data = trainingset, method="class")


rpart.plot(rpartmodel)


#summary of rpart model
summary(rpartmodel)


#predicting thetestingset
rpartpredict <- predict(rpartmodel,testingset[-typeColNum],type="class")

#checking the predictions accuracy
mean(rpartpredict==testingset$Target)
#0.9832


#Creating a Confusion matrix between rpartpredict, testingset$Target
testingset$Target <- as.factor(testingset$Target)
rpartconfusionmatrix <- confusionMatrix(rpartpredict, testingset$Target)
rpartconfusionmatrix

#Calculating Precision
rpartprecision <- rpartconfusionmatrix$table[1,1]/(rpartconfusionmatrix$table[1,1]+rpartconfusionmatrix$table[1,2])
rpartprecision
#0.985156

#Calculating Recall
rpartrecall <- rpartconfusionmatrix$table[1,1]/(rpartconfusionmatrix$table[1,1]+rpartconfusionmatrix$table[2,1])
rpartrecall
#0.9978465

#Calculating F1
rpartf1 <- 2*(rpartprecision*rpartrecall/(rpartprecision+rpartrecall))
rpartf1
#0.991460


#Building a rpartmodel for number of data partition, it will give us a vector data with accuracy


multiple.rpart.runs <- function(df,class_variable_name,train_fraction,nruns){
  
  
  
  
  typeColNum <- grep(class_variable_name,names(df))
  #initialize the accuracy vector
  accuracies <- rep(NA,nruns)
  set.seed(1)
  for (i in 1:nruns){
    #partition of the data
    trainingdatasize <- floor(train_fraction * nrow(df))
    trainingdataindices <- sample(seq_len(nrow(df)), size = trainingdatasize)
    trainingset <- df[trainingdataindices, ]
    testingset <- df[-trainingdataindices, ]
    #building model 
    #paste builds formula string and as.formula interprets it as an R formula
    rpartmodel <- rpart(as.formula(paste(class_variable_name,"~.")),data = trainingset, method="class")
    #predict on test data
    rpartpredict <- predict(rpartmodel,testingset[,-typeColNum],type="class")
    #accuracy
    accuracies[i] <- mean(rpartpredict==testingset[[class_variable_name]])
  }
  return(accuracies)
}

#calculating avg accuracy of 30 random partitions
accuracy.results <- multiple.rpart.runs(trainingdata,"Target",0.8,30)
mean(accuracy.results) 
#0.9999962

#Calculating standard deviation
sd(accuracy.results)
accuracy.results
#A low sd means data is around the mean 


#GBM
#as our target is unbalanced. GBM builds trees, where new tree corrects errors made by previous tree
#Its very good at detecting anamoly, especially when data is unbalanced

#Setting seed so that every time our set is not some random number
set.seed(42)

#Splitting the target data into two sets for training and testing
binarytrain = createDataPartition(y = trainingdata$Target, p = 0.8, list = F)
binarytraining = trainingdata[binarytrain, -1]
binarytesting = trainingdata[-binarytrain, -1]

#creating a table illustrating all the frequency of $target
table_training <- table(binarytraining$Target)
table_training
#0=102267, 1=2801

table_testing <- table(binarytesting$Target)
table_testing
#0=25547, 1=720

#checking the percentages of data that is 0 & 1
table_training_percentage <- prop.table(table_training)
table_training_percentage
#0=0.97334107, 1=0.02665893

table_testing_percentage <- prop.table(table_testing)
table_testing_percentage
#0=0.97258918 , 1= 0.02741082

#Evidently, both test and train split had approximately same amount of 0 & 1

#comparing it to the set overall
vehicle_percentage <- prop.table(table(binarytesting$Target))
vehicle_percentage
#0=0.97258, 1=0.02741

#Training the Model
trainingthemodel <-binarytraining[,-1]        # Pulling out the dependent variable
testingthemodel <- binarytesting[,-1]
sapply(trainingthemodel,summary)

#setting up the grid for the gbm model
gbm_grid =  expand.grid(
  shrinkage = c(0.01, 0.001),
  n.minobsinnode = c(10,15),
  interaction.depth = c(3,4,5),
  n.trees = c(350,450)
)

binarycontrol <- trainControl(method = "repeatedcv", #10 fold of cross validation
                              number = 5,
                              summaryFunction = twoClassSummary, #Using the AUC to pick the best model
                              classProbs = TRUE,
                              allowParallel = TRUE,
                              savePredictions = TRUE)

binarytraining$Target <- as.factor(binarytraining$Target)
levels(binarytraining$Target) <- c("Unlikely", "Likely")
levels(binarytraining$Target)

binarytesting$Target <- as.factor(binarytesting$Target)
levels(binarytesting$Target) <- c("Unlikely", "Likely")
levels(binarytesting$Target)


#grid

# Setting up parallel processing to compare speed with and without parellel computing   
registerDoParallel(2)       # Register a parallel backend for train
getDoParWorkers()


start <- proc.time()

#fitting our gbm object
gbmfit = train(x = trainingthemodel, y = binarytraining$Target, 
               method = "gbm", 
               trControl = binarycontrol, 
               verbose = FALSE,
               metric = "ROC",
               tuneGrid = gbm_grid)

# Look at the tuning results
# Note that ROC was the performance criterion used to select the optimal model.   

end <- proc.time() - start
end_time <- as.numeric((paste(end[3])))
end_time #
#Note: proc.time() determines how much real and CPU time (in seconds) the currently running R process has already taken.


print(gbmfit)

gbmfit$bestTune

plot(gbmfit)

#Checking residulas
res <- gbmfit$results
res

#Checking the variable importance
varImp(gbmfit)

### GBM Model Predictions and Performance
# Make predictions using the test data set
gbmpred <- predict(gbmfit, testingthemodel)
gbmpred

#Note: age band and gender has high relative influence
#confusion matrix  
gbmconfusionmatrix <- confusionMatrix(gbmpred, binarytesting$Target)
gbmconfusionmatrix

#Precision
gbmprecision <- gbmconfusionmatrix$table[1,1]/(gbmconfusionmatrix$table[1,1]+gbmconfusionmatrix$table[1,2])
gbmprecision
#0.9865

#Recall
gbmrecall <- gbmconfusionmatrix$table[1,1]/(gbmconfusionmatrix$table[1,1]+gbmconfusionmatrix$table[2,1])
gbmrecall
#0.999

#F1
gbmf1 <- 2*(gbmprecision*gbmrecall/(gbmprecision+gbmrecall))
gbmf1
#0.9928

#ROC CURVE
gbm.probs <- predict(gbmfit,testingthemodel, type ="prob")
head(gbm.probs)

gbmroc <- roc(predictor=gbm.probs$Likely,
              response=binarytesting$Target,
              levels=rev(levels(binarytesting$Target)))
gbmroc$auc

#Area under the curve: 0.98
#A high AUC score shows that models prediction are correct

plot(gbmroc,main="GBM ROC")


#Importing the validation dataset
validationdata <- read_csv("repurchase_validation.csv")
head(validationdata)
str(validationdata)

#Converting to factor/categorical variable
#Converting Age Band
validationdata$age_band <- as.factor(validationdata$age_band)
levels(validationdata$age_band)

#Converting Gender
validationdata$gender <- as.factor(validationdata$gender)
levels(validationdata$gender)

#Converting Car_model
validationdata$car_model <- as.factor(validationdata$car_model)
validationdata$car_model <- factor(validationdata$car_model,
                                   levels = c("model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8", "model_9", "model_10", "model_11", "model_12", "model_13", "model_14", "model_15", "model_16", "model_17", "model_18"))
levels(validationdata$car_model)

#Converting Car_Segment
validationdata$car_segment <- as.factor(validationdata$car_segment)
validationdata$car_segment <- factor(validationdata$car_segment,
                                     levels = c("Small/Medium", "Large/SUV", "LCV", "Other"))
levels(validationdata$car_segment)

summary(validationdata)
#Gender has 26375 NUlls, age_band has 42734 Nulls

#predicting the validation data
validationprob <- predict(gbmfit, validationdata, type="prob")
validationprob[,2]


validationpred <- predict(gbmfit,validationdata)
validationpred <- as.numeric(factor(validationpred))
validationpred


validationdata$target_class <- validationpred
validationdata$target_probability <- validationprob[,2]


#Exporting the dataset
export <- validationdata %>%
  select(ID, target_probability, target_class)

#changing data from 1 to 0 and from 2 to 1
export <- export %>%
  mutate(target_class = replace(target_class, target_class == 1, 0))
export <- export %>%
  mutate(target_class = replace(target_class, target_class == 2, 1))

#exporting csv
write.csv(export, file="repurchase_validation_14217754.csv", row.names=FALSE)

