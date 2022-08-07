#installing packages
library(readr)
library(dplyr)
library(tidyverse)
library(caTools)
library(caret)

#Reading the csv file from the local folder
data <- read_csv("transactions.csv")

#Taking a look at the input data set
data

#taking a look at the only the first 6 data rows 
head(data)

#checking class of the date column
class (data$date)

#changing class of date column from charachter to date
data$date <- as.Date(data$date, format = "%d/%m/%Y")

#checking class of the date column again
class (data$date) #Class of Date Column is Date now


#checking for missing values
is.na(data)
which(is.na(data))

#checking for percentage of missing values in each column
colMeans(is.na(data))*100
#There are No missing values in the dataset

#plotting the Total Transaction By Industry
TTBI <- ggplot(data=data, aes(x=industry, y=monthly_amount, fill=industry)) +
  geom_bar(stat="identity") +
  labs (title = "Total Transactions By Industry",
        x = "Industries", y = "Total Amount") +
  scale_x_continuous(breaks=1:10)

TTBI #Industry 2 & 1 have the highest total transactions followed by 6 & 7respectively. Industry 7 & 8 are industries with the lowest amount of total transactions

#Plotting Total Transaction By Location
TTBL <- ggplot(data=data, aes(x=location, y=monthly_amount, fill=location)) +
  geom_bar(stat="identity") +
  labs (title = "Total Transactions By Location",
        x = "Locations", y = "Total Amount") +
  scale_x_continuous(breaks=1:10)

TTBL #Location 1 & 2 has the highest total transactions. Whereas Location 6, 9 and 10 are location swith lowest amount of total transactions


#Creating an aggregated data set using the fields date, industry and location, with a mean of monthly_amount from the input data
aggdata <- data %>%
  group_by(date, industry, location) %>%
  summarize(average = mean(monthly_amount))

#Taking a look at the aggregated dataset and the summary of that data
aggdata
summary(aggdata)

#saving the aggregated data set into the working directory
write.csv(aggdata,"aggdata.csv", row.names = FALSE)

#Filtering Industry 1 & location 1 from the aggregated dataset
ind1loc1 <- filter(aggdata, industry==1 & location==1)
ind1loc1  

#Plotting a plot line for industry 1 & location 1 aggregated data set(ind1loc1)
ind1loc1 %>% ggplot(aes(x = date, y = average, colour = "Red")) +
  geom_smooth(stat="identity") + 
  labs(x = "Date",
       y = "Average Monthly amount",
       title = "Monthly Transaction amount for Industry 1 & Location 1")


#linear regression uses the least square method
#linear regression is a statistical analysis that shows relationship
#between two variables. in our case between monthly amount and date
#Linear regression performs exceptionally well for linearly separable data


#in order to account for seasonality of the data we are creating below month and year variable
#Adding Year and month columns to Ind1Loc1 data by turning dataframe
newdf <- function(df) {
  
  output = df %>%
    group_by(date, industry, location) %>%
    summarize(average = mean(average))
  
  output = output %>%
    mutate(month = format(as.Date(date), "%m")) %>%
    mutate(year = format(as.Date(date), "%Y"))
  
  output$month = as.integer(output$month)
  output$year = as.integer(output$year)
  
  transform(output, month = as.integer(month), 
            year = as.integer(year))
  
  return(output)
  
}

ind1loc1 <- newdf(ind1loc1)
ind1loc1

#Adding Unique ID's to each row in order to label the rows
ind1loc1$id <- 1:nrow(ind1loc1)
ind1loc1 <- as.data.frame(ind1loc1)
ind1loc1

#Using slice function to create a training model and testing model
#as there are 47 rows of data for Industry 1 location 1. we allot 38 rows for training and 9 for testing
#Note: Earlier i had split data differently, But the difference between predicted and actual was a lot
#I select 38 rows for training and 9 for testing is because the difference was minimum for this split
ind1loc1.train <- slice(ind1loc1, 1:38)
ind1loc1.test <- slice(ind1loc1, 39:47)


#Creating a linear model on the training set by using independent variables like date, id and month
#We have included month in order to account for seasonality

ind1loc1.lm = lm(formula = average ~ date + id + month, 
            data = ind1loc1.train)

#Summary of the model
summary(ind1loc1.lm) #Median Residual is 2334, Which is negligible. Hence, Model is a good fit


#plotting the linear regression model
ind1loc1lmplot <- ggplot(data=ind1loc1.train, aes(x=date, y=average)) +
  geom_smooth(method = "lm", colour = "Red", se = FALSE) +
  geom_point()

ind1loc1lmplot

#Linear model plots
plot(ind1loc1.lm)


#prediction for 9 of the testing rows assigned
prediction <- predict(ind1loc1.lm, ind1loc1.test)
prediction

#actual values of the 9 testing rows
actual <- ind1loc1.test$average
actual

#difference between predicted and actual
difference <- (prediction-ind1loc1.test$average)
difference

#Calculating percentage difference
Percentagediff <- ((prediction - ind1loc1.test$average)/ind1loc1.test$average)*100
Percentagediff

#The difference between actual values and predicted values is not much, It ranges from -14% to 6%
#Model is a good fit

#predicting the Average monthly transaction for dec 2016
#first we create a data frame called dec2016 to save the outcome of dec 2016 in
dec2016 <- data.frame(date = "01/12/16",
                         industry=1,
                         location=1,
                         average=0,
                         month=12,
                         year=2016,
                         id=48)


#formatting the columns to appropriate class types
dec2016$date <- as.Date(dec2016$date,
                             format = "%d/%m/%y")
dec2016$industry <- as.integer(dec2016$industry)
dec2016$location <- as.integer(dec2016$location)
dec2016$id <- as.integer(dec2016$id)

#now we apply the linear model and predict december 2016
dec2016$average <- predict(ind1loc1.lm,dec2016)
#Viewing the prediction for december 2016
dec2016

#Binding the two data frames to create a plot i.e Ind1loc1 and december 2016 prediction dataframes
ind1loc1collab <-  rbind(ind1loc1, dec2016)
#viewing the binded dataframe
ind1loc1collab

#Creating a plot for the binded data frame with the prediction
ind1loc1collabplot <- ggplot(data=ind1loc1collab, aes(x=date, y=average)) +
  geom_smooth(stat="identity", method = "lm") +
  labs (title = "Prediction for Dececmber 2016 (Industry 1 - Location 1)",
        x = "Date", y = "Average Monthly Amount")

ind1loc1collabplot #Our model has accounted for seasonality. It appeared to us earlier transaction reduces in the month of december

#Applying LM of IND1LOC1 to whole dataset
aggdata <- newdf(aggdata)
aggdata

#in our previous dataframe we created a extra columns for row id year and month
calculate_predictions <- function(df, industries, locations) {
  output = data.frame()
  testnumber = 9 #we will take 9 rows for testing as we had for our ind1loc1 model
  for (ind in industries) {
    for (loc in locations) {
      
      #Creating a subset for the data
      tempdata = df[df$industry == ind & df$location == loc, ]
      
      #Checking just to make sure that i have enough number of training rows than testing rows
      if (length(unique(tempdata$date)) >= testnumber) {
        
        #Here we arrange the data by date
        arrange(tempdata, date)
        
        #Adding an ID to rows
        tempdata$time_number = c(1:nrow(tempdata))
        
        #We previously took testing number as 9. Here we take total rows and minus that from testing rows and set it as training rows
        trainingnumber = nrow(tempdata) - testnumber
        
        #Arranging the training set.
        trainingset = head(arrange(tempdata, time_number), trainingnumber)
        
        #Arranging the testing set
        testingset = tail(arrange(tempdata, time_number), testnumber)
        
        # Running the LM model
        training.model = lm(average~time_number, data=trainingset)
        
        #Calculating MSE
        training.mse <- mean(residuals(training.model)^2)
        
        #Calculating RMSE
        training.rmse <- sqrt(training.mse)
        
        # now we Add an extra row into tempdata for the prediction
        
        # Create a dataframe to store dec2016 data
        december_2016 = data.frame(date = "2016-12-01",
                                   industry=ind,
                                   location=loc,
                                   average=0,
                                   month=12,
                                   year=2016,
                                   time_number=(nrow(tempdata)+1))
        
        #Ensuring that tempdata is of type data frame
        tempdata = as.data.frame(tempdata)
        
        #Binding the predicted data to tempdata
        tempdata = rbind(tempdata, december_2016)
        
        #adding prediction to the tempdata data frame
        tempdata$prediction = predict(training.model, tempdata)
        testingset$prediction = predict(training.model, testingset)
        
        #Getting the prediction for Dec 2016 value
        train_dec_2016_prediction = tail(tempdata$prediction, 1)
        
        # Creating a new row to add to the data frame
        dataRow = c(ind,loc,training.rmse,train_dec_2016_prediction)
        
      } else {
        # changing the entry to output frame when there is not enough data to compute the model
        dataRow = c(ind,loc,NA,NA)
      }
      
      # Add the row to the output dataframe
      output = rbind(output, dataRow)
    }
  }
  
  #Add column names to the output data frame
  colnames(output) <- c("Industry","Location", "RMSE", "Dec 2016 Prediction")
  
  #Return the output
  return(output)
}

#Sorting industries in order 
industries <- sort(unique(aggdata$industry))

#Sorting locations in order
locations <- sort(unique(aggdata$location))

#Calculating the prediction for all industries and location
allpredictions <- calculate_predictions(aggdata, industries, locations)

#Ordering all predictions by RMSE to check for which the model performs the worst
arrange(allpredictions, RMSE)

#After Using the Linear model we created for industry 1 and location 1 we got very poor results
#not feasible can only predict accurately less than 5% of data
#maybe the model might improve if we apply to Just Industries and remove the locations or vice versa
#best thing to do would be to individually create linear models for each industry and location manually although that would be time consuming
#that would ensure us the best fitting model
#or allott more rows to train