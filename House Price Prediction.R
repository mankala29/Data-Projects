library(tidyr)
library(tidyverse)
library(dplyr)
library(lubridate)
library(finalfit)
library(plyr)
library(e1071)
library(caret)
library(hydroGOF)

######################### loading the Data
kc_data = read.csv("C:/Users/parin/Northeastern University/kc_house_data.csv",stringsAsFactors = FALSE)


######## to clean the bathrooms column
find <- c(1.25,1.75,2.25,2.75)
replace <- c(1,1.5,2,2.5)
found <- match(kc_data$bathrooms,find)
ifelse(is.na(found), kc_data$bathrooms , replace[found])

######################removing insignificant column
kc_data$id <- NULL

###### Correcting the Date Format
kc_data$date <- gsub('.{7}$' , '', kc_data$date)
kc_data$date <- ymd(kc_data$date)


##### to check for NA values
colSums(is.na(kc_data))


##### to check categorical and continuous variables 
ff_glimpse(kc_data)

#####################################################################################

########## Data Spliiting

shuffle_index <- sample(1:nrow(kc_data))
head(kc_data)

kc_data <- kc_data[shuffle_index,]
head(kc_data)

set.seed(123)
index = sample(1:nrow(kc_data))%>% 
  createDataPartition(p = 0.7, list = FALSE)            #####Data partition

train.data  <- kc_data[index, ]
test.data <- kc_data[-index, ]

dim(train.data)
dim(test.data)

###########################################################################
##### Initial Model #######################################################

model_kc <- lm(log(price) ~ . , data = kc_data)   ### linear model on dataset
summary(model_kc)
pred.model <- predict(model_kc)

actual_pred <- data.frame(cbind(actuals=log(kc_data$price), predicteds=pred.model))
cor.model<- cor(actual_pred)   
cor.model

########### original model
data.frame(
  RMSE = RMSE(pred.model, log(kc_data$price)),  
  MSE = mse(pred.model, log(kc_data$price)),
  R2 = R2(pred.model, log(kc_data$price))
)

#RMSE        MSE        R2
#1 0.2513569 0.06318027 0.7722279

#################################################################################
################### This Week -  Model No. 1######################################
###################################################################################

library(MASS)

model_train <-lm(log(price) ~ . , data = train.data)
step_model <- stepAIC(model_train, direction = "both", trace = FALSE)
summary(step_model)
new_pred<-predict(step_model,newdata = test.data)

data.frame(
  RMSE = RMSE(new_pred, log(test.data$price)),
  MSE = mse(new_pred, log(test.data$price)),
  R2 = R2(new_pred, log(test.data$price))
)
####################################
##### getting the same RMSE and R2 and MSE values as the initial models


##### Graph No. 1
plot(fitted(model_train), fitted(step_model), col = "dodgerblue", pch = 20,    
     xlab = "Original Model ", ylab = "Model after Stepwise", cex = 1.5)
abline(a = 0, b = 1, col = "darkorange", lwd = 2)


#################################################################################
############### this week  - Model no. 2 ######################################3
############################################

library(car)

car::vif(step_model)

model2 <- lm(log(price) ~ . - sqft_above - sqft_living, data = train.data) #  remove all the variables with vif >5 
summary(model2)

pred_new <- model2 %>% predict(test.data)

data.frame(
  RMSE = RMSE(pred_new, log(test.data$price)),
  MSE = mse(pred_new, log(test.data$price)),
  R2 = R2(pred_new, log(test.data$price))
)

#RMSE        MSE       R2
#1 0.2563208 0.06570035 0.763432

#### again getting the same RMSE and R2 & MSE values... ###############################


####################################################################################
################# Model No. 3 ############################################
############ Extreme Gradient Boosting   ###############

library(xgboost)

set.seed(123)
kc_model <- train(
  log(price) ~. , data = train.data, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)

kc_model

######## Graph No. 2
p<- varImp(kc_model)
ggplot(p, aes(x=varaibles, y=performance)) +
  geom_point(size = 4) +ggplot2::labs(title = "Variable Importance Plot", subtitle = "Gradient Boosting") 


# Make predictions on the test data
pred.1 <- kc_model %>% predict(test.data)
head(pred.1)

# Compute model prediction accuracy rate
mean(pred.1 == log(test.data$price))
#[1] 0

data.frame(
  RMSE = RMSE(pred.1, log(test.data$price)),
  MSE = mse(pred.1, log(test.data$price)),
  R2 = R2(pred.1, log(test.data$price))
)

#RMSE        MSE        R2
#1 0.1665681 0.02774495 0.9002762

#### Graph No. 3
trellis.par.set(caretTheme())
densityplot(kc_model, pch = "|", resamples="all" ,lwd = 2, 
            col = "dark green", main = "Gradient Boosting Model")

##### RMSE and MSE reduced considerably and R2 (accuracy) increased considerably. Thus this model prooved to be the best fit for our initial model. 
