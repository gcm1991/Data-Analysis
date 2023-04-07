rm(list = ls())
setwd("C:/Users/cayde/OneDrive/Graduate Studies/Misc/TA/Lab 102/Data Sets")

library(ggplot2)
library(dplyr)
library(e1071)
library(randomForest)

anes <- read.csv("anes_timeseries_2016.csv") 

#Inspecting the Data
#####################

head(anes)
str(anes)

sapply(anes, class) #look at the data types
sort(sapply(anes, function(x){mean(is.na(x))*100})) #get a sense of which columns are missing data

summary(anes$feeling_dem_party)
summary(anes$feeling_rep_party)

#Splitting data
###################

set.seed(452)
anes$assignment <- sample(c("training", "test"), nrow(anes), replace=TRUE, prob = c(0.7, 0.3))

#Data Transformation
####################

table(anes$vote_choice_president)
anes_2party <- filter(anes, vote_choice_president == '1. Hillary Clinton' | vote_choice_president == '2. Donald Trump')
anes_2party$vote_choice_president <- recode(anes_2party$vote_choice_president, "1. Hillary Clinton" = "Clinton", "2. Donald Trump" = "Trump")
table(anes_2party$vote_choice_president)

anes_2party$vote_choice_president <- as.factor(anes_2party$vote_choice_president)



#Data Visualization 

ggplot(data = anes_2party, mapping = aes(x = feeling_dem_party, y = feeling_rep_party)) +
  geom_point(aes(color = vote_choice_president)) +
  geom_smooth(method='lm') +
  scale_color_manual(values = c("blue", "red", "gray", "gray")) +
  xlab("Democratic Party") +
  ylab("Republican Party") + 
  ggtitle("Feeling Thermometer for Parties") + 
  guides(color = guide_legend("Vote Choice")) 
  
################
#Classification 
################

#Feelings
#########


DV <- "vote_choice_president"
feeling_variables <- c("feeling_dem_party", "feeling_rep_party")

#Subsetting the data by relevant variables and removing NA's
anes_2party_feelings <- subset(anes_2party, select = c(DV, feeling_variables, "assignment"))
anes_2party_feelings <- na.omit(anes_2party_feelings)

#Splitting the new data into testing and training sets based on previous assignment  
anes_2party_feelings_train <- filter(anes_2party_feelings, assignment == "training")
anes_2party_feelings_train <- subset(anes_2party_feelings_train, select = c(DV, feeling_variables))

anes_2party_feelings_test <- filter(anes_2party_feelings, assignment == "test")
anes_2party_feelings_test <- subset(anes_2party_feelings_test, select = c(DV, feeling_variables))

#Estimating a model on the training set  
logit_feelings <- glm(vote_choice_president ~ ., 
                      data = anes_2party_feelings_train,
                      family = "binomial")

predictions_feelings <- round(predict.glm(logit_feelings, newdata = anes_2party_feelings_test, type = "response"))
predictions_feelings[predictions_feelings == 0] <- "Clinton"
predictions_feelings[predictions_feelings == 1] <- "Trump"

table(predictions_feelings, anes_2party_feelings_test$vote_choice_president)

#Policy 
#######

#Identifying the relevant variables for the model
policy_variables <- c("have_health_insurance", "favor_2010_health_care_law", 
                        "party_better_economy", "importance_of_guns", "policy_immigrants", 
                        "build_wall", "speak_english", "syrian_refugees", "gov_waste_.tax_money", 
                        "global_warming", "death_penalty")

#Subsetting the data by relevant variables and removing NA's
anes_2party_policy <- anes_2party %>%
  subset(select = c(DV, policy_variables, "assignment")) %>%
  na.omit()

#Splitting the new data into testing and training sets based on previous assignment  
anes_2party_policy_train <- anes_2party_policy %>%
  filter(assignment == "training") %>%
  subset(select = c(DV, policy_variables))

anes_2party_policy_test <- anes_2party_policy %>%
  filter(assignment == "test") %>%
  subset(select = c(DV, policy_variables))

#Estimating a model on the training set  
logit_policy <- glm(vote_choice_president ~ ., 
                      data = anes_2party_policy_train,
                      family = "binomial")

predictions_policy <- round(predict.glm(logit_policy, newdata = anes_2party_policy_test, type = "response"))
predictions_policy[predictions_policy == 0] <- "Clinton"
predictions_policy[predictions_policy == 1] <- "Trump"

table(predictions_policy, anes_2party_policy_test$vote_choice_president)



#Demographics

#Subsetting the data by relevant variables and removing NA's
demographic_variables <- c("religion_important", "age", "married", "education", "hispanic", "white")

anes_2party_demo <- anes_2party %>%
  subset(select = c(DV, demographic_variables, "assignment")) %>%
  na.omit()

#Splitting the new data into testing and training sets based on previous assignment  
anes_2party_demo_train <- anes_2party_demo %>%
  filter(assignment == "training") %>%
  subset(select = c(DV, demographic_variables))

anes_2party_demo_test <- anes_2party_demo %>%
  filter(assignment == "test") %>%
  subset(select = c(DV, demographic_variables))

#Estimating a model on the training set
logit_demographics <- glm(vote_choice_president ~.,
                          data = anes_2party_demo_train,
                          family = "binomial")

predictions_demographics <- round(predict.glm(logit_demographics, newdata = anes_2party_demo_test, type = "response"))
predictions_demographics[predictions_demographics == 0] <- "Clinton"
predictions_demographics[predictions_demographics == 1] <- "Trump"

table(predictions_demographics, anes_2party_demo_test$vote_choice_president)

#Full Model

sapply(anes_2party, function(x){mean(is.na(x))})

anes_2party_full <- anes_2party %>%
  subset(select = c(DV, policy_variables, feeling_variables, demographic_variables, "assignment")) %>%
  na.omit()

#Splitting the new data into testing and training sets based on previous assignment  
anes_2party_full_train <- anes_2party_full %>%
  filter(assignment == "training") %>%
  subset(select = c(DV, policy_variables, feeling_variables, demographic_variables))

anes_2party_full_test <- anes_2party_full %>%
  filter(assignment == "test") %>%
  subset(select = c(DV, policy_variables, feeling_variables, demographic_variables))

logit_full <- glm(vote_choice_president ~.,
                  data = anes_2party_full_train,
                  family = "binomial")

predictions_full <- round(predict.glm(logit_full, newdata = anes_2party_full_test, type = "response"))
predictions_full[predictions_full == 0] <- "Clinton"
predictions_full[predictions_full == 1] <- "Trump"

table(anes_2party_full_test$vote_choice_president, predictions_full)


#Support Vector Machine 
########################


classifier <- svm(formula = vote_choice_president ~ ., data = anes_2party_full_train, type = 'C-classification', kernal = 'linear')

predictions <- predict(classifier, newdata = anes_2party_full_test)

confustion_matrix <- table(anes_2party_full_test$vote_choice_president, predictions)
confustion_matrix


#Visualization

plot(classifier, training)


plot(classifier, training, svSymbol = "X", dataSymbol = "O")


#Random Forest
###################

rf_model <- randomForest(vote_choice_president ~ ., data = anes_2party_full)
print(rf_model)


