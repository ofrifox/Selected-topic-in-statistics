
library(magrittr)
library(data.table)
library(dplyr)
library(tidyverse)
##install.packages("psych")
library(psych)
library(plyr)
library(ggplot2)
library(purrr)
library(tibble) 
library(tidyr)
library(ggplot2)
library(Hmisc)
library(magrittr)
##install.packages("GPLTR")

raw_data <-read.csv("C:/Users/ofrifox/Documents/לימודים/שנה ג/סמסטר א/נושאים נבחרים/פרוייקט/Machine Learning YouTube Meta Data.csv")

#Check for duplicates:
duplicated(raw_data) %>% any()#All the songs ids are unique in each sample.

#Removing some variables: 
clean_data <- subset(raw_data, select = -c(position))  ## remove the id 

## na to 0
clean_data[is.na(clean_data)] <- 0


#Making Duration in log transformation :


clean_data$durationSec <- log(clean_data$durationSec) 
clean_data <- clean_data %>% dplyr::rename(log_durationSec=durationSec)
hist(clean_data[,'log_durationSec'],xlab='log_durationSec' , main = 'log_durationSec')

clean_data$likeCount <- log(clean_data$likeCount) 
clean_data <- clean_data %>% dplyr::rename(log_likeCount=likeCount)
hist(clean_data[,'log_likeCount'],xlab='log_likeCount' , main ='log_likeCount')


clean_data$dislikeCount <- log(clean_data$dislikeCount) 
clean_data <- clean_data %>% dplyr::rename(log_dislikeCount=dislikeCount)
hist(clean_data[,'log_dislikeCount'],xlab='log_dislikeCount' , main ='log_dislikeCount')

clean_data$commentCount <- log(clean_data$commentCount) 
clean_data <- clean_data %>% dplyr::rename(log_commentCount=commentCount)
hist(clean_data[,'log_commentCount'],xlab='log_commentCount' , main ='log_commentCount')


clean_data$viewCount <- log(clean_data$viewCount) 
clean_data <- clean_data %>% dplyr::rename(log_viewCount=viewCount)
hist(clean_data[,'log_viewCount'],xlab='log_viewCount' , main ='log_viewCount')

##numeric
numerical_vars<-c('log_viewCount','log_likeCount','log_dislikeCount','log_commentCount','log_durationSec')


## make "-nf" to 0
clean_data[clean_data$log_likeCount<0,"log_likeCount"]<-0
clean_data[clean_data$log_dislikeCount<(-1),"log_dislikeCount"]<-0
clean_data[clean_data$log_commentCount<0,"log_commentCount"]<-0

## corr in the numeric 

cors <- function(df) {
  M <- Hmisc::rcorr(as.matrix(df))
  Mdf <- map(M, ~data.frame(.x))
}

formatted_cors <- function(df){ 
  cors(df) %>% 
    map(~rownames_to_column(.x, var="measure1")) %>% 
    map(~pivot_longer(.x, -measure1, "measure2")) %>% 
    bind_rows(.id = "id") %>% 
    pivot_wider(names_from = id, values_from = value)%>% 
    mutate(sig_p = T, p_if_sig = P, r_if_sig = r) 
  }
formatted_cors(clean_data[,numerical_vars]) %>% 
  ggplot(aes(measure1, measure2, fill=r, label=round(r_if_sig,2))) + 
  geom_tile() + 
  labs(x = NULL , y = NULL, fill = "Pearson's\nCorrelation", title="Correlations in the data") + 
  scale_fill_gradient2(mid="#FBFEF9",low="#0C6291",high="#A63446", limits=c(-1,1)) + 
  geom_text() + theme(axis.text.x = element_text(angle = 90)) + 
  scale_x_discrete(expand=c(0,0)) + 
  scale_y_discrete(expand=c(0,0))



#Describing each factorial variable distribution : 
factorial_vars <- c("videoCategoryLabel","definition","caption","DAY","licensedContent") 
par(mfrow = c(3,2))
for ( i in factorial_vars){ 
  plot(as.factor(clean_data[,i]),xlab=i , main = i) 
  # print(class(as.numeric(clean_data[,i]))) 
}



## block 
clean_data$DAY <- as.factor(clean_data$DAY)
clean_data$videoCategoryLabel <- as.factor(clean_data$videoCategoryLabel)
av<-lm(log_viewCount ~ DAY + videoCategoryLabel +log_likeCount + log_dislikeCount + log_commentCount
         + log_durationSec + publishedAt.year + publishedAt.month+ 
           definition + licensedContent,data=clean_data)
summary(av)
anova(av)

## make factor 
#clean_data$DAY <- as.factor(clean_data$DAY)
#clean_data$videoCategoryLabel <- as.factor(clean_data$videoCategoryLabel)
clean_data$definition <- as.factor(clean_data$definition)
clean_data$licensedContent <- as.factor(clean_data$licensedContent)
clean_data$caption<- as.factor(clean_data$caption)
   
# Describing each continues variable distribution : 
par(mfrow = c(2,3)) 
for ( i in numerical_vars){ 
  hist(clean_data[,i],xlab=i , main = i) }


summary(clean_data)

## make a copy 
data <- subset(clean_data, select = -c(position))



########################### model########################
library(MASS)
library(caret)
library(tidyverse)
library(lme4)
library(olsrr)
library(plyr)
library(readr)
library(dplyr)
library(ggplot2)
library(repr)
glimpse(clean_data)
library(glmnet)
#install.packages("repr")
#install.packages("glmnet")

summary(clean_data)

####Train\Test Set
set.seed(100) 
index = sample(1:nrow(clean_data), 0.8*nrow(clean_data)) 
train = clean_data[index,] # Create the training data 
test = clean_data[-index,] # Create the test data
dim(train)
dim(test)

cols_reg = c('publishedAt.year', 'publishedAt.month', 'videoCategoryLabel'
             , 'DAY', 'log_durationSec', 'definition', 'caption',
             'licensedContent', 'log_likeCount', 'log_dislikeCount',
             'log_commentCount', 'log_viewCount')

dummies <- dummyVars(log_viewCount ~ ., data = clean_data[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])
print(dim(train_dummies)); print(dim(test_dummies))


x = as.matrix(train_dummies)
y_train = train$log_viewCount
x_test = as.matrix(test_dummies)
y_test = test$log_viewCount

###EVAL RESULTE
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

#############################################Linear Model###############################
model = lm (log_viewCount ~ log_likeCount + log_dislikeCount + log_commentCount
            + log_durationSec + publishedAt.year + publishedAt.month + DAY +
              videoCategoryId + definition + licensedContent, data = clean_data)
model2 = lm (log_viewCount ~ 1, data= train)
scope <- list(upper=model , lower=model2)
step_reg <- step(model2 , scope = scope , direction = "both" ,
                 trace=TRUE,k=log(nrow(train)))

k <- ols_step_all_possible(model)
#library(olsrr)
f <- ols_step_forward_p(model, penter = 0.05)
b <- ols_step_backward_p(model, prem =  0.05)
both <- ols_step_both_p(model, penter = 0.05, prem = 0.05)

predictions_test = predict(step_reg, newdata = test)
eval_results(test$log_viewCount, predictions_test, test)


############################################polynomial model
model_pol <- lm (log_viewCount ~ log_likeCount +I(log_likeCount^2)+ I(log_durationSec^2) + publishedAt.year 
                 + log_likeCount:log_dislikeCount + log_likeCount:log_commentCount
                 + log_likeCount:publishedAt.year + log_likeCount:publishedAt.month 
                 + log_likeCount:licensedContent +log_commentCount:publishedAt.year
                 + log_commentCount:videoCategoryId + log_durationSec:DAY 
                 + DAY:videoCategoryId ,data = train)

model2.1 = lm (log_viewCount ~ 1, data= train)
scope_pol <- list(upper=model_pol , lower=model2.1)
step_reg_pol <- step(model2.1 , scope = scope_pol , direction = "both" ,
                     trace=TRUE,k=log(nrow(train)))

step_predict_pol <- predict(step_reg_pol, newdata=test)
summary(model_pol)
# predicting and evaluating the model on test data
predictions_test_pol = predict(step_reg_pol, newdata = test)
eval_results(test$log_viewCount, predictions_test_pol, test)






################################ random effect
library(glmnet)
library(olsrr)


model = lm(log_viewCount ~ log_likeCount + log_dislikeCount + log_commentCount
            + publishedAt.year + publishedAt.month +videoCategoryLabel+ DAY 
            , data = data)

## day 
library(lme4)
library(glmnet)

library(lmerTest)
lme.reg.day <-lme4::lmer(log_viewCount ~ (1|DAY) + (1|videoCategoryLabel), data =clean_data) 
summary(lme.reg.day)

anova(lme.reg.day,model)

##videoCategoryLabel
lme.reg.videoCategoryLabel <-lme4::lmer(log_viewCount ~ (1|videoCategoryLabel),data =clean_data ) 
summary(lme.reg.videoCategoryLabel)

anova(lme.reg.videoCategoryLabel,model)

##videoCategoryLabel & day 

lme.reg.videoCategoryLabel.day <- lmer(log_viewCount ~ log_likeCount + log_dislikeCount + log_commentCount
                                   + log_durationSec + publishedAt.year + publishedAt.month+ (1|DAY)+ 
                                     definition + licensedContent+(1|videoCategoryLabel),data = clean_data ) 
summary(lme.reg.videoCategoryLabel.day)




###########################################RIDGE
lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements ridge regression
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

# Prediction and evaluation on train data
predictions_train_rig <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train_rig, train)

# Prediction and evaluation on test data
predictions_test_rig <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test_rig, test)

print(eval_results(y_test, predictions_test_rig, test))



###########################################LASSO
lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best_lasso <- lasso_reg$lambda.min 
lambda_best_lasso

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best_lasso, standardize = TRUE)

predictions_train_lasso <- predict(lasso_model, s = lambda_best_lasso, newx = x)
eval_results(y_train, predictions_train_lasso, train)

predictions_test_lasso <- predict(lasso_model, s = lambda_best_lasso, newx = x_test)
eval_results(y_test, predictions_test_lasso, test)

print(eval_results(y_test, predictions_test_lasso, test))




############################################# Not Linear Model###############################

#############################################DECISION TREE
##install.packages("rpart.plot")
##install.packages("tree")
library(rpart)
library(rpart.plot)
library(caret)
library(tree)


fit <- rpart(log_viewCount~  log_commentCount + videoCategoryId + log_likeCount
             + publishedAt.year + publishedAt.month,
             method="anova", data=train)

rpart.plot(fit, box.palette="RdBu", shadow.col="gray", nn=TRUE)


# R-SQUARE
predictions_test_tree = predict(fit, newdata = test)
eval_results(test$log_viewCount, predictions_test_tree, test)



###########################################KNN
#install.packages('FNN')
library(FNN)
library(class)

Mse_Vector <- vector ()
for ( i in seq(10,100,5) ){
  knn.1 <- FNN::knn.reg(train=x, test= x_test , y = train$log_viewCount
                        , k=i )
  knn.mse <- MSE(knn.1$pred- (test$log_viewCount))
  Mse_Vector <- append (Mse_Vector, knn.mse)
}

{plot(seq(10,100,5),Mse_Vector, type='l', xlab = "Number of Neighbors -
K " , main = "MSE of KNN" ,ylab="MSE")
  abline(v=30, col = "red")}

MSE <- function(x) x^2 %>% mean
knn.2 <- FNN::knn.reg(train=x, test= x_test , y = train$log_viewCount, k=30)
knn.mse <- MSE(knn.2$pred-(test$log_viewCount))


r2.knn <- 1 - knn.mse/var(train$log_viewCount)
r2.knn
cat("The R-Square Of the KNN is: ", r2.knn)




###################################################best model print

library(glmnet)

ridge_reg_optimal = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = 0.3162278)

coef(ridge_reg_optimal)
pval(ridge_reg_optimal)

##Validate our assumption in the step-wise regression: 

library(car)
par(mfrow=c(1,2)) 
plot(predict(ridge_reg, s = optimal_lambda, newx = x),resid(ridge_reg, s = optimal_lambda, newx = x),ylab="Residuals",xlab="Fitted Values" ) 
abline(0,0) 
qqnorm(predict(ridge_reg, s = optimal_lambda, newx = x),resid(ridge_reg, s = optimal_lambda, newx = x)) 
qqline(predict(ridge_reg, s = optimal_lambda, newx = x),col = "steelblue")


install.packages('FrF2')
library(FrF2)

x=FrF2(16, 5)
write.csv(x,"Path to export the DataFrame\\File Name.csv", row.names = FALSE)


