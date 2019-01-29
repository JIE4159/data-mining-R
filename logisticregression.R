mydata <- read.csv("/Users/mengzhang/Desktop/5703 project/report 2/training data.csv")
head(mydata)
summary(mydata)
sapply(mydata, sd)  ###standard deviations  DELETE the NA variables

#logistic
###divide sample Sample Indexes deletecode
indexes = sample(1:nrow(mydata), size=0.5*nrow(mydata))
# Split data
test = mydata[indexes,]
dim(test)  
train = mydata[-indexes,]
dim(train) 

###logistic regression model
model <- glm(LoanStatus ~  Source  +  ModifiedCreditScore+ ModifiedBankruptcyScore +EmploymentStatus + EmployedMonths +PrevEmployedMonths +CoEmployedMonths
             + PrimeMonthlyIncome +CototalMonthlyIncome + PrimeMonthlyLiability + PrimeMonthlyRent + TotalMonthlyDebtBeforeLoan
             +VehicleMileage + isNewVehicle + TotalVehicleValue +AmountRequested + DownPayment + OccupancyDuration 
             +  LTV  + MemberIndicator + CoApplicantIndicator , data = train, family = "binomial")
summary(model)

step(model, scope=list(lower=model, upper=model), direction="forward")
step(model, data=train, direction="backward")

trainmodel <- glm(LoanStatus ~ Source + ModifiedCreditScore + ModifiedBankruptcyScore + 
                    EmploymentStatus + EmployedMonths + CoEmployedMonths + PrimeMonthlyIncome + 
                    CototalMonthlyIncome + TotalMonthlyDebtBeforeLoan + VehicleMileage + 
                    isNewVehicle + TotalVehicleValue + AmountRequested + OccupancyDuration + 
                    LTV + MemberIndicator + CoApplicantIndicator , data = train, family = "binomial")
summary(trainmodel)

#confusion matrix
predict <- predict(trainmodel, type = 'response')
#ROCR Curve
table(train$LoanStatus,predict > 0.5)
####predict accuracy
predicted <- predict(trainmodel, test,type="response")
library(InformationValue)
optCutOff <- optimalCutoff(test$LoanStatus, predicted)[1] 
misClassError(test$LoanStatus, predicted, threshold = optCutOff)
plotROC(test$LoanStatus, predicted)

confusionMatrix(test$LoanStatus, predicted, threshold = optCutOff)
sensitivity(test$LoanStatus, predicted, threshold = optCutOff)
specificity(test$LoanStatus, predicted, threshold = optCutOff)


#boosting
CFE<-mydata[c(-0,-1)]  ###subset variables excepte ID
factor(CFE$LoanStatus)
factor(CFE$Source)
factor(CFE$EmploymentStatus)
factor(CFE$VehicleYear)
factor(CFE$VehicleMake)
factor(CFE$isNewVehicle)
factor(CFE$OccupancyStatus)
factor(CFE$RequestType)
factor(CFE$MemberIndicator)
factor(CFE$CoApplicantIndicator)


######################### LDA based on  cross validation $$$$$$$$$$$$$$$$$$$$$$$$$$$
##check distribution of some features
scoredensity<-density(CFE$ModifiedCreditScore)
plot(scoredensity,xlim=c(200,1000))
bankruptdensity<-density(CFE$ModifiedBankruptcyScore)
plot(bankruptdensity)
AmountRequesteddensity<-density(CFE$AmountRequested)
plot(AmountRequesteddensity,xlim=c(0,100000))
TotalVehicleValuedensity<-density(CFE$TotalVehicleValue)
plot(TotalVehicleValuedensity,xlim=c(0,80000))
lda.CFE<-lda(LoanStatus ~ ., data = CFE)
CFE.cv <- lda(LoanStatus ~ ., CV=TRUE, data = CFE)
table(CFE$LoanStatus, CFE.cv$class, dnn = c('Actual','Predicted'))
####################### Lass Regression based on cross validation ## features reduction#######
factor(CFE$LoanStatus)
x=model.matrix(LoanStatus~.,CFE)[,-1]
y=CFE$LoanStatus
train<-sample(1:nrow(x),nrow(x)/2)
test<-(-train)
library(glmnet)
y.test=y[test]
dat=data.frame(x=x,y=as.factor(y))
grid=10^seq(10,-2,length=100)
lasso.mod=glmnet(x[train,],y[train],alpha = 1,lambda = grid,family = "binomial")
plot(lasso.mod)
set.seed(2)
cv.out<-cv.glmnet(x[train,],y[train],alpha=1,family = "binomial",)
plot(cv.out)
bestlambda<-cv.out$lambda.min
lasso.pred<-predict(lasso.mod,s=bestlambda,newx = x[test,],type = "class")
coef(lasso.mod,s=bestlambda)        ##features selection and final model
###error rates table
table<-table(lasso.pred,y.test)      
mean(lasso.pred==y.test)
####################### svm   #############
library(gbm)
set.seed(1)
train<-1:50000
CFE.train<-CFE[train,]
CFE.test<-CFE[-train,]
set.seed(1)
boost.CFE<-gbm(LoanStatus~.,data=CFE.train,distribution = "gaussian",n.trees = 1000,shrinkage = 0.01)
summary(boost.CFE)
plot(boost.CFE,i="ModifiedCreditScore")
plot(boost.CFE,i="DTI")
plot(boost.CFE,i="MemberIndicator")
plot(boost.CFE,i="VehicleMileage")
probs.test<-predict(boost.CFE,CFE.test,n.trees = 1000,type = "response")
pred.test<-ifelse(probs.test>0.5,1,0)   ##p=0.5
table(CFE.test$LoanStatus,pred.test)
errorate<-(3840+6869)/(dim(CFE)[1]-50000)
pred.test2<-ifelse(probs.test>0.4,1,0)   ##p=0.4
table(CFE.test$LoanStatus,pred.test2)
errorate<-(1257+11665)/(dim(CFE)[1]-50000)
pred.test3<-ifelse(probs.test>0.6,1,0)   ##p=0.6
table(CFE.test$LoanStatus,pred.test3)
errorate<-(5281+5163)/(dim(CFE)[1]-50000)

