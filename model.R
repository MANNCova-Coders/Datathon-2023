### Code for the model

data = read.csv("trainings.csv")

library(randomForest)
data_new = read.csv("trainings_train_processed.csv")
data_test = read.csv("trainings_test_processed.csv")
training_data = data[data$type != "",]
test_data = data[data$type == "",]
training_data$avg_hr <- data_new$avg_hr
training_data$uphill <- data_new$uphill
training_data$downhill <- data_new$downhill
training_data$distance <- data_new$distance
training_data$duration <- data_new$duration
training_data$avg_speed <- training_data$distance/training_data$duration 
training_data$type <- factor(training_data$type)

test_data$avg_hr <- data_test$avg_hr
test_data$uphill <- data_test$uphill
test_data$downhill <- data_test$downhill
test_data$distance <- data_test$distance
test_data$duration <- data_test$duration
test_data$avg_speed <- test_data$distance/test_data$duration 
test_data$type <- factor(test_data$type)

full_data = rbind(training_data, test_data)


user_id_unique <- unique(data_test$user_id)
final_pred <- c()
train_id <- c()
for (i in 1:length(user_id_unique)){
  set.seed(0)
  user = user_id_unique[i]
  data_user =full_data[full_data$user_id == user,]
  data_user$type <- droplevels(data_user$type)
  order_user = order(data_user$start_date)
  data_user_ordered = data_user[order_user,]
  training_ind <- which(data_user_ordered$type != "")
  time = seq(1, length(order_user))
  type <- data_user_ordered$type[training_ind]
  data_rf <- data_user_ordered[training_ind, c(4,5,6,7,8,9,10,12,13)]
  data_rf$time <- time[training_ind]
  data_rf$type <- droplevels(data_rf$type)
  rf <- randomForest(type~., data = data_rf, na.action = na.exclude)
  data_test <- data_user_ordered[-training_ind, c(4,5,6,7,8,9,10,12,13)]
  data_test$time <- time[-training_ind]
  pred = predict(rf, data_test)
  final_pred = c(final_pred, as.character(pred))
  train_id = c(train_id, data_user_ordered[-training_ind, "training_id"])
}



final = read.csv("exam_dataset.csv")

for (i in 1:250){
  j = which(train_id == final$training_id[i])
  final$type[i] = final_pred[j]
}
 
write.csv(final, "solutions.csv")
