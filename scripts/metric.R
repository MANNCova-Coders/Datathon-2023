## Metric
data = read.csv("trainings.csv")

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
data = full_data
user_id_unique <- unique(data$user_id)
days_year <- seq(1, 365)
user = user_id_unique[11]
data_user = data[data$user_id == user,]
order_user = order(data_user$start_date)
data_user_ordered = data_user[order_user,]
days = yday(data_user_ordered$start_date)

freq <- c(0)
for (i in 1:(nrow(data_user_ordered)-1)){
  freq <- c(freq, as.numeric(as.Date(data_user_ordered$start_date[i+1])-as.Date(data_user_ordered$start_date[i])))
}

data_year = data.frame(matrix(0, nrow = 365, ncol = 7))
colnames(data_year) <- c("distance", "duration", "avg_hr", "max_hr", "avg_speed", "temperature", "freq")
j = 1

for (i in 1:365){
  if (i %in% days){
    data_year$distance[i] = data_user_ordered[j, "distance"]
    data_year$duration[i] = data_user_ordered[j, "duration"]
    data_year$avg_hr[i] = data_user_ordered[j, "avg_hr"]
    data_year$max_hr[i] = data_user_ordered[j, "max_hr"]
    data_year$temperature[i] = data_user_ordered[j, "temperature"]
    data_year$avg_speed[i] = data_user_ordered[j, "distance"]/data_user_ordered[j, "duration"]
    #data_year$freq[i] = freq[j]
    if (freq[j] != 0) {
      data_year$freq[(i-freq[j] + 1) : (i - 1)] = seq(1, freq[j] - 1)
    }
    j = j +1
  } 
}


optimal_temp <- 7
max_dist = max(data_year$distance)
max_dur = max(data_year$duration)
max_speed = max(data_year$avg_speed)
max_avg_hr = max(data_year$avg_hr)
max_max_hr = max(data_year$max_hr)
data_year$score = 0
j = days[1]
for (i in 1:365){
  data_year$score[i] = (((3*data_year$distance[i]/max_dist + 3*data_year$duration[i]/max_dur + 3*data_year$avg_speed[i]/max_speed + 3*((data_year$avg_speed[i]-data_year$avg_speed[j])/max_speed)*(data_year$avg_speed[i] >0) + 3*((data_year$distance[i]-data_year$distance[j])/max_dist)*(data_year$distance[i] >0)- data_year$avg_hr[i]/max_avg_hr - data_year$max_hr[i]/max_max_hr)-abs(data_year$temperature[i]-optimal_temp)/20)+i/1000*(data_year$avg_hr[i] == 0)) / (10*(data_year$freq[i]>5) + 2)
  if (i %in% days){
    j = i
  }
}


data_year$final_score = 0
for (i in 2:365){
  k = i - 1
  weights = 1/(i - seq(1, k))
  x1 = data_year$score[1:k]
  data_year$final_score[i] = weighted.mean(x1, weights)
}

data_year$final_final_score = 0
for (i in 2:365){
  data_year$final_final_score[i] = mean(data_year$final_score[((i-4)*(i-4 >=0)):i])
}

par(mfrow=c(1,1))
plot(days_year[-1], data_year$final_final_score[-1], xlab = "days", ylab = "Performance")
lines(days_year[-1], data_year$final_final_score[-1])
