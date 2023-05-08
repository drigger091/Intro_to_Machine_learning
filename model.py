import numpy as np
import statistics

class KNN_Classifier():

  # initiating the parameters
  def __init__(self, distance_metric):

    self.distance_metric = distance_metric

  # getting the distance metric
  def get_distance_metric(self,training_point, test_point):

    if (self.distance_metric == 'euclidean'):

      dist = 0
      
      for i in range(len(training_point) - 1):   # excluding the target column
          
         dist = dist + (training_point[i] - test_point[i])**2

      euclidean_dist = np.sqrt(dist)
    
      return euclidean_dist

    elif (self.distance_metric == 'manhattan'):

      dist = 0

      for i in range(len(training_point) - 1):
          
        dist = dist + abs(training_point[i] - test_point[i])

      manhattan_dist = dist

      return manhattan_dist

  # getting the nearest neighbors
  def nearest_neighbors(self,X_train, test_data, k):

    distance_list = []

    for training_data in X_train:

      distance = self.get_distance_metric(training_data, test_data)
      
      distance_list.append((training_data, distance))

    distance_list.sort(key=lambda x: x[1])     # sorting by the distance

    neighbors_list = []

    for j in range(k):
        
      neighbors_list.append(distance_list[j][0])   # we are only considering the points not the distance

    return neighbors_list


  # predict the class of the new data point:
  def predict(self,X_train, test_data, k):
      
    neighbors = self.nearest_neighbors(X_train, test_data, k)
    
    for dt in neighbors:
        
      label = []
      
      label.append(dt[-1]) # we are only taking the last label or the target class

    predicted_class = statistics.mode(label) #using the mode to signify the majority of labels reappearing

    return predicted_class