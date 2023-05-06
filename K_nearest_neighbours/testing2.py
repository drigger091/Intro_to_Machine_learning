import statistics
import numpy as np

class KNN_classifier():

    def __init__(self,distance_metric):

    
        self.get_distance_metric = distance_metric


    def get_distance_metric(self,training_point,testing_point):  # getting the distance metric

        if (self.get_distance_metric == "euclidean"):

            dist = 0

            for i in range(len(training_point) - 1):  # Iterate over the length of the lists as we will exclude the target variable
                    
                    dist = dist + (training_point[i] - testing_point[i])**2

                    
                    euclidean_distance = np.sqrt(dist)

                    return euclidean_distance
        
        elif (self.get_distance_metric == "manhattan"):

            dist = 0

            for i in range(len(training_point) -1):

                dist = dist + abs(training_point[i] - testing_point[i])

                manhattan_distance = dist

                return manhattan_distance
# getting the nearest neighbours

    def nearest_neighbour(self ,X_train,test_data , k):
        
        distance_list = []

        for training_data in X_train:

            distance = self.get_distance_metric(training_data,test_data)
            distance_list.append(training_data,distance)  # adding the individual points along with the distance with the required point

        distance_list.sort(key=lambda x:x[1])   # sorting the list based on the distance ascending order

        neighbours_list = []

        for j in range(k):
            neighbours_list.append(distance_list[j][0])    #fetching only the data not the distance so we can find the neighbouring points

        return neighbours_list 

    def predict(self,X_train ,test_data , k):
        neighbours = self.nearest_neighbour(X_train,test_data,k)

        for dt in neighbours:
            label = []
            label.append(dt[-1])

        predicted_class = statistics.mode(label)   # so as we know the majority wins , so which class has highest numbers in the K neighbours it prevails

        return predicted_class