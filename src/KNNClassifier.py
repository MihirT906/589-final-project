from DistanceCalculator import EuclideanDistanceCalculator 
import pandas as pd
from collections import Counter
import numpy as np

class KNN_Classifier:
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        
    def get_params(self):
        return ['k= ' + str(self.k)]
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return
        
    def predict(self, X_test):
        
        def calculate_distance(point):
            '''
            Calculate the distance of a point in X_test with all data points in X_train
            '''
            distances = []
            for index, x_row in enumerate(self.X_train):
                distances.append((index,EuclideanDistanceCalculator(point, x_row).calculate_distance()))
            
            return distances
        
        def get_k_neighbors(distances):
            '''
                return the indices of the data points which are the k nearest neighbors
            '''
            sorted_distances = sorted(distances, key=lambda x: x[1])
            closest_indices = [index for index, _ in sorted_distances[:self.k]]
            return closest_indices
        
        def get_majority_vote(closest_indices):
            '''
                Get the labels of the k neighbors and compute the majority vote
            '''
            neighbors = []
            for i in closest_indices:
                neighbors.append(self.y_train[i])
            majority_class = Counter(neighbors).most_common(1)[0][0]
            return majority_class

        labels = []
        for x_row in X_test:
            labels.append(get_majority_vote(get_k_neighbors(calculate_distance(x_row))))
            
        return np.array(labels)
            