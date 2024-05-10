import numpy as np
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier 
from collections import Counter
np.random.seed(42)


class RandomForestClassifier:
    def __init__(self, ntree = 2, criterion ="information_gain", threshold=1, maximal_depth=10):
        self.ntree = ntree
        self.maximal_depth=maximal_depth
        self.forest = []
        self.max_depth_of_tree = 0
        for i in range(ntree):
            DTC = DecisionTreeClassifier(criterion, threshold=threshold, maximal_depth=maximal_depth)
            self.forest.append(DTC)
    
    def fit(self, X_train, y_train, X_col_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_col_type = X_col_type
        D = list(range(0, X_train.shape[0], 1))
        L = list(range(0,X_train.shape[1],1))
        self.bootstraps = self._create_bootstraps(D)
        for i, bootstrap in enumerate(self.bootstraps):
            #print(bootstrap)
            self.forest[i].fit(X_train[bootstrap], y_train[bootstrap], X_col_type)
        
        self.max_depth_of_tree = max([tree.max_depth_of_tree for tree in self.forest])
    def predict(self, X_test):
        #print("preduction")
        labels = []
        for idx, x in enumerate(X_test):
            #print(idx)
            labels.append(self._predict_instance(x))
            #print(labels[idx])

        return labels  
    def _predict_instance(self, x):
        
        votes = []
        for i in range(len(self.forest)):
            votes.append(self.forest[i]._predict_instance(x))
        
        if not votes:
            return None
        #print(votes)
        counter = Counter(votes)
        max_occuring_element, _ = counter.most_common(1)[0]
        
        return max_occuring_element
        
            
                  
        
        
    def _create_bootstraps(self, D):
        bootstraps = []
        for _ in range(self.ntree):
            #temp_arr = []
            #for _ in range(len(D)):
                #temp_arr.append(np.random.choice(D, size=len(D), replace=True))
            bootstraps.append(np.random.choice(D, size=len(D), replace=True).tolist())
                
        return bootstraps        
        


# rfc = RandomForestClassifier(50)
# data = pd.read_csv(r"./datasets/hw3_house_votes_84.csv", sep='\t')
# from sklearn.utils import shuffle
# #data = shuffle(data)
# X=data.iloc[:,0:16].values
# y=data.iloc[:,-1].values
# X_col_type = [0 for _ in range(16)]
# #print(len(y))
# #print(X_col_type[0,1])
# rfc.fit(X, y, X_col_type)
# print("Predictin")
# print(len(rfc.predict(X)))
# #rfc._predict_instance(X[6])
# from multiple_utils import compute_confusion_matrix, calculate_accuracy, calculate_recall, calculate_precision, calculate_f1score
# confusion_matrix, n = compute_confusion_matrix(y, rfc.predict(X), [1,2,3])
# print("confusion matric here:", confusion_matrix)
# print(calculate_accuracy(confusion_matrix, n))
# print(calculate_precision(confusion_matrix))
# print(calculate_recall(confusion_matrix))
# print(calculate_f1score(calculate_precision(confusion_matrix), calculate_recall(confusion_matrix)))