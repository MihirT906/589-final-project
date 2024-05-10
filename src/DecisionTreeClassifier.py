import math
from collections import Counter
from TreeNode import TreeNode 
from ImpurityCalculator import InformationGainCalculator, GiniImpurityCalculator
import numpy as np
np.random.seed(42)

class DecisionTreeClassifier:
    def __init__(self, criterion, threshold=1, maximal_depth=10):
        if criterion not in ('information_gain', 'gini_impurity'):
            raise ValueError("Criterion must be one of: 'information_gain', 'gini_impurity'")
        self.root = None
        self.criterion = criterion
        self.threshold = threshold
        self.maximal_depth = maximal_depth
        self.impurity_calc = None
        self.max_depth_of_tree =  0
        if self.criterion == "information_gain":
             self.impurity_calc = InformationGainCalculator()
        elif self.criterion == "gini_impurity":
             self.impurity_calc = GiniImpurityCalculator()
    
    def fit(self, X_train, y_train, X_col_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_col_type = X_col_type
        D = list(range(0, X_train.shape[0], 1))
        L = list(range(0,X_train.shape[1],1))
        #L = [0,2]
        self.orig_L = L.copy()
        self.root = self._build_tree(D, L, 0)
    
    def _predict_instance(self, x):
        curr_node = self.root
        
        while len(curr_node.children)!=0:
            attribute_idx_to_test = curr_node.feature_index 
            class_label = x[attribute_idx_to_test]
            if self.X_col_type[attribute_idx_to_test] == 0:
                
                #child_node_idx = curr_node.edges.index(class_label) if class_label in curr_node.edges else -1
                closest_value_idx = -1
                closest_difference = float('inf')

                # Iterate over each value in curr_node.edges
                for idx, edge_value in enumerate(curr_node.edges):
                    difference = abs(class_label - edge_value)
                    if difference < closest_difference:
                        closest_difference = difference
                        closest_value_idx = idx

                child_node_idx = closest_value_idx
            
            
            else:
                if len(curr_node.edges) == 0:
                    child_node_idx = -1
                elif len(curr_node.edges) == 1:
                    child_node_idx = 0
                else:
                    partition_val = curr_node.edges[0][1]
                    if class_label <= partition_val:
                        child_node_idx = 0
                    else:
                        child_node_idx = 1
            # print("attribute idx to test:", attribute_idx_to_test)
            # print("class label", class_label)
            # print("child node index", child_node_idx)
            if child_node_idx == -1:
                print(f"attribute to test: {attribute_idx_to_test}") 
                print(f"class label: {class_label}") 
                print(f"Possible edges: {curr_node.edges}") 
                print(f"majority class: {curr_node.majority_class}")
                return curr_node.majority_class
            else:
                curr_node = curr_node.children[child_node_idx]
                
        return curr_node.majority_class
            
    def predict(self, X_test):
        labels = []
        for x in X_test:
            labels.append(self._predict_instance(x))

        return labels            
    
    def _build_tree(self,orig_D,orig_L, depth):
        self.max_depth_of_tree = max(self.max_depth_of_tree, depth)
        #size=int(math.sqrt(X_train.shape[1]))
        D = orig_D.copy()
        if self.criterion == 'information_gain':
            L = np.random.choice(list(range(0,self.X_train.shape[1],1)), size=int(math.sqrt(self.X_train.shape[1])),  replace=False).tolist()
        else:
            L = np.random.choice(orig_L, size=min(int(math.sqrt(self.X_train.shape[1])), len(orig_L)),  replace=False).tolist()
        #L = orig_L.copy()
        #print("L=", L)
        # L = orig_L.copy()
        
        #Create a new node
        N = TreeNode()
        X = [self.X_train[index] for index in D]
        y = [self.y_train[index] for index in D]
        total_samples = len(y)
        class_counts = Counter(y)
        majority_class = class_counts.most_common(1)[0][0]
        #N.majority_class = majority_class
        #Check for stopping condition
        for candidate_class, count in class_counts.items():
            class_proportion = count / total_samples
            if class_proportion >= self.threshold:
                #print("Majority found:", candidate_class)
                N.majority_class = candidate_class
                return N
        
        # print("Checking for 2ns stop condition:", len(D))
        if len(L) == 0 or depth > self.maximal_depth:
            
            majority_class = class_counts.most_common(1)[0][0]
            N.majority_class = majority_class
            #print("Majority found:", majority_class)
            return N
        
        #find best split
        partition_val, attribute = self._best_attribute(X, y, L)
        #print("partition_val", partition_val)
        #print("attribute:", L[attribute])
        A = L[attribute]
        N.feature_index = A
        
        
        if self.X_col_type[A] == 0:
            L.remove(A)
        
        if partition_val == None:
            #Creating subtrees
            V = set(row[A] for row in self.X_train)
            for v in V:
                Dv = [idx for idx in D if self.X_train[idx][A] == v]
                if(len(Dv)==0):
                    class_counts = Counter(y)
                    majority_class = class_counts.most_common(1)[0][0]
                    if majority_class == None:
                        print("Error: No majority class found")
                    Tv = TreeNode(majority_class=majority_class) 
                else:
                    #print("building subtree for ", v)
                    Tv = self._build_tree(Dv, L, depth+1)
                    
                N.add_sub_tree(Tv, v)
            return N
        
        else:
            #Creating subtrees
            Dv1 = [idx for idx in D if self.X_train[idx][A] <= partition_val]
            Dv2 = [idx for idx in D if self.X_train[idx][A] > partition_val]
            for i, Dv in enumerate([Dv1,Dv2]):
                #print(Dv)
                #Dv = [idx for idx in D if self.X_train[idx][A] == v]
                if(len(Dv)==0):
                    class_counts = Counter(y)
                    majority_class = class_counts.most_common(1)[0][0]
                    if majority_class == None:
                        print("Error: No majority class found")
                    Tv = TreeNode(majority_class=majority_class) 
                else:
                    #print("building subtree for ", i, ":", partition_val)
                    
                    Tv = self._build_tree(Dv, L, depth+1)
                if i==0:  
                    N.add_sub_tree(Tv, ("left", partition_val))
                else:
                    N.add_sub_tree(Tv, ("right", partition_val))
            return N
    
    def _partition_with_num_value(self, feature_values, y, partition_val):
        partitions = {}
        orig_impurity = self.impurity_calc.calculate_impurity_list(y)
        unique_classes = set(y)
        partitions['left'] = {cls: 0 for cls in unique_classes}
        partitions['right'] = {cls: 0 for cls in unique_classes}
        for idx, feature_value in enumerate(feature_values):
            if feature_value <= partition_val:
                partitions['left'][y[idx]] +=1
            else:
                partitions['right'][y[idx]] +=1
        
        avg_impurity = self.impurity_calc.calculate_impurity_partition(partitions)
        information_gain = orig_impurity - avg_impurity
        if self.criterion == "information_gain":
            return information_gain, partitions
        elif self.criterion == 'gini_impurity':
            return avg_impurity, partitions
        
        
    def _partition_numeric_data(self, X, y, feature_idx):
        partitions = {}
        unique_classes = set(y)
        feature_values = [x[feature_idx] for x in X]
        sorted_feature_values = sorted(feature_values)
        partition_values_to_check = [(sorted_feature_values[i] + sorted_feature_values[i + 1])/2 for i in range(len(feature_values)-1)]
        impurities = []
        for partition_val in partition_values_to_check:
            impurity, _ = self._partition_with_num_value(feature_values, y, partition_val)
            impurities.append(impurity)
            
        if self.criterion == "information_gain":
            best_idx = impurities.index(max(impurities))
        elif self.criterion == 'gini_impurity':
            best_idx = impurities.index(min(impurities))
            
        # best_partition = partition_values_to_check[best_idx]
        # impurity, partition = self._partition_with_num_value(feature_values, y, best_partition)
        # print(best_partition)
        # print(partition)
        #print(impurity)
        return partition_values_to_check[best_idx], impurities[best_idx]
        
    def _partition_data(self, X, y, feature_idx):
        partitions = {}
        unique_classes = set(y)
        feature_values = [x[feature_idx] for x in X]
        
        for idx, feature_value in enumerate(feature_values):
            if feature_value not in partitions:
                partitions[feature_value] = {cls: 0 for cls in unique_classes}
            partitions[feature_value][y[idx]] += 1
        
        return partitions
        
        
    def _best_attribute(self, X, y, L):
        #print("in best attribute:")
        #print(L)
        # if len(L) == 1:
        #     return None, L[0]
        orig_impurity = self.impurity_calc.calculate_impurity_list(y)
        # print(f"original impurity: {orig_impurity}")
        impurities = []
        partition_values = []
        for feature_idx in L:
            if self.X_col_type[feature_idx] == 0:
                partitions = self._partition_data(X, y, feature_idx)
                avg_impurity = self.impurity_calc.calculate_impurity_partition(partitions)
                if self.criterion == 'information_gain':
                    information_gain = orig_impurity - avg_impurity
                    impurities.append(information_gain)
                    partition_values.append(None)
                elif self.criterion == 'gini_impurity':
                    impurities.append(avg_impurity)
                    partition_values.append(None)
            else:
                partition_val, avg_impurity = self._partition_numeric_data(X, y, feature_idx)
                impurities.append(avg_impurity)
                partition_values.append(partition_val)
            
            
        
        if self.criterion == "information_gain":
            best_idx = impurities.index(max(impurities))
        elif self.criterion == 'gini_impurity':
            best_idx = impurities.index(min(impurities))
          
        # print("partition_values", partition_values)  
        #print("impurities", impurities)
        # print("best partition:", partition_values[best_idx])
        # print("L:", L)
        #(impurities)
        if self.X_col_type[L[best_idx]] == 0:
            return None, best_idx
        else:
            return partition_values[best_idx], best_idx
        
    
    
# import math
# from collections import Counter
# import pandas as pd
# data = pd.read_csv(r"./datasets/hw3_house_votes_84.csv")
# from sklearn.utils import shuffle
# #data = shuffle(data)
# X=data.iloc[:,0:16].values
# y=data.iloc[:,-1].values
# X_col_type = [0 for _ in range(16)]
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# DTC = DecisionTreeClassifier(criterion="gini_impurity")
# DTC.fit(X, y, X_col_type)
# #print(DTC._best_attribute(X, y, [0,1,2] ))
# #DTC._partition_numeric_data(X, y, 2)
# print("----------------")
#print(DTC.predict(X))
# DTC = DecisionTreeClassifier(criterion="information_gain")
# DTC.fit(X, y, X_col_type)


        
        
        
        