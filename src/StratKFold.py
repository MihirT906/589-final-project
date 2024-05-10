import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.utils import shuffle

class StratKFold:
    def __init__(self, n_splits=2):
        self.n_splits = n_splits
    
    def add_data(self, X, y):
        self.X = X
        self.y = y
    
    def get_folds(self):
        #split data indices by class
        class_indices = {cls: [] for cls in np.unique(self.y)}
        for i, target_class in enumerate(self.y):
            class_indices[target_class].append(i)
            
        # return class_indices
        
        #print(class_indices)
        #split each of the index lists into n_fold
        fold_indices = {i: [] for i in range(self.n_splits)}
        for cls in class_indices:
            temp_list = class_indices[cls]
            for idx, arr in enumerate(np.array_split(shuffle(temp_list), self.n_splits)):
                #print(arr)
                for j in arr:
                    fold_indices[idx].append(j)
           
        train_test_split = []
        for idx in fold_indices:
            train_test_split.append(self.get_train_test_split(fold_indices,idx))
        
        print(train_test_split)
        return train_test_split

    def get_train_test_split(self, fold_indices, test_fold):
        test_indices = fold_indices[test_fold]
        train_indices = []
        for fold_no, fold_idxs in fold_indices.items():
            if fold_no != test_fold:
                for fi in fold_idxs:
                    train_indices.append(fi)
       
        return train_indices, test_indices
            
        
        
    

# data = pd.read_csv(r"./Data/StatQuestData.csv")
# from sklearn.utils import shuffle
# #data = shuffle(data)
# X=data.iloc[:,0:3].values
# y=data.iloc[:,-1].values
# sk = StratKFold()
# sk.add_data(X,y)
# sk.get_folds()
# list = [0,1,2,3]
# print(shuffle(list))
# skf = StratifiedKFold(n_splits=2)
# data = pd.read_csv(r"./Data/DTC_slide_data.csv")
# from sklearn.utils import shuffle
# data = shuffle(data)
# X=data.iloc[:,0:3].values
# y=data.iloc[:,-1].values

# skf.get_n_splits(X, y)

# print(skf.split(X, y))

# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")
    
