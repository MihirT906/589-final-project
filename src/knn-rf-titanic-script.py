# %%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ### Load data

# %%
data = pd.read_csv(r"../data/titanic.csv")
data

# %%
data.drop(columns=['Name'], inplace=True)
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
data

# %%
data.shape

# %%
X=data.iloc[:,1:7].values
y=data.iloc[:,0].values

# %%
from StratKFold import StratKFold
sk = StratKFold(n_splits=10)
sk.add_data(X,y)
folds = sk.get_folds()

# %%
X_col_type = np.ones(X[0].shape)
X_col_type[0] = 0
X_col_type[1] = 0
X_col_type

# %% [markdown]
# ### Random Forest

# %%
from RandomForestClassifier import RandomForestClassifier 
# rfc = RandomForestClassifier(5)

# %%
from multiple_utils import compute_confusion_matrix, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1score
accuracies = []
f1_scores = []
n_tree_values = [1,5,10,20,30,40,50]
for ntrees in n_tree_values:
    rfc = RandomForestClassifier(criterion="gini_impurity",ntree=ntrees, maximal_depth=15)
    mean_accuracy = 0
    mean_f1_score = 0

    for fold in folds:
        train_indices = fold[0]
        test_indices = fold[1]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        rfc.fit(X_train, y_train, X_col_type)
        y_pred = rfc.predict(X_test)
        confusion_mat, n = compute_confusion_matrix(y_test, y_pred,[0,1])
        mean_accuracy += calculate_accuracy(confusion_mat, n) 
        mean_f1_score += calculate_f1score(calculate_precision(confusion_mat), calculate_recall(confusion_mat))


    mean_accuracy /= len(folds)
    mean_f1_score /= len(folds)
    print("accuracy: ", mean_accuracy)
    print("f1 score: ", mean_f1_score)
    accuracies.append(mean_accuracy)
    f1_scores.append(mean_f1_score)


# %%
plt.figure(figsize=(6, 4))
plt.plot(n_tree_values, accuracies, label="Accuracy")
plt.plot(n_tree_values, f1_scores, label="F1 score")
plt.xlabel('Number of trees')
plt.ylabel('Accuracy/F1 score')
plt.title('Trend of Accuracy/F1 score over number of trees')
plt.legend()
plt.show()

# %%
max_index = accuracies.index(max(accuracies))
print("Optimal accuracy reached at number of trees: ", n_tree_values[max_index])
max_index = f1_scores.index(max(f1_scores))
print("Optimal F1 score reached at number of trees: ", n_tree_values[max_index])

# %%
from multiple_utils import compute_confusion_matrix, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1score
accuracies = []
f1_scores = []
max_depths = [10,20,30,40,50]
for max_depth in max_depths:
    rfc = RandomForestClassifier(criterion="gini_impurity",ntree=50, maximal_depth=max_depth)
    mean_accuracy = 0
    mean_f1_score = 0

    for fold in folds:
        train_indices = fold[0]
        test_indices = fold[1]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        rfc.fit(X_train, y_train, X_col_type)
        y_pred = rfc.predict(X_test)
        confusion_mat, n = compute_confusion_matrix(y_test, y_pred,[0,1])
        mean_accuracy += calculate_accuracy(confusion_mat, n) 
        mean_f1_score += calculate_f1score(calculate_precision(confusion_mat), calculate_recall(confusion_mat))


    mean_accuracy /= len(folds)
    mean_f1_score /= len(folds)
    print("accuracy: ", mean_accuracy)
    print("f1 score: ", mean_f1_score)
    accuracies.append(mean_accuracy)
    f1_scores.append(mean_f1_score)


# %%
plt.figure(figsize=(6, 4))
plt.plot(max_depths, accuracies, label="Accuracy")
plt.plot(max_depths, f1_scores, label="F1 score")
plt.xlabel('max depth')
plt.ylabel('Accuracy/F1 score')
plt.title('Trend of Accuracy/F1 score over max depth')
plt.legend()
plt.show()

# %% [markdown]
# ### KNN

# %%
from KNNClassifier import KNN_Classifier

# %%
from multiple_utils import compute_confusion_matrix, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1score
from utils import Normalizer
scaler = Normalizer()

accuracies = []
f1_scores = []
n_neighbors = [1,5,10,20,30,40,50]
#n_neighbors = [1]
for n_neighbor in n_neighbors:
    knn = KNN_Classifier(n_neighbors=n_neighbor) 
    mean_accuracy = 0
    mean_f1_score = 0

    for fold in folds:
        train_indices = fold[0]
        test_indices = fold[1]
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_train = scaler.fit_transform(X_train)
        X_test = X[test_indices]
        y_test = y[test_indices]
        X_test = scaler.transform(X_test)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        confusion_mat, n = compute_confusion_matrix(y_test, y_pred,[0,1])
        mean_accuracy += calculate_accuracy(confusion_mat, n) 
        mean_f1_score += calculate_f1score(calculate_precision(confusion_mat), calculate_recall(confusion_mat))

    mean_accuracy /= len(folds)
    mean_f1_score /= len(folds)
    print("accuracy: ", mean_accuracy)
    print("f1 score: ", mean_f1_score)
    accuracies.append(mean_accuracy)
    f1_scores.append(mean_f1_score)


# %%
plt.figure(figsize=(6, 4))
plt.plot(n_neighbors, accuracies, label="Accuracy")
plt.plot(n_neighbors, f1_scores, label="F1 score")
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy/F1 score')
plt.title('Trend of Accuracy/F1 score over number of neighbors')
plt.legend()
plt.show()

# %%
max_index = accuracies.index(max(accuracies))
print("Optimal accuracy reached at number of neighbors: ", n_neighbors[max_index])
max_index = f1_scores.index(max(f1_scores))
print("Optimal F1 score reached at number of neighbors: ", n_neighbors[max_index])


