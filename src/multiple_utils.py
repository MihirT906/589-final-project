def compute_confusion_matrix(y_true, y_pred, class_list):
    confusion_matrix = {true_label: {pred_label: 0 for pred_label in class_list} 
                        for true_label in class_list}
    
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1

    return confusion_matrix, len(y_true)



    
def calculate_accuracy(confusion_matrix, n):
    correct_predictions = sum(confusion_matrix[i][i] for i in confusion_matrix)
    
    accuracy = correct_predictions / n
    
    return accuracy

def calculate_precision(confusion_matrix):
    precision = {}
    
    for i in confusion_matrix:
        true_positives = confusion_matrix[i][i]
        false_positives = sum(confusion_matrix[j][i] for j in confusion_matrix if j != i)
        
        if true_positives + false_positives == 0:
            precision[i] = 0
        else:
            precision[i] = true_positives / (true_positives + false_positives)
    
    total_sum = sum(precision.values())
    
    mean = total_sum / len(precision)
    
    return mean


def calculate_recall(confusion_matrix):
    recall = {}
    
    for i in confusion_matrix:
        true_positives = confusion_matrix[i][i]
        false_negatives = sum(confusion_matrix[i].values()) - true_positives
        
        if true_positives + false_negatives == 0:
            recall[i] = 0
        else:
            recall[i] = true_positives / (true_positives + false_negatives)
    
    mean_recall = sum(recall.values()) / len(recall)
    
    return mean_recall


def calculate_f1score(precision, recall):
    return (2*precision*recall)/(precision + recall)


class Normalizer:
    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        return (X - self.mean) / self.std
    
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("No data has been fit")
        
        return (X - self.mean) / self.std


# confusion_matrix = {0: {0: 266, 1: 1}, 1: {0: 0, 1: 168}}
# print(calculate_precision(confusion_matrix))
# print(calculate_recall(confusion_matrix))
# print(calculate_accuracy(confusion_matrix, n))
# print(calculate_precision(confusion_matrix))
# print(calculate_recall(confusion_matrix))
# print(calculate_f1score(calculate_precision(confusion_matrix), calculate_recall(confusion_matrix)))

# y_true = ['Yes', 'Yes', 'No', 'No']
# y_pred = ['Yes', 'No', 'Yes', 'No']
# print(calculate_accuracy(y_true, y_pred))

# class Normalizer:
#     def fit_transform(self, X):
#         self.mean = X.mean(axis=0)
#         self.std = X.std(axis=0)
        
#         return (X - self.mean) / self.std
    
#     def transform(self, X):
#         if self.mean is None or self.std is None:
#             raise ValueError("No data has been fit")
        
#         return (X - self.mean) / self.std