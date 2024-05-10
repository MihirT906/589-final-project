def compute_confusion_matrix(y_true, y_pred):
    # Initialize counts to zero
    TP = TN = FP = FN = 0

    # Calculate TP, TN, FP, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    # Construct the confusion matrix as a dictionary
    confusion_matrix = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

    return confusion_matrix, len(y_true)

# Example usage:
# y_true = [1, 0, 1, 0, 1]
# y_pred = [1, 1, 0, 0, 1]
# confusion_matrix, n = compute_confusion_matrix(y_true, y_pred)
# print(confusion_matrix)

    
def calculate_accuracy(confusion_matrix, n):
    return (confusion_matrix['TP'] + confusion_matrix['TN'])/n

def calculate_precision(confusion_matrix):
    return confusion_matrix['TP']/(confusion_matrix['TP'] + confusion_matrix['FP'])

def calculate_recall(confusion_matrix):
    return confusion_matrix['TP']/(confusion_matrix['TP'] + confusion_matrix['FN'])

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
