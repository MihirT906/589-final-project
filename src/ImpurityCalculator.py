from collections import Counter
import math

class ImpurityCalcualtor:
    def calculate_impurity_list(self, y):
        pass
    
    def calculate_impurity_partition(self, partitions):
        pass
    
class InformationGainCalculator(ImpurityCalcualtor):
    def calculate_impurity_list(self, y):
        class_counts = Counter(y)
        total_samples = len(y)
        probabilities = [count / total_samples for count in class_counts.values()]
        
        entropy = 0
        for probability in probabilities:
            entropy -= probability * math.log2(probability)
        return entropy
    
    def calculate_impurity_partition(self, partitions):
        entropies = []
        weights = []
        for _, dict in partitions.items():
            dict_size = sum(dict.values())
            weights.append(dict_size)
            entropy = 0
            for count in dict.values():
                if dict_size == 0:
                    probability = 0
                else:
                    probability = count / dict_size
                if probability != 0:
                    entropy -= probability * math.log2(probability)
            entropies.append(entropy)
        

        weighted_sum = sum(weight * entropy for weight, entropy in zip(weights, entropies))
        total_weight = sum(weights)
        return weighted_sum / total_weight
    

class GiniImpurityCalculator(ImpurityCalcualtor):
    def calculate_impurity_list(self, y):
        class_counts = Counter(y)
        total_samples = len(y)
        probabilities = [count / total_samples for count in class_counts.values()]
        #print(probabilities)
        gini_impurity = 1
        for probability in probabilities:
            gini_impurity -= probability**2
        return gini_impurity
    
    def calculate_impurity_partition(self, partitions):
        gini_impurities = []
        weights = []
        for _, dict in partitions.items():
            dict_size = sum(dict.values())
            weights.append(dict_size)
            gini_impurity = 1
            for count in dict.values():
                if dict_size == 0:
                    probability = 0
                else:
                    probability = count / dict_size
                if probability != 0:
                    gini_impurity -= probability**2
            gini_impurities.append(gini_impurity)
        

        weighted_sum = sum(weight * entropy for weight, entropy in zip(weights, gini_impurities))
        total_weight = sum(weights)
        return weighted_sum / total_weight
        
    