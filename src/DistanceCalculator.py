import math
class DistanceCalculator:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
    
    def calculate_distance(self):
        pass
    

class EuclideanDistanceCalculator(DistanceCalculator):
    def calculate_distance(self):
        if len(self.point1) != len(self.point2):
            raise ValueError("Points must have the same number of dimensions")

        squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(self.point1, self.point2))
        return round(math.sqrt(squared_distance), 3)

    
    
# if __name__ == "__main__":
#     point1 = (1, 2, 3)
#     point2 = (4, 5, 6)

#     euclidean_calculator = EuclideanDistanceCalculator(point1, point2)
#     distance = euclidean_calculator.calculate_distance()

#     print("Euclidean distance between", point1, "and", point2, "is", distance)
