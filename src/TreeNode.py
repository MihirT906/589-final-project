class TreeNode:
    def __init__(self, feature_index=None, majority_class = None, isLeaf = False):
        self.feature_index = feature_index
        self.majority_class = majority_class
        self.children = []
        self.edges = []
       # self.isLeaf = isLeaf
    
    def add_sub_tree(self, childNode, edgeLabel):
        self.children.append(childNode)
        self.edges.append(edgeLabel)
        