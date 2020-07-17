
"""
This class represents a node in the embedded decision tree for a Neural-Backed Decision Tree.
Each node is able to make a prediction given an input and its weight vector.

The NBDT will have an Embedded Decision Tree made out of TreeNode objects.
"""
from numpy import dot

class TreeNode():
    
    """
    TreeNode constructor. Each TreeNode should have a vector of weights along with a left and right child.

    Arguments:
    weights (numpy array): a 1-D vector obtained from the last layer of the model. Each node has a vector of weights
    to be able to make predictions.
    right_child (TreeNode): The right child of given TreeNode.
    left_child (TreeNode): The left child of given TreeNode
    """
    def __init__(self, weights, right_child, left_child):
        self.weight = weights
        self.right = right_child
        self.left = left_child
        


    """
    Get all leaves of a sub tree. This is used to constuct the decision tree
    in weight-space.

    Arguments:
    node (TreeNode): root node of the tree we want the leaves of.
    leaves (list): List of TreeNodes. These are the leaves found.

    """
    def get_leaves(self, node, leaves):
        if node.is_leaf():
            leaves.append(node.weight)
            return
        
        self.get_leaves(node.right, leaves)
        self.get_leaves(node.left, leaves)
    


    """
    Checks if given TreeNode is a leaf.
    """
    def is_leaf(self):
        if self.right is None and self.left is None:
            return True
        return False



    """
    Use weights of node to make a prediction given x. The prediction is 
    the inner product between x and the node weights.

    Arguments:
    x (numpy array): Featurized sample. This is the output of the nueral network when the 
    last layer is removed.

    Returns:
    Inner product between featurized input and node weights.
    """
    def predict(self, x):
        # make sure x and weights dimensions are compatible
        # should return dot product of x and nodes weights
        return dot(x, self.weight)
        
