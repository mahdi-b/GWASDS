
from queue import SimpleQueue
from tree_node import TreeNode
import numpy as np


"""
The embedded decision tree in a NBDT. This is the tree that the featurized
data point is fed through to provide explanations for a data point.
"""
class EmbeddedTree():

    """
    Constructor builds the decision tree in weight space.

    Arguments:
    weight_matrix (numpy array): This is the weights of the final FC layer in the model.
    """
    def __init__(self, weight_matrix):
        self.tree = self.__build_tree(weight_matrix)
    

    """
    Make a prediction for x.

    Arguments:
    x (numpy array): This is the featurized data point obtained by cutting out the last
    layer of the neural net and feeding it our data point of interest.
    """
    def inference(self, x):
        # should start from the root of the tree and traverse until a leaf node is reached
        # at each node get the dot product of the input with the children weights
        # go to the child node that has the highest innter product
        # do this until a leaf is reached

        curr = self.tree
        print('INFERENCE ON ', x)
        while curr.is_leaf() == False:
            right_sim = curr.right.predict(x)
            left_sim = curr.left.predict(x)
            print('RIGHT: ', right_sim)
            print('LEFT: ', left_sim)
            if right_sim > left_sim:
                print('RIGHT > LEFT')
                curr = curr.right
            elif right_sim < left_sim:
                print('RIGHT < LEFT')
                curr = curr.left
            else:
                curr = curr.right

        return curr
        # need to return leaf node class?



    """
    Private function to build decision tree in weight space.

    Arguments:
    weights (numpy array): Last FC layer weights of neural net.
    """
    # TODO consider an odd number of leaves
    def __build_tree(self, weights):
        
        q = SimpleQueue()
        w = weights.T

        for i in range(0, len(w)):
            q.put( TreeNode(weights=w[i].T, right_child=None, left_child=None) )

        node1 = None
        node2 = None

        while q.qsize() != 1:
            leaves = []
            node1 = q.get()
            node2 = q.get()
            parent_node = TreeNode(weights=None, right_child=node1, left_child=node2)

            # Get weights in leaves of above parent node
            print('COMPUTING REP VECTOR FROM LEAVES OF NEW NODE...')
            parent_node.get_leaves(parent_node, leaves)
            rep_vector = np.array(leaves)
            print('CHILDREN VECTORS: ', rep_vector.shape)
            parent_node.weight = np.mean(np.array(leaves), axis=0)
            parent_node.weight = parent_node.weight.reshape(parent_node.weight.shape[0], 1)
            print('NEW REP VECTOR: ', parent_node.weight.shape)
            q.put(parent_node)

        root_node = q.get()
        root_node.right = node1
        root_node.left = node2
        
        return root_node



# TESTS
#tree = EmbeddedTree(np.random.rand(20,2))
#tree.inference(np.random.rand(1, 20))


