
from queue import SimpleQueue
from tree_node import TreeNode
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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
    def __init__(self, weight_matrix, neural_net):
        self.neural_backbone = Model(inputs=neural_net.input, 
                                     outputs=neural_net.layers[len(neural_net.layers)-2].output)
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

        backbone_out = self.neural_backbone(x)
        curr = self.tree
        while curr.is_leaf() == False:
            right_sim = curr.right.predict(backbone_out)
            left_sim = curr.left.predict(backbone_out)
            if right_sim > left_sim:
                curr = curr.right
            elif right_sim < left_sim:
                curr = curr.left
            else:
                curr = curr.right

        print(curr.class_idx)
        return curr.class_idx



    def soft_inf(self, x):
        self.__soft_inference(x, 1.0, self.tree)
        prediction = []
        self.tree.get_leaf_probs(self.tree, prediction)
        
        try:
            assert sum(prediction) <= 1.0
        except AssertionError:

            print('Softmax does not sum to 1: ', sum(prediction))
            return

        return tf.convert_to_tensor([prediction])



    def __soft_inference(self, x, parent_prob, node):
        if node.is_leaf():
            return
        
        softmax_dist = node.soft_predict(x)

        # set path probabilities of each child
        node.right.path_prob = softmax_dist[1] * parent_prob
        node.left.path_prob = softmax_dist[0] * parent_prob

        # recursive call on subtrees
        self.__soft_inference(x, node.right.path_prob, node=node.right)
        self.__soft_inference(x, node.left.path_prob, node=node.left)        



    """
    Private function to build decision tree in weight space.

    Arguments:
    weights (numpy array): Last FC layer weights of neural net.
    """
    # TODO consider an odd number of leaves (maybe not)
    def __build_tree(self, weights):
        
        q = SimpleQueue()
        w = weights.T
        for i in range(0, len(w)):
            q.put( TreeNode(weights=w[i], 
                            right_child=None, 
                            left_child=None,
                            class_idx=i, 
                            ) )

        node1 = None
        node2 = None

        print('TREE CONSTRUCTION')
        while q.qsize() != 1:
            node1 = q.get()
            node2 = q.get()
            parent_weights = (node1.weight + node2.weight) / 2.0
            parent_node = TreeNode(weights=parent_weights, 
                                   right_child=node1, 
                                   left_child=node2,
                                   class_idx=None,
                                   )

            q.put(parent_node)

        root_node = q.get()
        root_node.right = node1
        root_node.left = node2
        
        return root_node





# TEST
model_inputs = Input(shape=(10,))
z = Dense(20, activation='relu')(model_inputs)
z = Dense(20, activation='relu')(z)
out = Dense(10, activation='softmax', name='output')(z)
model = Model(inputs=model_inputs, outputs=out)

#print(binary_model.get_weights())
#print('WEIGHTS: ', binary_model.get_weights()[2])
x_tree = np.random.rand(1, 20)
x_net = np.random.rand(1,10)
print(model.layers)
print('TREE PREDICTION SOFTMAX')
print(model.get_weights()[4].T.shape)
print(EmbeddedTree(model.get_weights()[4], model).soft_inf(x_tree))
print('NET PREDICTION')
#print('NET PREDICTION')
print(model(x_net))
