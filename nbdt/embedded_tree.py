
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
class EmbeddedTree:

    """
    Constructor builds the decision tree in weight space.

    Arguments:
    weight_matrix (numpy array): This is the weights of the final FC layer in the model.
    """
    def __init__(self, weight_matrix, neural_net):
        self.neural_backbone = Model(inputs=neural_net.input, 
                                     outputs=neural_net.layers[len(neural_net.layers)-2].output)
        print(self.neural_backbone.summary())
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

        return curr.class_idx


    # TODO comments
    def soft_inf(self, x):
        self.__soft_inference(x, 1.0, self.tree)
        prediction = []
        self.tree.get_leaf_probs(self.tree, prediction)
        #print(prediction)
        prediction = sorted(prediction, key=lambda x: x[0])
        prediction = [x[1] for x in prediction]
        #print('PREDICTION: ', prediction)
        #print(sum(prediction))
        return tf.convert_to_tensor(prediction, dtype=tf.float32)


    # TODO comments
    def __soft_inference(self, x, parent_prob, node):
        if node.is_leaf():
            return
        
        softmax_dist = node.soft_predict(x)

        # set path probabilities of each child
        node.right.path_prob = softmax_dist[1] * parent_prob
        node.left.path_prob = softmax_dist[0] * parent_prob
        #print(node.weight)

        # recursive call on subtrees
        self.__soft_inference(x, node.right.path_prob, node=node.right)
        self.__soft_inference(x, node.left.path_prob, node=node.left)        



    """
    Function for agglomeritive clustering from leaf nodes
    """
    def __cluster(self, nodes):
        clusters = [] + nodes

        while len(clusters) > 1:
            #print(len(clusters))
            best_sim = clusters[0].similarity(clusters[1])
            node1 = clusters[0]
            node2 = clusters[1]
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    if i == j: continue
                    sim_i_j = clusters[i].similarity(clusters[j])
                    #print(sim_i_j)
                    if sim_i_j < best_sim:
                        #print('UPDATING BEST SIMILARITY')
                        best_sim = sim_i_j
                        node1 = clusters[i]
                        node2 = clusters[j]
            
            parent_weights = (node1.weight + node2.weight) / 2.0
            parent_node = TreeNode(weights=parent_weights,
                                   right_child=node1,
                                   left_child=node2,
                                   class_idx=None,
                                   )
            clusters.append(parent_node)
            clusters.remove(node1)
            clusters.remove(node2)

        return clusters[0]



    """
    Private function to build decision tree in weight space.

    Arguments:
    weights (numpy array): Last FC layer weights of neural net.
    """
    def __build_tree(self, weights):
        
        leaves = []
        w = weights
        for i in range(0, w.shape[1]):
            leaves.append(TreeNode(weights=w[:, i], 
                          right_child=None, 
                          left_child=None,
                          class_idx=i, 
                        ) )

        tree = self.__cluster(leaves)
        return tree




# TEST
"""
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


root = EmbeddedTree(model.get_weights()[4], model).tree

leaves = []
root.get_leaves(root, leaves)
print(len(leaves))
"""