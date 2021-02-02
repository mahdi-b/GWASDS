
from queue import SimpleQueue
from tree_node import TreeNode
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.cluster import ward_tree
import shap
import sys

sys.setrecursionlimit(1000000)

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
        #print(self.neural_backbone.summary())
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
    def soft_inf(self, x, interpret=False, background=None, sample=None, shaps=None):
        self.__soft_inference(x, 1.0, self.tree, interpret, background, sample, shaps=shaps)
        prediction = []
        self.tree.get_leaf_probs(self.tree, prediction)
        prediction = sorted(prediction, key=lambda x: x[0])
        prediction = [x[1] for x in prediction]
        return tf.convert_to_tensor(prediction, dtype=tf.float32)


    # TODO comments
    def __soft_inference(self, x, parent_prob, node, interpret=False, background=None, sample=None, shaps=None):
        if node.is_leaf():
            return
        
        softmax_dist = node.soft_predict(x)
        #print(softmax_dist)
        path_next = np.argmax(softmax_dist)

        if interpret:
            left_w = node.left.weight.reshape((node.left.weight.shape[0], 1))
            right_w = node.right.weight.reshape((node.right.weight.shape[0], 1))
            weights = np.concatenate((left_w, right_w), axis=1)
            
            # create new layer with one node and set the weights
            node_layer = Dense(2, activation='softmax', use_bias=False, name='node_layer')

            # connect to backbone
            backbone = Sequential(self.neural_backbone.layers)
            backbone.add(node_layer)
            backbone.get_layer('node_layer').set_weights(list(weights.reshape((1, weights.shape[0], weights.shape[1]))))
            #backbone.layers[-1].set_weights(list(weights.reshape((1, weights.shape[0], weights.shape[1]))))
            #print(backbone.summary())
            #print("NODE OUTPUT TEST: ", backbone(sample))
            # run DeepLIFT
            e = shap.DeepExplainer(backbone, background)
            
            shap_values = e.shap_values(sample, check_additivity=True)
            #shap_values = shap_values[path_next]
            #shaps.append(np.absolute(shap_values))
            #print(np.argmax(shaps))
            shaps += (np.sum(np.absolute(shap_values), axis=0) / 2.0)
            #shaps.append(np.sum(shap_values, axis=0) / 2.0) # This could be a memory issue

            #if len(shaps) == 0: shaps = np.sum(shap_values, axis=0) / 2.0
            #else: shaps += (np.sum(shap_values, axis=0) / 2.0)
            #print("TOP SNPs: (MAX: ", np.argmax(abs(shap_values[np.argmax(softmax_dist)])), ")")
            

        # set path probabilities of each child
        node.right.path_prob = softmax_dist[1] * parent_prob
        node.left.path_prob = softmax_dist[0] * parent_prob

        # recursive call on subtrees
        self.__soft_inference(x, node.right.path_prob, node=node.right, interpret=interpret, background=background, sample=sample, shaps=shaps)
        self.__soft_inference(x, node.left.path_prob, node=node.left, interpret=interpret, background=background, sample=sample, shaps=shaps)        


    """
    Function for agglomeritive clustering from leaf nodes
    """
    def __cluster(self, nodes):
        clusters = [] + nodes

        while len(clusters) > 1:

            best_sim = clusters[0].similarity(clusters[1])
            node1 = clusters[0]
            node2 = clusters[1]

            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    sim_i_j = clusters[i].similarity(clusters[j])
                    if sim_i_j < best_sim:
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
        
        # get clusters with ward_tree function
        pairs = ward_tree(weights.T)[0]
        w = weights.T
        n_samples = weights.T.shape[0]
        tree_nodes = {}

        idx = 0
        for pair in pairs:
            w_list = []
            children = []
            for el in pair:
                if el < n_samples:
                    tree_nodes[el] = TreeNode(weights=w[el],
                                              right_child=None,
                                              left_child=None,
                                              class_idx=el,
                                              )
                    w_list.append(w[el])
                else:
                    w_list.append(tree_nodes[el].weight)

                children.append(el)
            
            tree_nodes[idx + n_samples] = TreeNode(weights = (w_list[0] + w_list[1])/2.0,
                                                   right_child=tree_nodes[children[1]],
                                                   left_child=tree_nodes[children[0]],
                                                   class_idx=None,
                                                   )
            idx += 1
        return tree_nodes[idx + n_samples - 1]


    # TODO COMMENTS
    def get_clusters(self):
        c1 = []
        c2 = []
        self.tree.get_leaves_clusters(self.tree.left, c1)
        self.tree.get_leaves_clusters(self.tree.right, c2)
        
        cluster_1 = [node.class_idx for node in c1]
        cluster_2 = [node.class_idx for node in c2]
        return cluster_1, cluster_2



        

