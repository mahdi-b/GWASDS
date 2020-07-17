

from tensorflow.keras import Sequential
from embedded_tree import EmbeddedTree


"""
Neural-Backed Decision Tree from Alvin et al. 2020

This model is a hybrid approach that combines neural networks and decision trees
to make a more interpretable model while using the power of deep learning.
"""
class NBDT():

    """
    Init Neural-Backed Decision Trees. Takes a neural network model and initiates
    the decision tree with the last FC layer weights.
    """
    def __init__(self, model):

        # this is the full neural network
        self.model = model

        # self.neural_backbone = ... cut out last layer
        self.backbone = Sequential(model.layers[:-1])

        # build tree from weights
        # TODO might need to consider attribute length
        self.tree = EmbeddedTree(model.layers[-1].weights[0].numpy())



    """
    Make prediction with full model.

    Arguments:
    x (numpy array): This is the data point from the dataset we want to predict

    Returns
    tree_prediction (numpy array): The prediction of the NBDT.
    """
    def predict(self, x):
        x = self.featurize(x)
        tree_prediction = self.tree.inference(x)
        return tree_prediction
        


    """
    Featurize a data point by feeding the input through the neural network with the last
    layer removed.

    Arguments:
    data (numpy array): The data point we want to featurize.
    """
    def featurize(self, data):
        return self.backbone(data)



    """
    Train the neural network on the dataset.
    """
    # TODO
    def train_network(self, dataset):
        pass


# TESTS
