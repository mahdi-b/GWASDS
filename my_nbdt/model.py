

import tensorflow as tf
from tensorflow.keras import Sequential
from embedded_tree import EmbeddedTree
from loss import TreeSupLoss
import numpy as np

# TODO implement training procedure with loss defined in paper
# TODO implement backpropgation procedure for training
# TODO Check batching 
# TODO Test on CIFAR10
# TODO Test on SNP data

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
        self.tree = EmbeddedTree(model.layers[-1].weights[0].numpy(), model)




    """
    Use whole network to make prediciton on inputs x
    """
    def network_predictions(self, x):
        return self.model(x)



    """
    Make prediction with full model.

    Arguments:
    x (numpy array): This is the data point from the dataset we want to predict

    Returns
    tree_prediction (tf.Tensor array): The prediction of the NBDT as a softmax distribution.
    """
    def nbdt_predict(self, x):
        x = self.featurize(x)
        tree_prediction = self.tree.soft_inf(x)
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
    Function for computing the gradient of the NBDT with the given loss function (loss_fn)
    """
    def gradient(self, xs, ys, loss_fn):
        with tf.GradientTape() as tape:
            loss_value = loss_fn(ys, self.model(xs), self.nbdt_predict(xs))
        return loss_value, tape.gradient(loss_value, self.backbone.trainable_variables) # TODO this needs to be the backbone, not the original net



    """
    Train the neural network on the dataset.

    The paper defines the loss function to be the Categorical Cross-entropy of the net outputs summed with a weighted 
    Categorical Crossentropy loss on the NBDT outputs.
    TODO docstring for arguments
    """
    def train_network(self, dataset, loss_function, epochs, tree_loss_weight, opt):
        # On every weight update, reconstruct the tree from the last layer weights.
        # Inference has to be done on the original network and the full NBDT
        # The network should be pretrained on the dataset of interest

        # iterate through dataset
            # for each member of the dataset
                # make prediction with model
                # make prediction with NBDT
                # compute the loss
                # update the neural network parameters

        training_loss_results = []
        training_accuracy_results = []
        loss_fn = TreeSupLoss(loss_function, tree_loss_weight)
        
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()

            for x, y in dataset:
                loss, grad = self.gradient(x, y, loss_fn)
                opt.apply_gradients(zip(grad, self.model.trainable_variables))
                epoch_loss_avg.update_state(loss)
                epoch_acc.update_state(y, self.nbdt_predict(x))

            training_loss_results.append(epoch_loss_avg.results())
            training_accuracy_results.append(epoch_acc.results())

            if epoch % 20 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_acc.result()))
        


