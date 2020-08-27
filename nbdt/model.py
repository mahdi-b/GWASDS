

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Progbar
from embedded_tree import EmbeddedTree
from loss import TreeSupLoss
import numpy as np
import pandas as pd
from loss import TreeSupLoss

# TODO Check batching 
# TODO Test on CIFAR10
# TODO Test on SNP data

"""
Neural-Backed Decision Tree from Alvin et al. 2020

This model is a hybrid approach that combines neural networks and decision trees
to make a more interpretable model while using the power of deep learning.
"""
class NBDT:

    """
    Init Neural-Backed Decision Trees. Takes a neural network model and initiates
    the decision tree with the last FC layer weights.
    """
    def __init__(self, model):

        # this is the full neural network
        #print('NET')
        self.model = model
        #print(self.model.summary())
        # self.neural_backbone = ... cut out last layer
        self.backbone = Sequential(model.layers[:-1])
        #print('BACKBONE')
        #print(self.backbone.summary())

        # build tree from weights
        # TODO might need to consider attribute length
        #print(model.layers[-1].weights[0])
        self.tree = EmbeddedTree(model.layers[-1].weights[0].numpy(), model)




    """
    Use whole network to make prediciton on inputs x
    """
    def network_predictions(self, x):
        #if len(x.shape) == 3:
         #   return self.model(x.numpy().reshape(1, *x.shape))
        pred = self.model(x)
        #print('NET PREDICTION: ', pred)
        return pred



    """
    Make prediction with full model.

    Arguments:
    x (numpy array): This is the data point from the dataset we want to predict

    Returns
    tree_prediction (tf.Tensor array): The prediction of the NBDT as a softmax distribution.
    """
    def nbdt_predict(self, x):
        featurized_sample = self.featurize(x)
        tree_prediction = self.tree.soft_inf(featurized_sample)
        #print('TREE PRED: ', tree_prediction)
        return tree_prediction
        

    """
    Featurize a data point by feeding the input through the neural network with the last
    layer removed.

    Arguments:
    data (numpy array): The data point we want to featurize.
    """
    def featurize(self, data):
        #if len(data.shape) == 3:
         #   return self.backbone(data) #.reshape(1, *data.shape))
        return self.backbone(data)


    """
    TODO comments
    """
    def compute_loss(self, x, y, weight):
        return TreeSupLoss(tf.keras.losses.CategoricalCrossentropy(), weight).call(y, 
                                                                           self.network_predictions(x),
                                                                           self.nbdt_predict(x),
                                                                           )


    """
    Function for computing the gradient of the NBDT with the given loss function (loss_fn)
    """
    def gradient(self, xs, ys, loss_fn, weight):
        with tf.GradientTape() as tape:
            loss_value = self.compute_loss(xs, ys, weight)
        return loss_value, tape.gradient(loss_value, [self.backbone.variables, self.model.variables])



    """
    Train the neural network on the dataset.

    The paper defines the loss function to be the Categorical Cross-entropy of the net outputs summed with a weighted 
    Categorical Crossentropy loss on the NBDT outputs.
    TODO docstring for arguments
    TODO include number of classes
    """
    def train_network(self, dataset, loss_function, epochs, tree_loss_weight, opt, size):
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
        tree_loss = TreeSupLoss(loss_function, tree_loss_weight)
        
        for epoch in range(epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_acc_avg = tf.keras.metrics.CategoricalAccuracy()

            i = 0
            progress = Progbar(target=size)
            for x, y in dataset:
                i = i + 1
                #progress.update(i)
                sample = x 
                loss, grad = self.gradient(sample, y, tree_loss, tree_loss_weight)
                opt.apply_gradients(zip(grad[1], self.model.variables))
                epoch_loss_avg.update_state(loss)
                nbdt_pred = self.nbdt_predict(sample)
                epoch_acc_avg.update_state(y, nbdt_pred)
                progress.update(i, values=[('loss:', epoch_loss_avg.result()), ('acc:', epoch_acc_avg.result())])
               
            training_loss_results.append(epoch_loss_avg.result().numpy())
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_acc_avg.result()),
                                                                )

        return training_loss_results



    def evaluate(self, dataset, size):
        count = 0
        total_samples = 0
        progress = Progbar(target=size)
        acc_avg = tf.keras.metrics.CategoricalAccuracy()

        for x, y in dataset:
            #progress.update(total_samples)
            #sample = x 
            total_samples = total_samples + 1
            acc_avg.update_state(y, self.nbdt_predict(x))
            progress.update(total_samples, values=[('acc: ', acc_avg.result())])
            #pred = np.zeros((y.shape[1],)).astype('uint8')
            #pred[np.argmax(self.nbdt_predict(sample))] = 1 
            #print('PRED: ' + str(np.argmax(pred)) + ' ------ ' + ' TRUE: ' + str(np.argmax(y)))
            #if np.argmax(y) == np.argmax(pred):
             #   count = count + 1
            
            
            #training_loss_results.append(epoch_loss_avg.result().numpy())
        print("Accuracy: {:.3%}".format(acc_avg.result()))
                                                                



# TODO test on simulated SNP data
# TODO test on multi class SNP data (cancer)
# TODO test on CIFAR10 Dataset