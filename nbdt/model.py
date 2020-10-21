

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
        self.model = model

        # self.neural_backbone = ... cut out last layer
        self.backbone = Sequential(self.model.layers[:-1])

        # build tree from last layer weights
        self.tree = EmbeddedTree(model.layers[-1].weights[0].numpy(), model)



    def get_clusters(self):
        return self.tree.get_clusters()

    def get_weights(self):
        return self.model.layers[-1].weights[0].numpy()

    """
    Use whole network to make prediciton on inputs x
    """
    def network_predictions(self, x):
        pred = self.model(x)
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
    def compute_loss(self, x, y, weight, loss_fn):
        return TreeSupLoss(loss_fn, weight).call(y, 
                                                 self.network_predictions(x),
                                                 self.nbdt_predict(x),
                                                )


    """
    Function for computing the gradient of the NBDT with the given loss function (loss_fn)
    """
    def gradient(self, xs, ys, loss_fn, weight):
        with tf.GradientTape() as tape:
            nbdt_loss, net_loss, loss_value = self.compute_loss(xs, ys, weight, loss_fn)
        return nbdt_loss, net_loss, loss_value, tape.gradient(loss_value, [#self.backbone.variables, 
                                                                          self.model.trainable_variables,
                                                                          ],
                                                             )



    """
    Train the neural network on the dataset.

    The paper defines the loss function to be the Categorical Cross-entropy of the net outputs summed with a weighted 
    Categorical Crossentropy loss on the NBDT outputs.
    TODO docstring for arguments
    TODO include number of classes
    """
    def train_network(self, dataset, test_data, test_data_size, loss_function, epochs, tree_loss_weight, opt, size):
        # Inference has to be done on the original network and the full NBDT
        # The network should be pretrained on the dataset of interest

        # iterate through dataset
            # for each member of the dataset
                # make prediction with model
                # make prediction with NBDT
                # compute the loss
                # update the neural network parameters

        training_loss_results = []
        #tree_loss = TreeSupLoss(loss_function, tree_loss_weight)
        self.model.layers[-1].trainable = False
        for epoch in range(epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_nbdt_loss_avg = tf.keras.metrics.Mean()
            epoch_net_loss_avg = tf.keras.metrics.Mean()
            epoch_acc_avg = tf.keras.metrics.CategoricalAccuracy()

            i = 0
            progress = Progbar(target=size)
            for x, y in dataset.take(size).batch(1):
                i = i + 1
                sample = x 
                nbdt_loss, net_loss, loss, grad = self.gradient(sample, y, loss_function, tree_loss_weight)
                opt.apply_gradients(zip(grad[0], self.model.trainable_variables))
                self.backbone = Sequential(self.model.layers[:-1])
                epoch_loss_avg.update_state(loss)
                epoch_nbdt_loss_avg.update_state(nbdt_loss)
                epoch_net_loss_avg.update_state(net_loss)
                nbdt_pred = self.nbdt_predict(sample)
                epoch_acc_avg.update_state(y, nbdt_pred)
                progress.update(i, values=[('nbdt_loss:', epoch_nbdt_loss_avg.result()),
                                            ('net_loss:', epoch_net_loss_avg.result()),
                                            ('loss:', epoch_loss_avg.result()),
                                            ('acc:', epoch_acc_avg.result()),
                                        ])
               
            training_loss_results.append(epoch_loss_avg.result().numpy())
            print()
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_acc_avg.result()),
                                                                )

            if epoch_acc_avg.result() >= .90:
                test_acc = self.evaluate(test_data.batch(1), size=test_data_size)
                if test_acc >= .90:
                    print("SAVING MODEL")
                    self.model.save("nn_nbdt_test_acc-{:.3f}_epoch-{:03d}_adam".format(test_acc, epoch))

            print()
            self.model.save("nn_nbdt_epoch-{:03d}_adam".format(epoch))

        return training_loss_results



    def evaluate(self, dataset, size):
        count = 0
        total_samples = 0
        progress = Progbar(target=size)
        acc_avg = tf.keras.metrics.CategoricalAccuracy()

        for x, y in dataset:
            total_samples = total_samples + 1
            acc_avg.update_state(y, self.nbdt_predict(x))
            progress.update(total_samples, values=[('acc: ', acc_avg.result())])
        print("Accuracy: {:.3%}".format(acc_avg.result()))
        return acc_avg.result()
                                                                
