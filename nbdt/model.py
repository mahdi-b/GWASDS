

import tensorflow as tf
from tensorflow.keras import Sequential
from embedded_tree import EmbeddedTree
from loss import TreeSupLoss
import numpy as np
import pandas as pd
from loss import TreeSupLoss

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


    def compute_loss(self, x, y):
        return TreeSupLoss(tf.keras.losses.CategoricalCrossentropy(), 1.0)(y, 
                                                                           self.network_predictions(x),
                                                                           self.nbdt_predict(x),
                                                                           )

    """
    Function for computing the gradient of the NBDT with the given loss function (loss_fn)
    """
    def gradient(self, xs, ys, loss_fn):
        with tf.GradientTape() as tape:
            tape.watch(self.backbone.trainable_variables)
            loss_value = self.compute_loss(xs, ys)
            #loss_value = loss_fn(ys, self.network_predictions(xs), self.nbdt_predict(xs))
        return loss_value, tape.gradient(loss_value, [self.backbone.trainable_variables])



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
        tree_loss = TreeSupLoss(loss_function, tree_loss_weight)
        
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()

            for x, y in dataset:
                loss, grad = self.gradient(x, y, tree_loss)
                opt.apply_gradients(zip(grad, self.model.trainable_variables))
                epoch_loss_avg.update_state(loss)
                epoch_acc.update_state(y, self.nbdt_predict(x))

            training_loss_results.append(epoch_loss_avg.result())
            training_accuracy_results.append(epoch_acc.result())

            #if epoch % 20 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_acc.result()))
        


# TEST
# DEFINE MODEL
model_inputs = tf.keras.layers.Input(shape=(10,))
z = tf.keras.layers.Dense(20, activation='relu')(model_inputs)
z = tf.keras.layers.Dense(20, activation='relu')(z)
out = tf.keras.layers.Dense(10, activation='softmax', name='output')(z)
model = tf.keras.Model(inputs=model_inputs, outputs=out)


# DEFINE DATASET
snps = pd.read_csv('fake_test_data.csv')
case = snps[snps.Type == 'CASE'].sample(n=1)
control = snps[snps.Type == 'CONTROL'].sample(n=1)
dataset = pd.concat([case, control], axis=0)
pheno = tf.data.Dataset.from_tensor_slices(pd.get_dummies(dataset.Type).values)
geno = tf.data.Dataset.from_tensor_slices(dataset[dataset.columns[1:-1]].values)

geno_pheno = tf.data.Dataset.zip((geno, pheno)) # test dataset

# DEFINE NBDT
nbdt = NBDT(model)
nbdt.train_network(dataset=geno_pheno.batch(1), 
                   loss_function=tf.keras.losses.CategoricalCrossentropy(), 
                   epochs=10, 
                   tree_loss_weight=1.0, 
                   opt=tf.keras.optimizers.Adam(),
                   )


