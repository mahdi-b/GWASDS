

import tensorflow as tf
from tensorflow.keras import Sequential
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
        return TreeSupLoss(tf.keras.losses.CategoricalCrossentropy(), 10.0).call(y, 
                                                                           self.network_predictions(x),
                                                                           self.nbdt_predict(x),
                                                                           )


    """
    Function for computing the gradient of the NBDT with the given loss function (loss_fn)
    """
    def gradient(self, xs, ys, loss_fn):
        with tf.GradientTape() as tape:
            #tape.watch(self.backbone.variables)
            #tape.watch(self.model.variables)
            loss_value = self.compute_loss(xs, ys)
            #print(loss_value)
        return loss_value, tape.gradient(loss_value, [self.backbone.variables, self.model.variables])



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
        tree_loss = TreeSupLoss(loss_function, tree_loss_weight)
        
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            count = 0
            total_samples = 0
            for x, y in dataset:
                total_samples = total_samples + 1
                loss, grad = self.gradient(x, y, tree_loss) 
                opt.apply_gradients(zip(grad[0], self.backbone.variables))
                opt.apply_gradients(zip(grad[1], self.model.variables))
                epoch_loss_avg.update_state(loss)
                pred = np.zeros((2,)).astype('uint8')
                pred[np.argmax(self.nbdt_predict(x))] = 1
                
                if np.argmax(y.numpy()[0]) == np.argmax(pred):
                    count = count + 1

            training_loss_results.append(epoch_loss_avg.result())

            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                count / total_samples))




# TEST
# DEFINE MODEL
model_inputs = tf.keras.layers.Input(shape=(10,))
z = tf.keras.layers.Dense(20, activation='relu')(model_inputs)
z = tf.keras.layers.Dense(20, activation='relu')(z)
out = tf.keras.layers.Dense(2, activation='softmax', name='output')(z)
model = tf.keras.Model(inputs=model_inputs, outputs=out)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

# DEFINE DATASET
snps = pd.read_csv('fake_test_data.csv')
case = snps[snps.Type == 'CASE'].sample(n=100)
control = snps[snps.Type == 'CONTROL'].sample(n=100)
dataset = pd.concat([case, control], axis=0)
dataset = dataset.sample(frac=1.0)
pheno = pd.get_dummies(dataset.Type).values
geno = dataset[dataset.columns[1:-1]].values
geno_pheno = tf.data.Dataset.from_tensor_slices((geno, pheno)) # test dataset
model.fit(geno, pheno, batch_size=20, epochs=100)

# DEFINE NBDT
nbdt = NBDT(model)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
nbdt.train_network(dataset=geno_pheno.batch(1), 
                   loss_function=loss_fn, 
                   epochs=10, 
                   tree_loss_weight=10.0, 
                   opt=tf.keras.optimizers.Adam(learning_rate=0.00001),
                   )


