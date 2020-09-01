
# file for running tests for NBDT class

import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import AUC
from model import NBDT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering


# TODO check heirarchy construction
# TODO speed up training/inference (subclass Layer for tree)
def cifar_test():
    size = 50000
    print('RUNNING CIFAR TEST')
    print('BUILDING MODEL')
    inputs = Input(shape=(32, 32, 3))
    z = layers.Conv2D(32, (3, 3), activation='relu', dtype=tf.float64)(inputs)
    z = layers.Conv2D(32, (3, 3), activation='relu')(z)
    z = layers.Conv2D(32, (3, 3), activation='relu')(z)
    z = layers.Conv2D(16, (3, 3), activation='relu')(z)

    z = layers.Flatten()(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(32, activation='relu')(z)
    out = layers.Dense(10, activation='softmax')(z)
    model = Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

    print('GETTING DATA')
    (train_im, train_labels), (test_im, test_labels) = datasets.cifar10.load_data()
    train_im = train_im / 255.0
    test_im = test_im / 255.0
    train_labels = np.eye(10)[train_labels].reshape(50000, 10)
    test_labels = np.eye(10)[test_labels].reshape(10000, 10)
    print('PRETRAIN NETWORK')

    #model.fit(train_im, train_labels, batch_size=64, epochs=50, 
     #               validation_data=(test_im, test_labels))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_im, train_labels))
    print('BUILD DATASET OBJECT')
    test_dataset = tf.data.Dataset.from_tensor_slices((test_im, test_labels))
    print('TRAIN NBDT')
    nbdt = NBDT(model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    nbdt.train_network(dataset=train_dataset.take(size).batch(1),
                       loss_function=loss_fn, 
                       epochs=10,
                       tree_loss_weight=1.0, 
                       opt=tf.keras.optimizers.Adam(learning_rate=0.00001),
                       size=size,
                      )
    
    nbdt.evaluate(test_dataset.take(1000).batch(1), size=100)
    

def mnist_test():
    size = 500
    print('RUNNING MNIST TEST')
    print('BUILDING MODEL')
    inputs = Input(shape=(28, 28, 1))
    #z = layers.Conv2D(64, (3, 3), activation='relu', dtype=tf.float64)(inputs)
    z = layers.Conv2D(32, (3,3), activation='relu', dtype=tf.float64)(inputs)
    z = layers.Conv2D(16, (3, 3), activation='relu')(z)
    z = layers.Flatten()(z)
    z = layers.Dense(32, activation='relu')(z)
    out = layers.Dense(10, activation='softmax')(z)
    model = Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

    print('GETTING DATA')
    (train_im, train_labels), (test_im, test_labels) = datasets.mnist.load_data()
    train_im = train_im / 255.0
    test_im = test_im / 255.0
    train_labels = np.eye(10)[train_labels].reshape(60000, 10)
    test_labels = np.eye(10)[test_labels].reshape(10000, 10)
    print('PRETRAIN NETWORK')
    model.fit(train_im, train_labels, batch_size=128, epochs=3, 
                    validation_data=(test_im, test_labels))

    print('BUILD DATASET OBJECT')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_im, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_im, test_labels))
    print('TRAIN NBDT')
    nbdt = NBDT(model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    nbdt.train_network(dataset=train_dataset.take(size).batch(1),
                       loss_function=loss_fn, 
                       epochs=50,
                       tree_loss_weight=1.0, 
                       opt=tf.keras.optimizers.Adam(learning_rate=1e-7),
                       size=size,
                      )
    
    nbdt.evaluate(test_dataset.take(100).batch(1), size=100)


def simulated_snp_test():
    #snps = pd.read_pickle('FILTERED_SIM_0.012.pkl')
    snps = pd.read_csv('fake_test_data.csv')
    #print(snps.columns)
    #print(snps)
    #snps['Type'] = 50000 * ['CASE'] + 50000 * ['CONTROL']

    X = snps[snps.columns[1:-1]].values
    y = pd.get_dummies(snps['Type']).values
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    inputs = Input(shape=(x_tr.shape[1],))
    h = layers.Dense(20, activation='relu')(inputs)
    #h = layers.BatchNormalization()(h)
    h = layers.Dense(20, activation='relu')(h)
    #h = layers.BatchNormalization()(h)
    output = layers.Dense(2, activation='softmax')(h)
    model = Model(inputs=inputs, outputs=output)

    opt = RMSprop(learning_rate=0.00001)
    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['acc', AUC()],
             )

    model.fit(x_tr, y_tr, batch_size=512, epochs=10, validation_split=0.2)
    loss, acc, auc = model.evaluate(x_te, y_te)
    print('TEST LOSS: ', loss)
    print('TEST ACC: ', acc)
    print('TEST AUC: ', auc)


    nbdt = NBDT(model)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    x_te = x_te.reshape(x_te.shape[0], 1, x_te.shape[1])
    y_te = y_te.reshape(y_te.shape[0], 1, y_te.shape[1])
    dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te))
    nbdt.train_network(dataset=dataset,
                       loss_function=loss_fn, 
                       epochs=10,
                       tree_loss_weight=1.0, 
                       opt=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                       size=x_te.shape[0],
                      )



def simulated_snp_test_multi(filename):
    snps = pd.read_csv(filename, dtype='uint8')

    #print(snps.columns)
    X = snps[snps.columns[1:-1]].values
    y = pd.get_dummies(snps['Type']).values    
    print(X)
    print(y)
    print(X.shape)
    print(y.shape)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    print()
    """
    print('TRAINING RANDOM FOREST')
    print('---------------------------------------------------------')
    clf = RandomForestClassifier(n_estimators=200,
                                warm_start=True,
                                oob_score=True,
                                )
    
    clf.fit(x_tr, y_tr)
    
    print('RF TEST ACC: ', clf.score(x_te, y_te))
    print()
    """
    print()
    print('TRAINING NEURAL NET...')
    print('---------------------------------------------------------')
    inputs = Input(shape=(x_tr.shape[1],))
    h = layers.Dense(1000, activation='relu')(inputs)
    h = layers.Dense(1500, activation='relu')(h)
    #h = layers.Dense(2000, activation='relu')(h)
    h = layers.Dropout(0.5)(h)
    #h = layers.Dense(1500, activation='relu')(h)
    h = layers.Dense(750, activation='relu')(h)
    h = layers.Dense(250, activation='relu')(h)
    h = layers.Dropout(0.5)(h)
    h = layers.Dense(100, activation='relu')(h)
    h = layers.Dense(50, activation='relu')(h)
    output = layers.Dense(y_tr.shape[1], activation='softmax')(h)
    model = Model(inputs=inputs, outputs=output)

    opt = RMSprop(learning_rate=0.00001)
    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['acc', AUC()],
             )

    model.fit(x_tr, y_tr, batch_size=128, epochs=100, validation_split=0.3)
    loss, acc, auc = model.evaluate(x_te, y_te)
    print('TEST LOSS: ', loss)
    print('TEST ACC: ', acc)
    print('TEST AUC: ', auc)
    print()
    print()
    print('NEURAL-BACKED DECISION TREE')
    print('--------------------------------------------------------')
    print('BUILDING NBDT')
    nbdt = NBDT(model)
    print()
    print('TRAINING')

    test_dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te))
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    size = 10000
    
    nbdt.train_network(dataset=dataset.take(size).batch(1),
                       loss_function=loss_fn, 
                       epochs=1,
                       tree_loss_weight=2.0, 
                       opt=tf.keras.optimizers.RMSprop(learning_rate=1e-6),
                       size=size,
                      )
    """
    nbdt.evaluate(test_dataset.batch(1), size=x_te.shape[0])


def clustering_test(filename):
    print('reading in data')
    data = pd.read_csv('noisy_snp_multi_30.csv')
    data = data[data.columns[1:]]
    X = data.drop(['Type'], axis=1).values
    print('build cluster model')
    agglo_cl = AgglomerativeClustering(n_clusters=5, compute_full_tree=False)
    print('fitting')
    agglo_cl.fit(X)
    print()
    print('done')
    print(agglo_cl.labels_)    


# ---------------------- TESTS -----------------------
#simulated_snp_test()
#simulated_snp_test_multi('noisy_snp_multi_30.csv')
#cifar_test()
#mnist_test()
# 

# CLUSTER TEST
print('reading in data')
data = pd.read_csv('noisy_snp_multi_30.csv')
data = data[data.columns[1:]]
X = data.drop(['Type'], axis=1).values
print('build cluster model')
agglo_cl = AgglomerativeClustering(n_clusters=5, compute_full_tree=False)
print('fitting')
agglo_cl.fit(X)
print()
print('done')
print(agglo_cl.labels_)