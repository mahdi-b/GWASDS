# imports
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import AUC, MSE
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from model import NBDT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#FILE = 'deep_gwas_sim_8k_5c.csv'
#FILE = 'deep_gwas_sim_3c.csv'
LR = 1e-5
EPOCHS = 100
VAL_SPLIT = 0.3
arch = [64, 128]
L1_coef = 0.1

# Read in data
#data = pd.read_csv(FILE)
#snps = data[data.columns[1:-1]].values[:, :30]
#print(snps.shape)
#y = pd.get_dummies(data.CLASS).values
snps_tr = np.load('cancer_train_inputs.npy')
y_tr = np.load('cancer_train_outputs.npy')
snps_te = np.load('cancer_test_inputs.npy')
y_te = np.load('cancer_test_outputs.npy')

snps_te, snps_val, y_te, y_val = train_test_split(snps_te, y_te, test_size=0.5, shuffle=True)

print(snps_tr.shape, y_tr.shape)
#print(snps_te.shape, )
print('CREATING NEURAL NET')
# Initial model
opt = Adam(learning_rate=LR)
inputs = Input(shape=(snps_tr.shape[1],))
h = layers.Dense(arch[0], kernel_regularizer=l1_l2(l1=L1_coef, l2=0), activation='relu')(inputs)
h = layers.BatchNormalization()(h)
h = layers.Dense(arch[1], activation='relu')(h)
h = layers.BatchNormalization()(h)
output = layers.Dense(y_tr.shape[1], activation='softmax')(h)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['acc', AUC()],
              )

model.fit(snps_tr, 
            y_tr, 
            batch_size=16, 
            epochs=EPOCHS, 
            verbose=1, 
            validation_data=(snps_val, y_val),
            )
print('DONE TRAINING')
print()
model.evaluate(snps_te, y_te)

print('====')
print('BUILDING NBDT')

# NBDT
train_dataset = tf.data.Dataset.from_tensor_slices((snps_tr, y_tr))
val_dataset = tf.data.Dataset.from_tensor_slices((snps_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((snps_te, y_te))

nbdt = NBDT(model)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

print('===NBDT TRAINING===')
nbdt.train_network(train_dataset, 
                   val_dataset,
                   test_data_size=snps_val.shape[0], 
                   loss_function=loss_fn, 
                   epochs=10, 
                   tree_loss_weight=5.0, 
                   opt=tf.keras.optimizers.Adam(learning_rate=1e-6), 
                   size=y_tr.shape[0],
                   )
print('===NBDT EVALUATE===')
acc, auc = nbdt.evaluate(test_dataset.batch(1), size=snps_te.shape[0])
print('===================')

"""
print('RE-TRAIN classifier with adjusted backbone of NBDT...')
nbdt.backbone.trainable = True
new_model = Sequential([nbdt.backbone,
                        layers.Dense(y_tr.shape[1], activation='softmax'),
                        ])

opt = Adam(learning_rate=LR)
new_model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['acc', AUC()],
              )
print('==== DNN TRAIN ===')
new_model.fit(snps_tr, 
              y_tr, 
              batch_size=16, 
              epochs=EPOCHS, 
              verbose=1, 
              validation_data=(snps_val, y_val),
            )

new_model.evaluate(snps_te, y_te)
#print('PLOTTING TREE...')

#plt.figure(figsize=(15,30))
#w = nbdt.get_weights(input_layer=True).T
#d = pdist(w, 'euclidean')
#clusters = linkage(w, method='ward', metric='euclidean')
#dendrogram(clusters)
#fname = "dendro_{}c_{}_l1-{:.3}.png".format(y_tr.shape[1], arch[0], L1_coef)
#print("NBDT TREE FILE SAVED")
#print(nbdt.tree.get_clusters())
#plt.savefig(fname)

"""
"""
print('=============== AUTOENCODER ==================')
# Autoencoder
encoder_inputs = Input(shape=(snps_tr.shape[1],))
h = layers.Dense(256, activation='relu')(encoder_inputs)
h = layers.BatchNormalization()(h)
enocoder_outs = layers.Dense(128, activation='relu')(h)

decoder_inputs = Input(shape=(128,))
l = layers.Dense(128, activation='relu')(decoder_inputs)
l = layers.BatchNormalization()(l)
l = layers.Dense(256, activation='relu')(l)
l = layers.BatchNormalization()(l)
decoder_output = layers.Dense(snps_tr.shape[1], activation='linear')(l)

encoder = Model(inputs=encoder_inputs, outputs=enocoder_outs)
decoder = Model(inputs=decoder_inputs, outputs=decoder_output)
autoencoder = Sequential([encoder, 
                          decoder])

autoencoder.compile(optimizer=Adam(learning_rate=LR), loss='MSE', metrics=['mse'])
autoencoder.fit(snps_tr, 
                snps_tr, 
                batch_size=16, 
                epochs=10, 
                validation_data=(snps_val, snps_val),
                )


ae_nbdt = NBDT(decoder)
plt.figure(figsize=(15,50))
w = ae_nbdt.get_weights(input_layer=False)
clusters = linkage(w, method='ward', metric='euclidean')
dendrogram(clusters)
fname = "ae_decoder_dendro_{}c_{}_l1-{:.3}.png".format(y_tr.shape[1], arch[0], L1_coef)
print("NBDT TREE FILE SAVED")
plt.savefig(fname)



print()
print('============ TRANSFERING TO CLASSIFIER ============')
# transfer encoder to classifier
encoder.trainable = False
clf_out = layers.Dense(y_tr.shape[1], activation='softmax')
clf = Sequential([encoder,
                 layers.BatchNormalization(),
                 clf_out,
                 ])

# train and evaluate
opt = Adam(learning_rate=1e-6)
clf.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', AUC()])
clf.fit(snps_tr, y_tr, batch_size=16, epochs=100, validation_data=(snps_val, y_val))
clf.evaluate(snps_te, y_te)
"""

