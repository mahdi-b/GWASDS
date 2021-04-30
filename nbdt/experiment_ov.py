# imports
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.metrics import AUC, MSE, Recall, Precision
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from model import NBDT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from neural_interaction_detection import get_interactions
#from pyensembl import EnsemblRelease
from scipy.stats import zscore
from scipy.stats import norm
import sys

# TODO
# Use autoencoder and cluster embeddings of genes, see if they change upon retraining
# Use clusters to see which classes are related
# Create a neural network for each class and train
# Use NID to get interactions and significant genes
# compare these interactions and genes with related classes

seed = int(sys.argv[1])
tf.random.set_seed(seed)

#gene_db = EnsemblRelease(77)
#import wandb

#wandb.init(project='capstone', entity='nimuh', sync_tensorboard=True)

NBDT_LR = 1e-7
LR = 1e-5
BATCH = 256
EPOCHS = 100
VAL_SPLIT = 0.4
arch = [128, 64, 32, 16]
L1_coef = 0.001
Drop_factor = 0.5
tree_weight = 0.1
MAX_INTERS = 5000

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                 factor=0.5,
                                                 patience=5, 
                                                 verbose=1,
                                                 min_delta=0.1,
                                                 )

# Read in data
snps = np.load('full_ov_genotypes.npy')
#print(snps[3][2046])
#vals, counts = np.unique(snps, return_counts=True)
r, c = np.where(snps > 2)
print(np.unique(c, return_counts=True))
snps = np.delete(snps, c, 1)
print(snps.shape)
#print(counts.shape)
snps[snps > 2] = 0
histo = np.load('histotypes.npy')
histo = pd.get_dummies(histo).values

print(snps.shape)
print(histo.shape)

snps_tr, snps_val, y_tr, y_val = train_test_split(snps, histo, test_size=0.4, shuffle=True)
snps_val, snps_te, y_val, y_te = train_test_split(snps_val, y_val, test_size=0.5, shuffle=True)

_, tr_class_dist = np.unique(y_tr, return_counts=True)
_, val_class_dist = np.unique(y_val, return_counts=True)
_, te_class_dist = np.unique(y_te, return_counts=True)


print('TRAIN CLASS DIST: ', tr_class_dist)
print('VAL CLASS DIST:   ', val_class_dist)
print('TEST CLASS DIST:  ', te_class_dist)

cancer_dict = {'brca': 0, 
               'ucec': 1,
               'kirc': 2,
               'luad': 3,
               'skcm': 4,
               }
cancer_dict_inv = {0: 'brca',
                   1: 'ucec',
                   2: 'kirc',
                   3: 'luad',
                   4: 'skcm',
                   }

#cancers = data.CANCER.map(cancer_dict)

print('==== DATA ====')
print()
print('X Features: ', snps.shape[1])
print('Y:          ', histo.shape[1])
print()
print('==== PARAMS ====')
#print('===> SEED: ', seed)
print('===> LR:     ', LR)
print('===> EPOCHS: ', EPOCHS)
print('===> L1:     ', L1_coef)
print('===> BATCH SIZE: ', BATCH)
print('===> GETTING {} INTERACTIONS'.format(MAX_INTERS))
print('=======================')
print()


print('CREATING NEURAL NET')
# Initial model
opt = Adam(learning_rate=LR)
inputs = Input(shape=(snps.shape[1],))
h = layers.Dense(arch[0], kernel_regularizer=l1_l2(l1=L1_coef, l2=0.0), activation='relu')(inputs)
h = layers.BatchNormalization()(h)
h = layers.Dense(arch[1], activation='relu')(h)
h = layers.BatchNormalization()(h)
h = layers.Dense(arch[2], activation='relu')(h)
h = layers.BatchNormalization()(h)
h = layers.Dense(arch[3], activation='relu')(h)
#h = layers.BatchNormalization()(h)
#h = layers.Dense(arch[4], activation='relu')(h)
output = layers.Dense(histo.shape[1], activation='softmax')(h)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc', AUC(), Recall(), Precision()],
              )

model.fit(snps_tr, 
          y_tr, 
          batch_size=BATCH, 
          epochs=EPOCHS,
          validation_data=(snps_val, y_val), 
          verbose=1,
          )
            
print('DONE TRAINING DNN')
print()
print('=== EVAL on TEST SET: ===')
model.evaluate(snps_te, y_te)
#print('Class weights: ', model.layers[-1].weights[0].numpy().shape)
#np.save('class_weights_2', model.layers[-1].weights[0].numpy().T)
#model.evaluate(snps_te, y_te)

# Compute input contributions with respect to each class
#print(model.weights)

"""
learned_weights = []
learned_weights_nid = []
learned_weights_rev = []
for i in range(len(model.weights)):
  if 'kernel' in model.weights[i].name:
    learned_weights_rev.append(model.weights[i].numpy())
    learned_weights_nid.append(model.weights[i].numpy().T)
    learned_weights.append(model.weights[i].numpy())

# Run NID on learned_weights list
# Save pairs
gene_interactions = get_interactions(learned_weights_nid, pairwise=True)[:MAX_INTERS]
gene1 = []
gene2 = []
inter_strengths = []

print('SAVING PAIR INTERACTIONS')
for i in range(len(gene_interactions)):
  inter, strength = gene_interactions[i]
  gene1.append(gene_ids[inter[0]])
  gene2.append(gene_ids[inter[1]])
  #gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  #gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  inter_strengths.append(strength)

df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
df.GENE_1 = gene1
df.GENE_2 = gene2
df.INTER_STRENGTH = inter_strengths
df.to_csv('all_gene_pair_strengths.csv')
del df

learned_weights_rev.reverse()

agg_w = learned_weights[0]
agg_w_rev = learned_weights_rev[0].T
for k in range(1, len(learned_weights)):
  agg_w_rev = np.dot(agg_w_rev, learned_weights_rev[k].T)
  agg_w = np.dot(agg_w, learned_weights[k])

np.save('agg_cancer_model_weights.npy', agg_w)
#np.save('agg_cancer_model_weights_rev.npy', agg_w_rev)

print('<=== MEASURING GENE SIGNIFICANCE ===>')
#agg_w = agg_w.T
print(agg_w.shape)
nu_individuals = x_tr.shape[1]

for c in range(agg_w.shape[1]):
  w = agg_w[:, c]
  p_vals = norm.sf(zscore(w))**2
  idx = np.argsort(p_vals)
  p_vals_sorted = p_vals[idx]
  BH_corrs = []
  raw_pvals = []
  FDR = 0.05
  gene_idx = []

  for i in range(len(p_vals_sorted)):
      rank = i + 1
      #print('RANK ', rank)
      BH_corr = (rank/nu_individuals) * FDR
      if p_vals_sorted[i] >= BH_corr: break
      else: 
          BH_corrs.append(BH_corr)
          gene_idx.append(idx[i])
          raw_pvals.append(p_vals_sorted[i])

  sig_gene_ids = gene_ids[gene_idx]
  sig_gene_names = []
  for i in range(len(sig_gene_ids)):
      try:
        gene_name = gene_db.gene_by_id(sig_gene_ids[i].split('.')[0]).gene_name.split('.')[0]
        sig_gene_names.append(gene_name)
        #sig_gene_names.append(sig_gene_ids[i].split('.')[0])
      except:
        sig_gene_names.append(sig_gene_ids[i].split('.')[0])

  sig_df = pd.DataFrame(columns=['GENE', 'BH_COR_PVAL', 'RAW_PVAL'])
  sig_df.GENE = sig_gene_names
  sig_df.BH_COR_PVAL = BH_corrs
  sig_df.RAW_PVAL = raw_pvals

  print('<=== ' + cancer_dict_inv[c] + ' ===>')
  sig_df.to_csv(cancer_dict_inv[c] + '_sig_genes.csv')


#print(agg_w.shape)
#print(agg_w_rev.shape)
"""
"""
print('PLOTTING TREE...')

plt.figure(figsize=(15,30))
#w = nbdt.get_weights(input_layer=False).T
#d = pdist(agg_w, 'euclidean')
clusters = linkage(agg_w, method='average', metric='euclidean')
dendrogram(clusters)
fname = "dendro_class_contrib_" + sys.argv[1]

#print("NBDT TREE FILE SAVED")
#print(nbdt.tree.get_clusters())
plt.savefig(fname)

"""
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
                   epochs=5, 
                   tree_loss_weight=tree_weight, 
                   opt=tf.keras.optimizers.Adam(learning_rate=NBDT_LR), 
                   size=snps_tr.shape[0],
                   )

print('===NBDT EVALUATE===')
acc, auc = nbdt.evaluate(test_dataset.batch(1), size=snps_val.shape[0])
#acc, auc = nbdt.evaluate(train_dataset.batch(1), size=snps.shape[0])
print('===================')

#print('LAST LAYER WEIGHTS===')
#weights = model.layers[-1].weights[0].numpy().T
#np.save('class_weights_5', weights)
#print(model.layers[-1].weights[0].numpy())
#print(model.layers[-1].weights[0].numpy().shape)


# get cluster indices
#groups = nbdt.tree.get_clusters()
#print(groups)

#np.save('groups_acc-{:.3}_seed-{}'.format(acc, seed), np.array(groups))

"""
g1 = groups[0]
g2 = groups[1]
g = g1 + g2
g1_str = ""
g2_str = ""

print('GROUP 1')
for i in range(len(g1)):
  g1_str += ("_" + cancer_dict_inv[g1[i]])
  print(cancer_dict_inv[g1[i]])

print()

print('GROUP 2')
for i in range(len(g2)):
  g2_str += ("_" + cancer_dict_inv[g2[i]])
  print(cancer_dict_inv[g2[i]])

print()

print('GROUP 1: ')
print(model.layers[-1].weights[0].numpy().T[g1].shape)
print()
print('GROUP 2: ')
print(model.layers[-1].weights[0].numpy().T[g2].shape)

backbone = nbdt.backbone
g1_weights = []
g2_weights = []

c1_weights = []
c2_weights = []
c3_weights = []
c4_weights = []
c5_weights = []

print(backbone.summary())
for i in range(len(backbone.weights)):
  print(backbone.weights[i].name)
  if 'kernel' in backbone.weights[i].name:
    g1_weights.append(backbone.weights[i].numpy().T)
    g2_weights.append(backbone.weights[i].numpy().T)

    c1_weights.append(backbone.weights[i].numpy().T)
    c2_weights.append(backbone.weights[i].numpy().T)
    c3_weights.append(backbone.weights[i].numpy().T)
    c4_weights.append(backbone.weights[i].numpy().T)
    c5_weights.append(backbone.weights[i].numpy().T)

g1_weights.append(model.layers[-1].weights[0].numpy().T[g1])
g2_weights.append(model.layers[-1].weights[0].numpy().T[g2])

c1_weights.append(model.layers[-1].weights[0].numpy().T[g[0]])
c2_weights.append(model.layers[-1].weights[0].numpy().T[g[1]])
c3_weights.append(model.layers[-1].weights[0].numpy().T[g[2]])
c4_weights.append(model.layers[-1].weights[0].numpy().T[g[3]])
c5_weights.append(model.layers[-1].weights[0].numpy().T[g[4]])

print('GETTING INTERACTIONS FOR CLUSTER 1')
g1_interactions = get_interactions(g1_weights, pairwise=True)[:MAX_INTERS]
print('GETTING INTERACTIONS FOR CLUSTER 2')
g2_interactions = get_interactions(g2_weights, pairwise=True)[:MAX_INTERS]

print('GETTING CLASS SPECIFIC INTERACTIONS')
c1_interactions = get_interactions(c1_weights, pairwise=True)[:MAX_INTERS]
c2_interactions = get_interactions(c2_weights, pairwise=True)[:MAX_INTERS]
c3_interactions = get_interactions(c3_weights, pairwise=True)[:MAX_INTERS]
c4_interactions = get_interactions(c4_weights, pairwise=True)[:MAX_INTERS]
c5_interactions = get_interactions(c5_weights, pairwise=True)[:MAX_INTERS]


print('===============')
print()

g1_gene1 = []
g1_gene2 = []
g1_inter = []

g2_gene1 = []
g2_gene2 = []
g2_inter = []

c1_gene1 = []
c1_gene2 = []
c1_inter = []

c2_gene1 = []
c2_gene2 = []
c2_inter = []

c3_gene1 = []
c3_gene2 = []
c3_inter = []

c4_gene1 = []
c4_gene2 = []
c4_inter = []

c5_gene1 = []
c5_gene2 = []
c5_inter = []


print('GROUP 1 INTERACTIONS')
for i in range(len(g1_interactions)):
  inter, strength = g1_interactions[i]
  g1_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  g1_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  g1_inter.append(strength)
  #print(gene_ids[inter[0]], gene_ids[inter[1]], strength)

print('GROUP 2 INTERACTIONS')
for i in range(len(g2_interactions)):
  inter, strength = g2_interactions[i]
  g2_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  g2_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  g2_inter.append(strength)
  #print(inter, strength)

print('CLASS 1 INTERACTIONS')
for i in range(len(c1_interactions)):
  inter, strength = c1_interactions[i]
  c1_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  c1_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  c1_inter.append(strength)
  #print(inter, strength)

print('CLASS 2 INTERACTIONS')
for i in range(len(c2_interactions)):
  inter, strength = c2_interactions[i]
  c2_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  c2_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  c2_inter.append(strength)
  #print(inter, strength)

print('CLASS 3 INTERACTIONS')
for i in range(len(c3_interactions)):
  inter, strength = c3_interactions[i]
  c3_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  c3_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  c3_inter.append(strength)
  #print(inter, strength)

print('CLASS 4 INTERACTIONS')
for i in range(len(c4_interactions)):
  inter, strength = c4_interactions[i]
  c4_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  c4_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  c4_inter.append(strength)
  #print(inter, strength)

print('CLASS 5 INTERACTIONS')
for i in range(len(c5_interactions)):
  inter, strength = c5_interactions[i]
  c5_gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
  c5_gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
  c5_inter.append(strength)
  #print(inter, strength)


g1_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
g2_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])

c1_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
c2_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
c3_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
c4_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])
c5_df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])

g1_df.GENE_1 = g1_gene1
g1_df.GENE_2 = g1_gene2
g1_df.INTER_STRENGTH = g1_inter

g2_df.GENE_1 = g2_gene1
g2_df.GENE_2 = g2_gene2
g2_df.INTER_STRENGTH = g2_inter

c1_df.GENE_1 = c1_gene1
c1_df.GENE_2 = c1_gene2
c1_df.INTER_STRENGTH = c1_inter

c2_df.GENE_1 = c2_gene1
c2_df.GENE_2 = c2_gene2
c2_df.INTER_STRENGTH = c2_inter

c3_df.GENE_1 = c3_gene1
c3_df.GENE_2 = c3_gene2
c3_df.INTER_STRENGTH = c3_inter

c4_df.GENE_1 = c4_gene1
c4_df.GENE_2 = c4_gene2
c4_df.INTER_STRENGTH = c4_inter

c5_df.GENE_1 = c5_gene1
c5_df.GENE_2 = c5_gene2
c5_df.INTER_STRENGTH = c5_inter

g1_df.to_csv('l1_group1_inters' + g1_str + '.csv')
g2_df.to_csv('l1_group2_inters' + g2_str + '.csv')

c1_df.to_csv('l1_' + str(cancer_dict_inv[0]) + '_inter.csv')
c2_df.to_csv('l1_' + str(cancer_dict_inv[1]) + '_inter.csv')
c3_df.to_csv('l1_' + str(cancer_dict_inv[2]) + '_inter.csv')
c4_df.to_csv('l1_' + str(cancer_dict_inv[3]) + '_inter.csv')
c5_df.to_csv('l1_' + str(cancer_dict_inv[4]) + '_inter.csv')
print('PLOTTING TREE...')

plt.figure(figsize=(15,30))
w = nbdt.get_weights(input_layer=False).T
d = pdist(w, 'euclidean')
clusters = linkage(w, method='average', metric='euclidean')
dendrogram(clusters)
fname = "dendro_acc-{:.3}_seed-{}.png".format(acc, seed)

print("NBDT TREE FILE SAVED")
print(nbdt.tree.get_clusters())
plt.savefig(fname)
"""
"""
# Create noisy SNPs for input
mu, sigma = 0, 0.001
#mask_tr = np.random.randint(low=0, high=2, size=snps_tr.shape)
#mask_val = np.random.randint(low=0, high=2, size=snps_val.shape)

n_tr = np.random.normal(mu, sigma, size=X.shape)
#n_val = np.random.normal(mu, sigma, size=X.shape)
noisy_snps_tr = X + n_tr
#noisy_snps_val = snps_val + n_val

#print(noisy_snps_tr)
#print(noisy_snps_tr.shape)

activ = 'relu'
print('=============== STACKED DENOISING AUTOENCODER ==================')
# Autoencoder
encoder_inputs = Input(shape=(X.shape[1],))
#h = layers.Dense(8192, activation='relu')(encoder_inputs)
#h = layers.BatchNormalization()(h)
h = layers.Dense(10000, activation=activ)(encoder_inputs)
h = layers.BatchNormalization()(h)
h = layers.Dense(5000, activation=activ)(h)
h = layers.BatchNormalization()(h)
h = layers.Dense(2000, activation=activ)(h)
h = layers.BatchNormalization()(h)
h = layers.Dense(500, activation=activ)(h)
h = layers.Dense(250, activation=activ)(h)
h = layers.Dense(100, activation=activ)(h)
enocoder_outs = layers.Dense(2, activation=activ)(h)

decoder_inputs = Input(shape=(2,))
l = layers.Dense(100, activation=activ)(decoder_inputs)
l = layers.Dense(250, activation=activ)(l)
l = layers.Dense(500, activation=activ)(l)
l = layers.BatchNormalization()(l)
l = layers.Dense(2000, activation=activ)(l)
l = layers.BatchNormalization()(l)
l = layers.Dense(5000, activation=activ)(l)
l = layers.BatchNormalization()(l)
l = layers.Dense(10000, activation=activ)(l)
#l = layers.BatchNormalization()(l)
decoder_output = layers.Dense(X.shape[1], activation='linear')(l)

encoder = Model(inputs=encoder_inputs, outputs=enocoder_outs)
decoder = Model(inputs=decoder_inputs, outputs=decoder_output)
autoencoder = Sequential([encoder, 
                          decoder])

autoencoder.compile(optimizer=SGD(learning_rate=1e-5), loss='MSE', metrics=['mse'])
autoencoder.fit(noisy_snps_tr, 
                X, 
                batch_size=32, 
                epochs=100,
                validation_split=0.3, 
                #validation_data=(noisy_snps_val, snps_val),
                )

print()
embedding = encoder(X)
np.save('SDAE_encoder_embedding_batchnorm_2', embedding)
"""
"""
print('============ TRANSFERING TO CLASSIFIER ============')
# transfer encoder to classifier
encoder.trainable = False
clf_out = layers.Dense(y_tr.shape[1], activation='softmax')
clf = Sequential([encoder,
                 layers.ReLU(),
                 #layers.BatchNormalization(),
                 clf_out,
                 ])

# train and evaluate
opt = Adam(learning_rate=LR)
clf.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', AUC(), Recall(), Precision()])
print(clf.summary())
clf.fit(snps_tr, 
        y_tr, 
        batch_size=32, 
        epochs=100, 
        validation_data=(snps_val, y_val),
        )

clf.evaluate(snps_te, y_te)

nbdt = NBDT(clf)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

print('===NBDT EVALUATE===')
test_dataset = tf.data.Dataset.from_tensor_slices((snps_te, y_te))
acc, auc = nbdt.evaluate(test_dataset.batch(1), size=snps_te.shape[0])

print('===================')

groups = nbdt.tree.get_clusters()
print(groups)
"""