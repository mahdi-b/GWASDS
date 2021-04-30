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
from neural_interaction_detection import get_interactions
from pyensembl import EnsemblRelease
import sys
import shap

seed = int(sys.argv[1])
tf.random.set_seed(seed)
gene_db = EnsemblRelease(77)
data = pd.read_csv('cancers_gene_exps.csv')
gene_ids = data.columns[1:-1]
del data

def load_model(arch, LR, X, y, L1_coef, encoder=None):
    print('CREATING NEURAL NET')
    opt = Adam(learning_rate=LR)
    model = None

    if encoder == None:
        inputs = Input(shape=(X.shape[1],))
        h = layers.Dense(arch[0], kernel_regularizer=l1_l2(l1=L1_coef, l2=0.0), activation='relu')(inputs)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(arch[1], activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(arch[2], activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(arch[3], activation='relu')(h)
        h = layers.BatchNormalization()(h)
        h = layers.Dense(arch[4], activation='relu')(h)
        output = layers.Dense(5, activation='softmax')(h)
        model = Model(inputs=inputs, outputs=output)

    else:
        print('Transfer encoder to classifier and fine-tuning...')
        model = Sequential([encoder, layers.Dense(5, activation='softmax')])

    model.compile(optimizer=opt,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['acc', AUC(), Recall(), Precision()],
                )

    return model
    
def load_train_sdae(arch, LR, X_tr, X_val, epochs):

    mu, sigma = 0, 0.1
    n_tr = np.random.normal(mu, sigma, size=X_tr.shape)
    noisy_X_tr = X_tr + n_tr

    n_val = np.random.normal(mu, sigma, size=X_val.shape)
    noisy_X_val = X_val + n_val

    activ = 'relu'
    print('=============== STACKED DENOISING AUTOENCODER ==================')
    # Autoencoder
    encoder_inputs = Input(shape=(X_tr.shape[1],))
    h = layers.Dense(arch[0], activation=activ)(encoder_inputs)
    h = layers.Dropout(0.5)(h)
    h = layers.Dense(arch[1], activation=activ)(h)
    h = layers.Dropout(0.5)(h)
    h = layers.Dense(arch[2], activation=activ)(h)
    #h = layers.BatchNormalization()(h)
    encoder_outs = layers.Dense(arch[3], activation=activ)(h)

    #decoder_inputs = Input(shape=(arch[3],))
    l = layers.Dense(arch[3], activation=activ)(encoder_outs)
    #l = layers.BatchNormalization()(l)
    l = layers.Dense(arch[2], activation=activ)(l)
    l = layers.Dropout(0.5)(l)
    l = layers.Dense(arch[1], activation=activ)(l)
    l = layers.Dropout(0.5)(l)
    l = layers.Dense(arch[0], activation=activ)(l)
    decoder_output = layers.Dense(X_tr.shape[1], activation='linear')(l)
    autoencoder = Model(inputs=encoder_inputs, outputs=decoder_output)
    autoencoder.compile(optimizer=SGD(learning_rate=1e-5), loss='MSE', metrics=['mse'])

    autoencoder.fit(noisy_X_tr, 
                    X_tr, 
                    batch_size=32, 
                    epochs=epochs,
                    #validation_split=0.3, 
                    validation_data=(noisy_X_val, X_val),
                    )
    return Model(inputs=encoder_inputs, outputs=encoder_outs)


def train_model(model, X, y, X_val, y_val, BATCH, EPOCHS):
    model.fit(X, 
          y, 
          batch_size=BATCH, 
          epochs=EPOCHS,
          validation_data=(X_val, y_val), 
          verbose=1,
          )


def load_data(inputs_file, outputs_file):
    X = np.load(inputs_file)
    y = np.load(outputs_file)
    return X, y


def nid(model, gene_ids, class_name, MAX_INTERS):
    print('COMPUTING INTERACTIONS FOR ', class_name)
    learned_weights = []
    for i in range(len(model.weights)):
        if 'kernel' in model.weights[i].name:
            learned_weights.append(model.weights[i].numpy().T)

    class_inters = get_interactions(learned_weights, pairwise=True)[:MAX_INTERS]
    gene1 = []
    gene2 = []
    inter_strengths = []

    print(class_name + ' INTERACTIONS SAVING...')
    for i in range(len(class_inters)):
        inter, strength = class_inters[i]
        try:
            gene1.append(gene_db.gene_by_id(gene_ids[inter[0]].split('.')[0]).gene_name)
        except:
            gene1.append(gene_ids[inter[0]])
        try:
            gene2.append(gene_db.gene_by_id(gene_ids[inter[1]].split('.')[0]).gene_name)
        except:
            gene2.append(gene_ids[inter[1]])

        inter_strengths.append(strength)

    df = pd.DataFrame(columns=['GENE_1', 'GENE_2', 'INTER_STRENGTH'])

    df.GENE_1 = gene1
    df.GENE_2 = gene2
    df.INTER_STRENGTH = inter_strengths

    df.to_csv(class_name + '_interactions.csv')

    del df
    del gene1
    del gene2
    del inter_strengths


    

arch = [4096, 2048, 1024, 512, 128, 64]
arch_sdae = [10000, 5000, 2000, 500]
epochs = 5
batch = 32
LR = 1e-5


"""
X = np.load('norm_gene_exps.npy')
y_brca = np.load('y_brca.npy')
y_kirc = np.load('y_kirc.npy')
y_luad = np.load('y_luad.npy')
y_skcm = np.load('y_skcm.npy')
y_ucec = np.load('y_ucec.npy')
"""

x_tr = np.load('gene_exps_train.npy')
y_tr = np.load('cancers_train.npy')

x_val = np.load('gene_exps_val.npy')
y_val = np.load('cancers_val.npy')

x_te = np.load('gene_exps_test.npy')
y_te = np.load('cancers_test.npy')

#print(x_tr)
#print(x_val)
print("NU SAMPLES: ", x_tr.shape[0] + x_val.shape[0])

encoder = load_train_sdae(arch_sdae, LR=1e-5, X_tr=x_tr, X_val=x_val, epochs=10)

model = load_model(arch, LR=1e-5, X=x_tr, y=y_tr, L1_coef=0.0, encoder=encoder)
train_model(model, x_tr, y_tr, x_val, y_val, BATCH=32, EPOCHS=10)

print("===> MODEL EVAL VAL and TEST<===")
#model.evaluate(x_val, y_val)
model.evaluate(x_te, y_te)
"""
print("===> BUILD NBDT <===")
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
te_dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te))
nbdt = NBDT(model)

print("====== > NBDT EVAL VALIDATION")
nbdt.evaluate(val_dataset.batch(1), size=x_val.shape[0])
nbdt.evaluate(te_dataset.batch(1), size=x_te.shape[0])
"""
print("====== > Getting aggregated weights < ========")
learned_weights = []
#learned_weights_nid = []
learned_weights_rev = []
for i in range(len(model.weights)):
  if 'kernel' in model.weights[i].name:
    learned_weights_rev.append(model.weights[i].numpy())
    #learned_weights_nid.append(model.weights[i].numpy().T)
    learned_weights.append(model.weights[i].numpy())

learned_weights_rev.reverse()

agg_w = learned_weights[0]
agg_w_rev = learned_weights_rev[0].T
for k in range(1, len(learned_weights)):
  agg_w_rev = np.dot(agg_w_rev, learned_weights_rev[k].T)
  agg_w = np.dot(agg_w, learned_weights[k])

print(agg_w.shape)
print(agg_w_rev.shape)
np.save('agg_cancer_model_weights_sdae.npy', agg_w)
np.save('agg_cancer_model_weights_rev_sdae.npy', agg_w_rev.T)

print('< ========= RUNNING NID ========== >')
nid(model, gene_ids, 'all', 1000)



"""
print()
print('=======')
print('===> TRAINING NBDT')

loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
tree_weight = 0.01
NBDT_LR = 1e-5

print('===NBDT TRAINING===')
nbdt.train_network(train_dataset, 
                   val_dataset,
                   test_data_size=x_val.shape[0], 
                   loss_function=loss_fn, 
                   epochs=5, 
                   tree_loss_weight=tree_weight, 
                   opt=tf.keras.optimizers.Adam(learning_rate=NBDT_LR), 
                   size=x_tr.shape[0],
                   )
"""
#nid(model, gene_ids, 'all', 1000)

"""
brca_model = load_model(arch, LR=1e-5, X=X, y=y_brca, L1_coef=0.001)
kirc_model = load_model(arch, LR=1e-5, X=X, y=y_kirc, L1_coef=0.001)
luad_model = load_model(arch, LR=1e-5, X=X, y=y_luad, L1_coef=0.001)
skcm_model = load_model(arch, LR=1e-5, X=X, y=y_skcm, L1_coef=0.001)
ucec_model = load_model(arch, LR=1e-5, X=X, y=y_ucec, L1_coef=0.001)

train_model(brca_model, X, y_brca, BATCH=32, EPOCHS=epochs)
train_model(kirc_model, X, y_kirc, BATCH=32, EPOCHS=epochs)
train_model(luad_model, X, y_luad, BATCH=32, EPOCHS=epochs)
train_model(skcm_model, X, y_skcm, BATCH=32, EPOCHS=epochs)
train_model(ucec_model, X, y_ucec, BATCH=32, EPOCHS=epochs)

nid(brca_model, gene_ids, 'brca', 1000)
nid(kirc_model, gene_ids, 'kirc', 1000)
nid(luad_model, gene_ids, 'luad', 1000)
nid(skcm_model, gene_ids, 'skcm', 1000)
nid(ucec_model, gene_ids, 'ucec', 1000)
"""



