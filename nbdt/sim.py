import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.utils import Progbar

NU_SNPS = 30
NU_NOISE = 10000
NU_CLASS = 5
NU_PER_CLASS = 10000
snp_dist = np.random.dirichlet(np.ones(NU_CLASS),size=NU_SNPS)

# Simulation
sim = None
filled = False
progress = Progbar(target=snp_dist.shape[0])

print('RUNNING SIMULATION FOR SNPS')
for i in range(snp_dist.shape[0]):
    #progress.update(i)
    snps = np.random.multinomial(2, snp_dist[i], size=NU_PER_CLASS)
    full = snps[:, 0]
    
    for j in range(1, snps.shape[1]):
        full = np.concatenate([full, snps[:, j]])
    if filled == False:
        sim = full.reshape(full.shape[0], 1)
        filled = True
    else:
        sim = np.concatenate([sim, full.reshape(full.shape[0], 1)], axis=1)

    progress.update(i)

print('-------------')
print('SIMULATING NOISY SNPS')
noisy_snps = np.round(np.random.uniform(0, 2, size=(50000, NU_NOISE)))
noisy_sim = np.concatenate((sim, noisy_snps), axis=1)
print('-------------')
cols = ['SNP_' + str(i) for i in range(NU_SNPS)] + ['NOISE_' + str(i) for i in range(NU_NOISE)]
noisy_pd_sim = pd.DataFrame(noisy_sim, columns=cols)
cols = list(noisy_pd_sim.columns)
random.shuffle(cols)
noisy_pd_sim = noisy_pd_sim[cols]
noisy_pd_sim['Type'] = ['1']*NU_PER_CLASS + ['2']*NU_PER_CLASS + ['3']*NU_PER_CLASS + ['4']*NU_PER_CLASS + ['5']*NU_PER_CLASS
print(noisy_pd_sim)
noisy_pd_sim.astype('uint8').to_csv('noisy_snp_multi_' + str(NU_SNPS) + '.csv')

print('DONE.')