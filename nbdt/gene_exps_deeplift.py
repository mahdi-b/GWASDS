import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from captum.attr import DeepLift, DeepLiftShap
from neural_interaction_detection import get_interactions
import pandas as pd
import sys


class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.input_layer = torch.nn.Linear(input_size, 10000)
        self.layer1 = torch.nn.Linear(10000, 5000)
        self.layer2 = torch.nn.Linear(5000, 2000)
        self.layer3 = torch.nn.Linear(2000, 500)

    def forward(self, x):
        z = F.relu(self.input_layer(x))
        z = F.relu(self.layer1(z))
        z = F.relu(self.layer2(z))
        z = self.layer3(z)
        return z


class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.layer1 = torch.nn.Linear(500, 2000)
        self.layer2 = torch.nn.Linear(2000, 5000)
        self.layer3 = torch.nn.Linear(5000, 10000)
        self.output_layer = torch.nn.Linear(10000, output_size)

    def forward(self, x):
        z = F.relu(self.layer1(x))
        z = F.relu(self.layer2(z))
        z = F.relu(self.layer3(z))
        z = self.output_layer(z)
        return z


class SDAE(torch.nn.Module):
    def __init__(self, input_size):
        super(SDAE, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, encoder):
        super(Net, self).__init__()
        self.encoder = encoder
        self.classify_layer = torch.nn.Linear(500, output_size)
    
    def forward(self, x):
        z = self.encoder(x)
        logits = self.classify_layer(z)
        #z = F.softmax(logits, dim=1)
        return logits


def nid(model, gene_ids, class_name, MAX_INTERS):
    print('COMPUTING INTERACTIONS FOR ', class_name)
    learned_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            learned_weights.append(param.cpu().detach().numpy())

    class_inters = get_interactions(learned_weights, pairwise=True)[:MAX_INTERS]
    gene1 = []
    gene2 = []
    inter_strengths = []

    print(class_name + ' INTERACTIONS SAVING...')
    for i in range(len(class_inters)):
        inter, strength = class_inters[i]
        try:
            gene1.append(gene_ids[inter[0]].split('.')[0])
        except:
            gene1.append(gene_ids[inter[0]])
        try:
            gene2.append(gene_ids[inter[1]].split('.')[0])
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

torch.manual_seed(int(sys.argv[1]))

data = pd.read_csv('filtered_cancers_gene_exps.csv')
gene_ids = data.columns[2:-1]
print(len(gene_ids))
del data

print('================================================')
print()
dl_contrib_threshold = float(sys.argv[2])
print('Running with deeplift threshold ', dl_contrib_threshold)


x_tr = np.load('gene_exps_train.npy')
y_tr = np.load('cancers_train.npy')
x_val = np.load('gene_exps_val.npy')
y_val = np.load('cancers_val.npy')
x_te = np.load('gene_exps_test.npy')
y_te = np.load('cancers_test.npy')

mu, sigma = 0, 0.1
n_tr = np.random.normal(mu, sigma, size=x_tr.shape)
noisy_x_tr = x_tr + n_tr

noisy_x_tr = torch.Tensor(noisy_x_tr)


#print(sum(x_tr).shape)
#deeplift_baseline = torch.Tensor(sum(x_tr) / x_tr.shape[0]).T

x_tr = torch.Tensor(x_tr)
x_val = torch.Tensor(x_val)
x_te = torch.Tensor(x_te)

y_tr = torch.Tensor(np.argmax(y_tr, axis=1))
y_val = torch.Tensor(np.argmax(y_val, axis=1))
y_te = torch.Tensor(np.argmax(y_te, axis=1))

deeplift_baseline = torch.mean(x_tr, 0)
print(deeplift_baseline.size)
#n_val = np.random.normal(mu, sigma, size=x_val.shape)
#noisy_x_val = x_val + n_val

ae_tr_data = TensorDataset(noisy_x_tr, x_tr)
ae_val_data = TensorDataset(x_val, x_val)

clf_tr_data = TensorDataset(x_tr, y_tr)
clf_val_data = TensorDataset(x_val, y_val)

te_data = TensorDataset(x_te, y_te)

ae_tr_loader = DataLoader(ae_tr_data, batch_size=32, shuffle=True)
ae_val_loader = DataLoader(ae_val_data, batch_size=32, shuffle=True)

clf_tr_loader = DataLoader(clf_tr_data, batch_size=16, shuffle=True)
clf_val_loader = DataLoader(clf_val_data, batch_size=16, shuffle=True)

te_loader = DataLoader(te_data, batch_size=32, shuffle=True)

# Define parameters
LR = 1e-5
EPOCHS = 10

# Define SDAE and 
sdae = SDAE(x_tr.shape[1])
#sdae.to(torch.device("cuda:0"))

# Train SDAE
sdae_criterion = torch.nn.MSELoss() #.cuda()
optimizer = torch.optim.SGD(sdae.parameters(), lr=LR)

print("< ============ Training SDAE ============ >")
print("<=========================================>")

# Train SDAE with noisy gene expressions
for epoch in range(EPOCHS):
    
    tr_epoch_loss = 0
    val_epoch_loss = 0

    for i, (inputs, targets) in enumerate(ae_tr_loader, 0):
        x = inputs #.cuda()
        y = targets #.cuda()

        ae_construction = sdae(x)
        optimizer.zero_grad()
        loss = sdae_criterion(ae_construction, y)
        tr_epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    for i, (inputs, targets) in enumerate(ae_val_loader, 0):
        x = inputs #.cuda()
        y = targets #.cuda()

        ae_construction = sdae(x)
        loss = sdae_criterion(ae_construction, y)
        val_epoch_loss += loss.item()

    TR_LOSS = tr_epoch_loss / len(ae_tr_loader)
    VAL_LOSS = val_epoch_loss / len(ae_val_loader)

    print('---------------------------')
    print("EPOCH: {} \
           TRAIN_LOSS: {} \
           VAL LOSS: {} ".format(epoch+1,
                                 TR_LOSS,
                                 VAL_LOSS,
                                ))
    print('---------------------------')


# Transfer SDAE to classifier
print("< =============== Training Classifier ============== >")
print("<====================================================>")
model = Net(x_tr.shape[1], 5, sdae.encoder)

model.to(torch.device("cuda:0"))
clf_criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    
    val_loss = 0
    tr_loss = 0
    tr_epoch_loss = 0
    val_epoch_loss = 0
    tr_epoch_acc = 0
    val_epoch_acc = 0
    total_tr = 0
    total_val = 0

    for i, (inputs, targets) in enumerate(clf_tr_loader, 0):
        x = inputs.cuda()
        y = targets.type(torch.int64).cuda()

        logits = model(x)
        preds = F.softmax(logits, dim=1)
        optimizer.zero_grad()
        tr_loss = clf_criterion(logits, y)
        tr_epoch_loss += tr_loss.item()

        tr_epoch_acc += (torch.argmax(preds, dim=1) == y).sum().item()
        total_tr += y.size(0)

        tr_loss.backward()
        optimizer.step()

    for i, (inputs, targets) in enumerate(clf_val_loader, 0):
        x = inputs.cuda()
        y = targets.type(torch.int64).cuda()
        total_val += y.size(0)

        logits = model(x)
        preds = F.softmax(logits, dim=1)
        val_epoch_acc += (torch.argmax(preds, dim=1) == y).sum().item()

        val_loss = clf_criterion(logits, y)
        val_epoch_loss += val_loss.item()

    TR_LOSS = tr_epoch_loss / total_tr
    VAL_LOSS = val_epoch_loss / total_val

    print('---------------------------')
    print("EPOCH: {} \
           TRAIN_LOSS: {} \
           VAL LOSS: {} \
           TRAIN ACC {} \
           VAL ACC {} ".format(epoch+1,
                               TR_LOSS,
                               VAL_LOSS,
                               tr_epoch_acc / total_tr,
                               val_epoch_acc / total_val,
                             ))
    print('---------------------------')

"""
print("====== > Getting aggregated weights < ========")
learned_weights = []
for name, param in model.named_parameters():
    if 'weight' in name:
        learned_weights.append(param.cpu().detach().numpy())


agg_w = learned_weights[0].T
for k in range(1, len(learned_weights)):
  agg_w = np.dot(agg_w, learned_weights[k].T)

print(agg_w.shape)
np.save('agg_cancer_model_weights_sdae.npy', agg_w)
"""

print("< =============== Run DeepLIFT to measure gene contributions ====================== >")

dl_loader = DataLoader(x_tr, batch_size=x_tr.shape[0])
model.to(torch.device("cpu"))
deeplift = DeepLift(model)
deeplift_shap = DeepLiftShap(model)
deeplift_baseline = deeplift_baseline.resize(1, deeplift_baseline.shape[0])

attribution_brca = 0
attribution_ucec = 0
attribution_kirc = 0
attribution_luad = 0
attribution_skcm = 0

total = 0

for i, inputs in enumerate(dl_loader, 0):
    attribution_brca = torch.abs(deeplift.attribute(inputs, target=0, baselines=deeplift_baseline))
    attribution_ucec = torch.abs(deeplift.attribute(inputs, target=1, baselines=deeplift_baseline))
    attribution_kirc = torch.abs(deeplift.attribute(inputs, target=2, baselines=deeplift_baseline))
    attribution_luad = torch.abs(deeplift.attribute(inputs, target=3, baselines=deeplift_baseline))
    attribution_skcm = torch.abs(deeplift.attribute(inputs, target=4, baselines=deeplift_baseline))

total = (attribution_brca + attribution_ucec + attribution_kirc + attribution_luad + attribution_skcm)

mean_df = pd.DataFrame(columns=['GENE', 'SCORE'])
mean_df.GENE = gene_ids
mean_df.SCORE = torch.mean(total, 0).detach().numpy()

df = pd.DataFrame(columns=['GENE', 'BRCA', 'UCEC', 'KIRC', 'LUAD', 'SKCM'])
df.GENE = gene_ids
df.BRCA = torch.mean(attribution_brca, 0).detach().numpy()
df.UCEC = torch.mean(attribution_ucec, 0).detach().numpy()
df.KIRC = torch.mean(attribution_kirc, 0).detach().numpy()
df.LUAD = torch.mean(attribution_luad, 0).detach().numpy()
df.SKCM = torch.mean(attribution_skcm, 0).detach().numpy()

with open('df_mean_gene_contribs_' + str(dl_contrib_threshold) + '.txt', 'w') as f:
    for gene in mean_df[mean_df.SCORE >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

with open('dl_brca_genes.txt', 'w') as f:
    for gene in df[df.BRCA >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

with open('dl_ucec_genes.txt', 'w') as f:
    for gene in df[df.UCEC >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

with open('dl_kirc_genes.txt', 'w') as f:
    for gene in df[df.KIRC >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

with open('dl_luad_genes.txt', 'w') as f:
    for gene in df[df.LUAD >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

with open('dl_skcm_genes.txt', 'w') as f:
    for gene in df[df.SKCM >= dl_contrib_threshold].GENE:
        f.write('%s\n' % gene.split('.')[0])

mean_df.to_csv('mean_abs_dl_gene_contribs_' + str(dl_contrib_threshold) + '.csv')
df.to_csv('abs_deeplift_gene_exps.csv')

"""
print('< ===================== Running NID on model ============================ >')
print()

nid(model, gene_ids, 'all', 5000)
"""