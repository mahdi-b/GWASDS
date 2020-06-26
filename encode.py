import pandas as pd
import numpy as np
import allel
import pyensembl


"""
Utility function to read GWAS summary file into dataframe.
"""
def read_file(filename):
    summary = open(filename, 'r').read()
    rows = summary.split('\n')
    cols = rows[0].split('\t')
    snps = rows[1:]
    df = pd.DataFrame([rows.split('\t') for rows in snps], columns=cols)
    return df[df.columns[:7]].iloc[:df.shape[0]-1]



# Code for generating gene information from SNP IDs
#########################################################################
"""
Helper function for get gene info. This formats the gene information for SNP ID into
a row to be added to a dataframe.
"""
def gene_info(snp, genes):
    if len(genes) > 0:
        rows = np.array([[snp,
                          gene.gene_id,
                          gene.gene_name,
                          gene.biotype,
                          gene.contig,
                          gene.start,
                          gene.end,
                          gene.strand] for gene in genes])
        return rows


"""
get gene information for set of SNPs
Args: snps -> SNP IDs to get gene info for
Returns: genes_df -> a dataframe containing gene information for each SNP ID
"""
def get_gene_info(snps):
    ensembl = pyensembl.EnsemblRelease()

    cols = ['snp_id', 
            'gene_id', 
            'gene_name', 
            'biotype', 
            'contig', 
            'start', 
            'end', 
            'strand',
            ]

    genes_df = pd.DataFrame(columns=cols)

    for row in snps.iterrows():
        snp = row[1].SNP
        if row[1].BP == None: break
        pos = int(row[1].BP)
        chro = row[1].CHR
        genes = ensembl.genes_at_locus(contig=chro, position=pos)

        rows = gene_info(snp, genes)

        if rows != None:
            genes_df = pd.concat([genes_df, pd.DataFrame(rows, columns=genes_df.columns)])

    return genes_df

##################################################################################

# Simulation and Fst code
##################################################################################

"""
Hudson Fst computation for SNPs
"""
def compute_fst(snps):
    subpops = [[i for i in range(int(snps.shape[0]/2))], [i for i in range(int(snps.shape[0]/2), snps.shape[0])]]
    gts = snps[snps.columns[:-1]]
    gt = allel.GenotypeArray(snps[snps.columns[:-1]].values.reshape(gts.shape[1], gts.shape[0], 1))
    ac1 = gt.count_alleles(subpop=subpops[0])
    ac2 = gt.count_alleles(subpop=subpops[1])
    num, den = allel.hudson_fst(ac1, ac2)
    fsts = num / den
    return fsts
    

"""
simulate data from binomial (additive encoding) and compute Fst
Args: snp_info -> summary file for SNPs containing frequencies in cases and controls
      nu_donors -> the number of genotypes to simulate

Returns: df -> dataframe containing simulated genotypes
         fsts -> the fst scores for all SNPs computed from the simulated data
"""
def get_fst_snps(snp_info, nu_donors, fst_thresh):
    sims = np.array([])
    p_a = np.array([float(p) for p in snp_info.FRQ_A_12882])
    p_u = np.array([float(p) for p in snp_info.FRQ_U_21770])
    snps = [snp for snp in snp_info.SNP]
    subpops = [[i for i in range(int(nu_donors/2))], [i for i in range(int(nu_donors/2), nu_donors)]]
    sim_snps = []
    
    for snp, p1, p2 in zip(snps, p_a, p_u):
        cases = np.random.binomial(2, p=p1, size=int(nu_donors/2))
        controls = np.random.binomial(2, p=p2, size=int(nu_donors/2))
    
        sim_for_snp = np.concatenate((cases, controls), axis=0).reshape(1, nu_donors, 1)
        gt = allel.GenotypeArray(np.concatenate((cases, controls), axis=0).reshape(1, nu_donors, 1))
    
        ac1 = gt.count_alleles(subpop=subpops[0])
        ac2 = gt.count_alleles(subpop=subpops[1])
        num, den = allel.hudson_fst(ac1, ac2)
        fsts = num / den
        
        if fsts[0] > fst_thresh:
            sim_snps.append(snp)
            if sims.size == 0:
                sims = sim_for_snp.reshape(1, nu_donors)
            else:
                sims = np.concatenate((sims, sim_for_snp.reshape(1, nu_donors)))
                sims = sims.astype('uint8')
    
    gt_sim = pd.DataFrame(np.transpose(sims), columns=sim_snps)
    gt_sim['Type'] = int(nu_donors/2) * ['CASE'] + int(nu_donors/2) * ['CONTROL']
    return gt_sim

###################################################################################

# Genotypic encoding model
#####################################################################
"""
Genotypic encoding from additive encoding generated from get_fst_snps.
Args: snps -> simulated data encoded as 0, 1, 2 (genotype of SNP)
Returns: sim_gt -> a dataframe containing the genotypes as one hot encodings
"""
def genotype_encode(snps):
    gt_enc = [(ids, np.eye(3)[snps[ids]].astype('uint8')) for ids in snps.columns[:len(snps.columns)-1]]
    sim_gt = None
    for en in gt_enc:
        sim_gt = pd.concat([sim_gt, pd.DataFrame(en[1], columns=[en[0] + '_AA', en[0] + '_AB', en[0] + '_BB'])], axis=1)
     
    sim_gt['Type'] = snps['Type']
    return sim_gt

####################################################################

# Recessive dominant encoding model
#####################################################
"""
Helper function for encoding rec/dom model
"""
def encode_snp(snp):
    rec_enc = np.zeros((20000,2))
    i = 0
    for val in np.where(snp == 1)[1]:
        if val == 0:
            rec_enc[i, :] = np.array([1, 0])
        elif val == 1:
            rec_enc[i, :] = np.array([1, 1])
        else:
            rec_enc[i, :] = np.array([0, 1])
        i = i + 1
    return rec_enc

"""
rec/dom encoding model
"""
def rec_dom_encode(gt_enc):
    columns = [gt_enc.columns[:3][0][:-3] + '_A', gt_enc.columns[:3][0][:-3] + '_B']
    enc = encode_snp(gt_enc[gt_enc.columns[:3]])
    for i in range(3, len(gt_enc.columns)-1, 3):
        columns.append(gt_enc.columns[i:i+3][0][:-3] + '_A')
        columns.append(gt_enc.columns[i:i+3][0][:-3] + '_B')
        enc = np.concatenate((enc, encode_snp(gt_enc[gt_enc.columns[i:i+3]])), axis=1)
    df = pd.DataFrame(enc, columns=columns)
    df['Type'] = gt_enc['Type']
    
    return pd.DataFrame(enc, columns=columns)
######################################################
















    

    
