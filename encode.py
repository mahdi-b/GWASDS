import pandas as pd
import numpy as np
import allel


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


"""
simulate data from binomial (additive encoding) and compute Fst
Args: snp_info -> summary file for SNPs containing frequencies in cases and controls
      nu_donors -> the number of genotypes to simulate

Returns: df -> dataframe containing simulated genotypes
         fsts -> the fst scores for all SNPs computed from the simulated data
"""
def get_fst_snps(snp_info, nu_donors):
    
    # Get major allele frequencies for each SNP
    p_a = [float(p) if float(p) < 0.5 else 1-float(p) for p in snp_info.FRQ_A_12882]
    p_u = [float(p) if float(p) < 0.5 else 1-float(p) for p in snp_info.FRQ_U_21770]
    
    # Simulate cases and controls
    cases = np.array([[np.random.binomial(2, p=p_a)] for i in range(int(nu_donors/2))])
    controls = np.array([[np.random.binomial(2, p=p_u)] for i in range(int(nu_donors/2))])
    
    # Reshape data to compute Fst
    cases = cases.reshape(cases.shape[2], cases.shape[0], cases.shape[1])
    controls = controls.reshape(controls.shape[2], controls.shape[0], controls.shape[1])
    
    # Compute Hudson Fst
    full_sim = np.concatenate((cases, controls), axis=1)
    snps = allel.GenotypeArray(full_sim)
    subpops = [[i for i in range(int(nu_donors/2))], [i for i in range(int(nu_donors/2), nu_donors)]]
    ac1 = snps.count_alleles(subpop=subpops[0])
    ac2 = snps.count_alleles(subpop=subpops[1])
    num, den = allel.hudson_fst(ac1, ac2)
    fsts = num / den
    
    # Create dataframe for simulations
    df = pd.DataFrame(full_sim.reshape(full_sim.shape[1], full_sim.shape[0]), columns=snp_info.SNP)
    df['Type'] = int(nu_donors/2) * ['CASE'] + int(nu_donors/2) * ['CONTROL']
    
    return df, fsts
    

"""
Genotypic encoding from additive encoding generated from get_fst_snps.
Args: snps -> simulated data encoded as 0, 1, 2 (genotype of SNP)
Returns: sim_gt -> a dataframe containing the genotypes as one hot encodings
"""
def genotype_encode(snps):
    gt_enc = [(ids, np.eye(3)[snps[ids]]) for ids in snps.columns[:len(snps.columns)-1]]
    sim_gt = None
    for en in gt_enc:
        sim_gt = pd.concat([sim_gt, pd.DataFrame(en[1], columns=[en[0] + '_AA', en[0] + '_AB', en[0] + '_BB'])], axis=1)
     
    sim_gt['Type'] = snps['Type']
    return sim_gt


# TODO
def rec_dom_encode(snps):
    pass


def hybrid_encode(snps):
    pass

















    

    
