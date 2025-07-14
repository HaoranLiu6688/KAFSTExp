import json
import scanpy as sc
import h5py
import os
import numpy as np
import pandas as pd

# Prepare the dataset required for leave-one-out cross-validation
# The split follows the partitioning method provided by HEST-1k

fm = 'UNI'
root_path = r'data\IDC'

cv_save_path = os.path.join(root_path, 'cv_data/'+fm)
split_dir = os.path.join(root_path, 'splits')
splits = os.listdir(split_dir)
n_splits = len(splits) // 2
with open(os.path.join(root_path, 'var_50genes.json'), 'r') as f:
    genes = json.load(f)['genes']

fold_num = 0
for i in range(n_splits):
    fold_num += 1
    train_split = os.path.join(split_dir, f'train_{i}.csv')
    test_split = os.path.join(split_dir, f'test_{i}.csv')

    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)

    for split_key, split in zip(['train', 'test'], [train_df, test_df]):
        embed_fold = np.empty((0,))
        dataframes = []
        for sample in range(len(split)):
            patches_path = split.iloc[sample]['patches_path']
            expr_path = split.iloc[sample]['expr_path']

            adata = sc.read_h5ad(os.path.join(root_path, expr_path))

            with (h5py.File(os.path.join(root_path, patches_path), 'r') as f):
                keys = list(f.keys())
                barcodes = f['barcodes'][:].flatten().astype(str).tolist()
                coords = f['coords'][:]
                embed = f['embeddings'][:]
                adata = adata[barcodes]
                adata = adata[:, genes]
                filtered_adata = adata.copy()
                filtered_adata.X = filtered_adata.X.astype(np.float64)
                sc.pp.log1p(filtered_adata)
                adata = filtered_adata.to_df()

                if embed_fold.size == 0:
                    embed_fold = embed
                else:
                    embed_fold = np.concatenate((embed_fold, embed), axis=0)
            dataframes.append(adata)

        adata_fold = pd.concat(dataframes, axis=0)
        np.savetxt(cv_save_path+'/fold'+str(fold_num)+'_'+split_key+'_embeddings.csv', embed_fold, delimiter=",")
        adata_fold.to_csv(cv_save_path+'/fold'+str(fold_num)+'_'+split_key+'_gene_labels.csv', index=False)
        pass
