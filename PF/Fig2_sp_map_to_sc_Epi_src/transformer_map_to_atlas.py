# build large-scale data in scBank format from a group of AnnData objects
# %%
import gc
import json
from pathlib import Path
import argparse
import shutil
import traceback
from typing import Dict, List, Optional
import warnings
import numpy as np
import os
import anndata

import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal
from scipy.sparse import issparse


import sys
sys.path.append('/home/xwanaf/bio/scGPT-dev-temp/Atlas_integration/lung')
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--atlas_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--adata_inte_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--fitted_NN_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--recompute_pca",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--use_existing_nn",
        action="store_true",
        help="Whether use gaussian latent prior.",
    )
parser.add_argument(
        "--save_nn_path",  
        type=str,
        default=None,
        help="Decoder type.",
    )
parser.add_argument(
        "--n_neighbors",  
        type=int,
        default=30,
        help="Decoder type.",
    )
args = parser.parse_args()

########################################
# define paths
########################################
Path(args.fitted_NN_path).mkdir(parents=True, exist_ok=True)
Path(args.save_nn_path).mkdir(parents=True, exist_ok=True)
loggings = configure_logging(str(Path(args.save_nn_path) / 'transformer_map_to_atlas_log'))
loggings.info(f'save_nn_path: {args.save_nn_path}')


########################################
# map to atlas
########################################
loggings.info(f' ============================== start map to atlas ============================== ')

atlas_adata = sc.read(args.atlas_path)
adata_inte_sub = sc.read(args.adata_inte_path)

same_gene = np.array_equal(atlas_adata.var.index, adata_inte_sub.var.index)
if not same_gene:
    overlap_genes = np.array(list(set(atlas_adata.var.index) & set(adata_inte_sub.var.index)))
    loggings.info(f'{overlap_genes.shape[0]} overlap_genes')
    atlas_adata = atlas_adata[:, overlap_genes]
    adata_inte_sub = adata_inte_sub[:, overlap_genes]
    loggings.info(f'after overlap genes atlas_adata shape: {atlas_adata.shape}')
    loggings.info(f'after overlap genes adata_inte_sub shape: {adata_inte_sub.shape}')


########################################
# fit NN
########################################
import pickle
loggings.info(f'load fitted nn ...')
file_path = args.fitted_NN_path
pickle_file = Path(file_path) / 'knnpickle_file'
if args.use_existing_nn:
    loggings.info(f'loaded calculated X_pca')
    atlas_pcs = np.load(Path(file_path) / 'atlas_pcs.npy')
    pcs = np.load(Path(file_path) / 'pcs.npy')
    atlas_adata.obsm['X_pca'] = atlas_pcs
    atlas_adata.varm['PCs'] = pcs
    
    loggings.info(f'loaded calculated atlas neibor files')
    nn_ = pickle.load(open(Path(file_path) / 'knnpickle_file', 'rb'))
    neigh_dist_atlas = np.load(Path(file_path) / 'neigh_dist_atlas.npy')
    neigh_ind_atlas = np.load(Path(file_path) / 'neigh_ind_atlas.npy')
else:
    if args.recompute_pca:
        loggings.info(f'recompute_pca ...')
        sc.tl.pca(atlas_adata, n_comps=30,use_highly_variable=False) #svd_solver='arpack', n_comps=10, use_highly_variable=False)
    else:
        assert 'PCs' in atlas_adata.varm.keys()
    atlas_pcs = atlas_adata.obsm['X_pca']
    pcs = atlas_adata.varm['PCs']
    np.save(Path(file_path) / 'atlas_pcs.npy', atlas_pcs)
    np.save(Path(file_path) / 'pcs.npy', pcs)
    
    

    loggings.info(f'refit nn ...')
    ds1 = atlas_pcs
    if isinstance(ds1, np.matrix):
        ds1 = np.asarray(ds1)


    nn_ = NearestNeighbors(n_neighbors=args.n_neighbors, metric='cosine')
    nn_.fit(ds1)
    neigh_dist_atlas, neigh_ind_atlas = nn_.kneighbors(return_distance=True)

    np.save(Path(file_path) / 'neigh_dist_atlas.npy', neigh_dist_atlas)
    np.save(Path(file_path) / 'neigh_ind_atlas.npy', neigh_ind_atlas)

    import pickle
    knnPickle = open(Path(file_path) / 'knnpickle_file', 'wb') 
    pickle.dump(nn_, knnPickle)  
    knnPickle.close()

    # load the model from disk
    nn_ = pickle.load(open(Path(file_path) / 'knnpickle_file', 'rb'))



########################################
# fit NN
########################################

loggings.info(f'get mapped pc ...')
adata_inte =  sc.AnnData(adata_inte_sub.to_df())
atlas_adata_bg = sc.AnnData(atlas_adata.to_df())
atlas_adata_bg.varm['PCs'] = atlas_adata.varm['PCs']
atlas_adata_bg.obsm['X_pca'] = atlas_adata.obsm['X_pca']


# map_cell_num = 19246
join = 'outer'
recompute_pca = False


pcs = atlas_adata_bg.varm['PCs']
atlas_pcs = atlas_adata_bg.obsm['X_pca']
adata_inte.obsm['X_pca'] = (adata_inte.X - atlas_adata_bg.X.mean(0)) @ pcs
adata_inte.varm['PCs'] = pcs

loggings.info(f'nn fit mapped data ...')
mapped_pcs = adata_inte.obsm['X_pca']

ds1 = atlas_pcs
ds2 = mapped_pcs
if isinstance(ds1, np.matrix):
    ds1 = np.asarray(ds1)
if isinstance(ds2, np.matrix):
    ds2 = np.asarray(ds2)

neigh_dist, neigh_ind = nn_.kneighbors(ds2, return_distance=True)

loggings.info(f'nn finish fit mapped data, make adata_all')
adata_all = anndata.concat(
    [atlas_adata_bg, adata_inte], 
    axis=0, 
    join = join, 
    merge="same", 
    label = 'atlas_inte', 
    keys = ['atlas', 'mapped_data']
)

save_nn_path = args.save_nn_path
save_nn_path = Path(save_nn_path)
loggings.info(f'save nn results to {str(save_nn_path)}')
np.save(save_nn_path / 'neigh_dist.npy', neigh_dist)
np.save(save_nn_path / 'neigh_ind.npy', neigh_ind)
adata_all.write(save_nn_path / 'adata_all.h5ad')


########################################
# make adata all
########################################
loggings.info(f'make distances and connectivities to adata_all')
knn_distances = np.vstack((neigh_dist_atlas, neigh_dist))
knn_indices = np.vstack((neigh_ind_atlas, neigh_ind))
adata_all.obsp['distances'], adata_all.obsp['connectivities'] = sc.neighbors._compute_connectivities_umap(
    knn_indices,
    knn_distances,
    n_obs=adata_all.shape[0],
    n_neighbors=args.n_neighbors,
)
# adata_all.write(save_nn_path / 'adata_all.h5ad')


from scipy.sparse import save_npz, load_npz
save_npz(save_nn_path / 'adata_all_connectivities.npz', adata_all.obsp['connectivities'])
save_npz(save_nn_path / 'adata_all_distances.npz', adata_all.obsp['distances'])
