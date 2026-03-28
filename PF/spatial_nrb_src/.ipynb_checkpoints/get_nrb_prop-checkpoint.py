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

import logging
import sys

def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger


save_path = './logs'
save_path = Path(save_path)
loggings = configure_logging(str(save_path) + 'nrb_prop')


normal_samples = [
    'VUHD113', 
    'VUHD069', 'VUHD038', 'VUHD049', 
    'VUHD116A', 
    'VUHD116B', 'VUHD095', 'VUHD090'
]
disease_samples = [
    'VUILD91LA', 'VUILD48LA2', 'VUILD102LA', 'VUILD48LA1', 
    'VUILD106MA', 
    'VUILD105MA1', 
    'VUILD107MA', 
    'VUILD96MA', 'VUILD104MA1', 
    'VUILD102MA', 'VUILD142MA', 'VUILD96LA', 'VUILD141MA', 
    'VUILD58MA', 'VUILD91MA', 'VUILD49LA', 'VUILD78LA', 
    'VUILD110LA', 'VUILD105MA2', 
    'VUILD115MA', 
    'VUILD78MA', 'VUILD104MA2'
]
sample_all = normal_samples + disease_samples



sf = np.load('Xenium_sf.npy').item()
loggings.info(f'sf: {sf}')




df_list = []
sp_log_list = []
for sample in sample_all:
    #==================================== load sp_log ====================================#
    print(f'reading sp_log ... ')
    sp_base_path = '/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan'
    sp_data_path = Path(sp_base_path) / 'parquet_all_samples_pretrainV4' / f'{sample}_1e3'
    data_file_name = f'{sample}_raw_log.h5ad'

    sp_log = sc.read(sp_data_path / data_file_name)
    
    sp_inte_data_path = Path(sp_base_path) / 'Vannan_inte_pretrainV4' 
    data_file_name = f'adata_inte_{sample}.h5ad'
    sp_inte = sc.read(sp_inte_data_path / data_file_name)
    sp_inte.obs.index = sp_log.obs.index[sp_inte.obs['cell_id'].astype(int)]
    sp_log.obsm['inte'] = sp_inte[sp_log.obs.index].X


    if sample == 'VUILD105MA1':
        x_start, x_end, y_start, y_end = (3000, 8000, 3000, 6000)
        x_start_sp = int(4350)
        x_end_sp = int(5450)
        y_start_sp = int(4500)
        y_end_sp = int(5130)


        ##############################
        # get_affine
        ##############################
        sx = (x_end - x_start) / (x_end_sp - x_start_sp)
        sy = (y_end - y_start) / (y_end_sp - y_start_sp)

        tx = x_start - x_start_sp * sx
        ty = y_start - y_start_sp * sy

        # Construct the affine transformation matrix
        affine_matrix = np.array([[sx, 0, tx],
                                   [0, sy, ty],
                                   [0, 0, 1]])
        sp_log.obs[['x_centroid', 'y_centroid']] = sp_log.obs[['x_centroid', 'y_centroid']].values * np.array([sx, sy])[None,:] + np.array([tx, ty])[None,:]
        sp_log.obs[['x_centroid', 'y_centroid']] = sp_log.obs[['x_centroid', 'y_centroid']] / sf
        sp_log.obsm['spatial'] = sp_log.obs[['x_centroid', 'y_centroid']].values


    
    #######################################
    ###### map2 atlas
    #######################################
    map_results_path = '/import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_spatial_nn'
    file_insert = 'Disease' if sample in disease_samples else 'Normal'
    map_results_file_name = f'{sample}_{file_insert}/adata_inte_mapped.h5ad'
    
    mapped_adata = sc.read_h5ad(Path(map_results_path) / map_results_file_name, backed = 'r')
    cell_id_Xe_index = [int(_.split('_')[-1]) for _ in mapped_adata.obs.index]
    mapped_adata.obs.index = sp_log.obs.index[cell_id_Xe_index]
    pred_col_map = f'predict_mapped_ct_anno'
    print(f'original mapped data shape: {mapped_adata.shape}')
    mapped_adata = mapped_adata[mapped_adata.obs[f'{pred_col_map}_prob'] > .3]
    print(f'after filter mapped data shape: {mapped_adata.shape}')
    sp_log.obs = sp_log.obs.merge(
        mapped_adata.obs[[pred_col_map, f'{pred_col_map}_prob']], 
        how = 'left',
        left_index = True, 
        right_index = True, suffixes = (None, '_y')
    )
    sp_log.obs = sp_log.obs.rename(columns = {pred_col_map: 'pred_atlas'})



    df_list.append(sp_log.obs)
    sp_log_list.append(sp_log)
    
################## adata_all & adata_inte ##################
adata_all = anndata.concat(
    [sp_log_list[i] for i in range(len(sp_log_list))if sample_all[i] in disease_samples], 
    axis=0, 
    join = 'inner', 
    merge="same", 
    label = 'Slides', 
    keys = [_ for _ in sample_all if _ in disease_samples]
)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 0.85))

traingene_base_path = Path('/import/home2/xwanaf/Img2Expr/data/lung_HLCA/pretrain_v4')
train_genes = np.load(traingene_base_path / 'pretrain_data_train_genes.npy')

pred_col = 'pred_atlas'

adata_inte = sc.AnnData(adata_all.obsm['inte'], obs = adata_all.obs, var = pd.DataFrame(index = train_genes))



df_all = pd.concat(df_list, axis = 0)
df_all = df_all[[
    'sample', 'sample_type', 'sample_affect',
    'final_CT', 'final_lineage', 
#     'pred_raw',
    'pred_atlas', 
#     'frame', 'frame_type', 
    'TNiche', 'CNiche'
]]
################## nrb cell type proportion ##################
categories_all = adata_all.obs['pred_atlas'].value_counts().index.tolist()
nrb_celltype_df_list = []

samples_nrb = [
    'VUILD91LA', 
    'VUILD48LA2', 
    'VUILD102LA', 'VUILD48LA1', 'VUILD106MA', 
    'VUILD105MA1', 'VUILD107MA', 'VUILD96MA', 'VUILD104MA1', 
    'VUILD102MA', 'VUILD142MA', 'VUILD96LA', 'VUILD141MA', 
    'VUILD58MA', 'VUILD91MA', 'VUILD49LA', 'VUILD78LA', 
    'VUILD110LA', 'VUILD105MA2', 'VUILD115MA', 'VUILD78MA', 'VUILD104MA2'
]
sample_idx = [sample_all.index(_) for _ in samples_nrb]
df_nrb = [df_list[_] for _ in sample_idx]
sample_col = []
# for df, sample in zip(df_list, sample_all):
for df, sample in zip(df_nrb, samples_nrb):
    print(f' ================ {sample} ================')
    adata_spaloc = sc.AnnData(df[['x_centroid', 'y_centroid']])
    adata_spaloc.obs = df
#     adata_spaloc.obs.index = [f'{i}_{j}' for i,j in zip(adata_spaloc.obs.index, adata_spaloc.obs['sample'])]
    index = [_ is not np.nan for _ in adata_spaloc.obs['pred_atlas']]
    adata_spaloc = adata_spaloc[index]
    sc.pp.neighbors(adata_spaloc, n_neighbors=30, n_pcs=None)
    
    connectivities = adata_spaloc.obsp['connectivities']
    nonzero_index = np.where(connectivities.A > 0)

    ct_col = 'pred_atlas'
    nrb_celltype = adata_spaloc.obs[ct_col].to_numpy()[nonzero_index[1]]
    nrb_sample = adata_spaloc.obs['sample'].to_numpy()[nonzero_index[1]]
    cell_index = adata_spaloc.obs.index.to_numpy()[nonzero_index[0]]
    # cell_index = [f'{i}_{j}' for i,j in zip(cell_index, nrb_sample)]
    nrb_celltype_df = pd.DataFrame(nrb_celltype, index = nonzero_index[0], columns = ['nrb_celltype'])
    nrb_celltype_df['cell_index'] = cell_index
    nrb_celltype_df['nrb_celltype'] = pd.Categorical(nrb_celltype_df['nrb_celltype'], categories=categories_all, ordered=True)
    

    nrb_celltype_df = nrb_celltype_df.groupby('cell_index')['nrb_celltype'].value_counts(normalize=True).unstack(fill_value=0)
    sample_col = sample_col + [sample] * nrb_celltype_df.shape[0]
    nrb_celltype_df_list.append(nrb_celltype_df)
    
    
nrb_celltype_df_all = pd.concat(nrb_celltype_df_list, axis = 0)
prop_adata = sc.AnnData(
    nrb_celltype_df_all, 
#     obs = adata_spaloc.obs.loc[nrb_celltype_df.index], 
)
prop_adata.obs['sample'] = sample_col
prop_adata.obs.index = [f'{i}_{j}' for i,j in zip(prop_adata.obs.index, prop_adata.obs['sample'])]

df_all.index = [f'{i}_{j}' for i,j in zip(df_all.index, df_all['sample'])]
prop_adata.obs = prop_adata.obs.merge(df_all.copy(), left_index = True, right_index = True)
prop_adata.write('/import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_nrb_output/prop_adata_diseases.h5ad')