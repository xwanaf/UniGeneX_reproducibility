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

import argparse
parser = argparse.ArgumentParser(description='simulation sour_sep')
parser.add_argument('--nrb_celltype', type=str, help='nrb celltype to calculate expression', default=None)

args = parser.parse_args()




save_path = './logs'
save_path = Path(save_path)
loggings = configure_logging(str(save_path) + 'nrb_expr')


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
    loggings.info(f'reading sp_log ... ')
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


################## nrb expr ##################
# nrb_expr_cts = [
# #     'EC venous systemic', 'Monocyte-derived Mph', 'CD4 T cells',
# #                   'FB ECM SOX4+', 'FB ECM', 'FB HAS1+', 'AT2',
# #                   'EC general capillary', 'Aberrant Basaloid',
# #                   'Alveolar macrophages', 'AT1',
# #     'Basal',
#     'Monocyte-derived Mph',
# #     'FB ECM',
# ]
nrb_expr_cts = [args.nrb_celltype]

expr_raw_adata_list = []
expr_inte_adata_list = []
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
sp_log_list_nrb = [sp_log_list[_] for _ in sample_idx]
sample_col = []

for i, (df, sample) in enumerate(zip(df_nrb, samples_nrb)):
    df.index = [f'{i}_{j}' for i,j in zip(df.index, df['sample'])]
    sp_log = sp_log_list_nrb[i]

    adata_inte = sc.AnnData(sp_log.obsm['inte'], obs = sp_log.obs, var = pd.DataFrame(index = train_genes))
    
    
    print(f' ================ {sample} ================')
    adata_spaloc = sc.AnnData(df[['x_centroid', 'y_centroid']])
    adata_spaloc.obs = df.copy()
#     adata_spaloc.obs.index = [f'{i}_{j}' for i,j in zip(adata_spaloc.obs.index, adata_spaloc.obs['sample'])]
    index = [_ is not np.nan for _ in adata_spaloc.obs['pred_atlas']]
    adata_spaloc = adata_spaloc[index].copy()
    sp_log = sp_log[index].copy()
    adata_inte = adata_inte[index].copy()
    
    sc.pp.neighbors(adata_spaloc, n_neighbors=30, n_pcs=None)
    
    connectivities = adata_spaloc.obsp['connectivities']
    nonzero_index = np.where(connectivities.A > 0)
    
    nrb_cell_index = adata_spaloc.obs.index.to_numpy()[nonzero_index[1]]
    cell_index = adata_spaloc.obs.index.to_numpy()[nonzero_index[0]]

    nrb_celltype_df = pd.DataFrame(cell_index, index = nonzero_index[0], columns = ['cell_index'])

    

    ct_col = 'pred_atlas'
    nrb_celltype = adata_spaloc.obs[ct_col].to_numpy()[nonzero_index[1]]
    for ct in nrb_expr_cts:
        index = np.where(nrb_celltype == ct)[0]
        nonzero_index_sub = nonzero_index[1][index]
        nrb_celltype_df_sub = nrb_celltype_df.iloc[index].copy()
        nrb_celltype_df_sub.set_index('cell_index', inplace = True)
        
        sp_log_expr = sp_log.to_df().iloc[nonzero_index_sub]
        expr_raw = sp_log_expr.copy()
        expr_raw.index = nrb_celltype_df_sub.index
        expr_raw = expr_raw.groupby('cell_index').mean()
        expr_raw_adata = sc.AnnData(expr_raw)
        expr_raw_adata.obs['sample'] = sample
        expr_raw_adata.obs['nrb_ct'] = ct
        
        
        adata_inte_expr = adata_inte.to_df().iloc[nonzero_index_sub]
        expr_inte = adata_inte_expr.copy()
        expr_inte.index = nrb_celltype_df_sub.index
        expr_inte = expr_inte.groupby('cell_index').mean()
        expr_inte_adata = sc.AnnData(expr_inte)
        expr_inte_adata.obs['sample'] = sample
        expr_inte_adata.obs['nrb_ct'] = ct
        
        expr_raw_adata_list.append(expr_raw_adata)
        expr_inte_adata_list.append(expr_inte_adata)

adata_all_raw = anndata.concat(
    expr_raw_adata_list, 
    axis=0, 
    join = 'inner', 
    merge="same", 
#     label = 'Slides', 
#     keys = [_ for _ in sample_all if _ in disease_samples]
)
adata_all_inte = anndata.concat(
    expr_inte_adata_list, 
    axis=0, 
    join = 'inner', 
    merge="same", 
#     label = 'Slides', 
#     keys = [_ for _ in sample_all if _ in disease_samples]
)

ct = nrb_expr_cts[0].replace(' ', '_')
save_base_path = '/import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_nrb_output'
save_base_path = Path(save_base_path)

save_file_name = f'nrb_expr_raw_{ct}.h5ad'
adata_all_raw.write(save_base_path / save_file_name)

save_file_name = f'nrb_expr_UGE_{ct}.h5ad'
adata_all_inte.write(save_base_path / save_file_name)
