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

save_path = './debug'
save_path = Path(save_path)
loggings = configure_logging(str(save_path) + '/get_nrb_connectivity_log')
##################
# load ref data
##################
#==================================== load atlas_adata_ref ====================================#
loggings.info(f'load atlas_adata_ref ...')
base_path = '/import/home2/xwanaf/Img2Expr/data/lung_HLCA/pretrain_v4'
file_path = Path(base_path)
# adata.write(file_path / 'atlas_4e3_maskp5_sub.h5ad')
atlas_adata_ref = sc.read(file_path / 'atlas_1e3_maskp5_2p_v7_normal_frac10.h5ad')


#==================================== load VUILD107 sp_log ref  ====================================#
loggings.info(f'load VUILD107 sp_log ref ...')
base_path = Path('/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan')
data_path = base_path

data_file_name = 'VUILD107_log.h5ad'
sp_log_ref = sc.read(data_path / data_file_name)
loggings.info(sp_log_ref)

#==================================== get "sample_all" & "HE_file_names_all" ====================================#
loggings.info(f'get "sample_all" & "HE_file_names_all" ...')
# out_dir_base = '/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/RCTD_results_allsample'
# file_all = os.listdir(out_dir_base)
# sample_all = list(set([_.split('_')[-1] for _ in file_all]))
# sample_all = sample_all + ['VUILD107MA']

base_path = '/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan'
image_path = base_path + '/HE'  

HE_file_names = os.listdir(image_path)
loggings.info(f'length of all HE: {len(HE_file_names)}')

normal_samples = [
    'VUHD113', 
    'VUHD069', 
    'VUHD038', 'VUHD049', 'VUHD116A', 'VUHD116B', 'VUHD095', 'VUHD090'
]
disease_samples = [
    'VUILD91LA', 
    'VUILD48LA2', 
    'VUILD102LA', 'VUILD48LA1', 'VUILD106MA', 
    'VUILD105MA1', 'VUILD107MA', 'VUILD96MA', 'VUILD104MA1', 
    'VUILD102MA', 'VUILD142MA', 'VUILD96LA', 'VUILD141MA', 
    'VUILD58MA', 'VUILD91MA', 'VUILD49LA', 'VUILD78LA', 
    'VUILD110LA', 'VUILD105MA2', 'VUILD115MA', 'VUILD78MA', 'VUILD104MA2'
]
sample_all = normal_samples + disease_samples
large_samples = ['adata_inte_VUILD107MA.h5ad',
 'adata_inte_VUILD110LA.h5ad',
 'adata_inte_VUILD106MA.h5ad',
 'adata_inte_VUILD58MA.h5ad',
 'adata_inte_VUILD115MA.h5ad',
 'adata_inte_VUILD96MA.h5ad']
large_samples = [_.split('adata_inte_')[1].split('.h5ad')[0] for _ in large_samples]
loggings.info(f'sample_all: {sample_all}')
# len(IPFTMA1_samples + IPFTMA2_samples)   


epi_sl_ct = [
   'AT1',                           
    'Suprabasal',                    
    'Basal',                         
    'Ciliated',                      
    'Ciliated MUC5AC+',              
    'Ciliated SCGB3A2+',             
    'Secretory MUC5B+',              
    'Secretory MUC5AC+',             
    'AT2',                           
    'SCGB1A1+/SCGB3A1+/SCGB3A2+',    
    'SCGB3A1+/SCGB3A2+',             
    'SMG duct',                      
    'SMG-Serous',                    
    'Basal-Prolif',                  
    'Suprabasal SCGB1A1+',           
    'Ionocyte',                      
    'Mesothelial',                   
    'Aberrant Basaloid',              
    'Squamous',                       
    'Neuroendocrine',                                                                                 
]
loggings.info(len(epi_sl_ct))
loggings.info(epi_sl_ct)


#==================================== load pathology anno df ====================================#
loggings.info(f'load pathology anno df ...')
pickle_path = '/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/HE_frameAnno'
pickle_path = Path(pickle_path)
pickle_file_name = 'frames.pickle'

import pickle

# Load the pickled dictionary from the file
with open(pickle_path / pickle_file_name, 'rb') as file:
    frame_annos = pickle.load(file)




#==================================== load sf ====================================#
loggings.info(f'load sf ...')
import zarr

base_path = '/home/share/xwanaf/Img2Expr/data/Xenium_melanoma_test/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs'
zarr_file_path = base_path + '/cells/masks'  # Replace with the actual path to the .zarr file
zarr_file = zarr.open(zarr_file_path, mode='r')
keys_in_zarr = list(zarr_file.keys())
loggings.info(f"Keys in the .zarr file: {keys_in_zarr}")
group = zarr_file['homogeneous_transform']  # Replace 'group_name' with the actual name of the group or array
data = group[:]
sf = data[0,0]
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
    
#     #==================================== load image ====================================#
#     from PIL import Image, ImageDraw
#     from skimage import io
#     import pandas as pd

#     HE_file_name = [_ for _ in HE_file_names if sample in _]

#     ################
#     # read img 
#     ################
#     loggings.info(f'reading img ... ')
#     image = io.imread(Path(image_path) / HE_file_name[0])
#     loggings.info(f'HE image shape: {image.shape}')

    
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


#     from einops import rearrange
#     uns_dict = {}
#     uns_dict['spatial'] = {}
#     uns_dict['spatial']['lung'] = {}
#     uns_dict['spatial']['lung']['images'] = {}
#     uns_dict['spatial']['lung']['images']['hires'] = image 
#     uns_dict['spatial']['lung']['scalefactors'] = {}
#     uns_dict['spatial']['lung']['scalefactors']['spot_diameter_fullres'] = 10
#     uns_dict['spatial']['lung']['scalefactors']['tissue_hires_scalef'] = sf

#     sp_log.uns['spatial'] = uns_dict['spatial']
#     sp_log.obsm['spatial'] = sp_log.obs.loc[:, ['x_centroid', 'y_centroid']].values

   
    
    #######################################
    ###### map2 atlas
    #######################################
    map_results_path = '/home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/Map2atlas_results_pretrainV4'
    file_insert = 'Disease' if sample in disease_samples else 'Normal'
    map_results_file_name = f'{sample}_v7_Manu{file_insert}/adata_inte_mapped.h5ad'
    
    mapped_adata = sc.read_h5ad(Path(map_results_path) / map_results_file_name, backed = 'r')
    cell_id_Xe_index = [int(_.split('_')[-1]) for _ in mapped_adata.obs.index]
    mapped_adata.obs.index = sp_log.obs.index[cell_id_Xe_index]
    pred_col_map = f'predict_mapped_ct_v7'
    print(f'original mapped data shape: {mapped_adata.shape}')
    mapped_adata = mapped_adata[mapped_adata.obs[f'{pred_col_map}_prob'] > .5]
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
# adata_inte.uns = adata_all.uns
# adata_inte.obsm = adata_all.obsm

# # adata_inte = adata_inte[adata_inte.obs[pred_col].isin(epi_sl_ct)].copy()
# value_counts = adata_inte.obs[pred_col].value_counts()
# sl_ct = value_counts[value_counts > 13].index.tolist()
# adata_inte = adata_inte[adata_inte.obs[pred_col].isin(sl_ct)]


################## nrb expr ##################
nrb_expr_cts = [
#     'EC venous systemic', 'Monocyte-derived Mph', 'CD4 T cells',
#                   'FB ECM SOX4+', 'FB ECM', 'FB HAS1+', 'AT2',
#                   'EC general capillary', 'Aberrant Basaloid',
#                   'Alveolar macrophages', 'AT1',
#     'Basal',
#     'Monocyte-derived Mph',
    'FB ECM',
]

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
# for df, sample in zip(df_list, sample_all):
for i, (df, sample) in enumerate(zip(df_nrb, samples_nrb)):
# for i, (df, sample) in enumerate(zip(df_list, sample_all)):
    df.index = [f'{i}_{j}' for i,j in zip(df.index, df['sample'])]
#     index = [_ is np.nan for _ in df['pred_atlas']]
#     df = df.loc[index]
    sp_log = sp_log_list_nrb[i]
#     sp_log.obs.index = [f'{i}_{j}' for i,j in zip(sp_log.obs.index, sp_log.obs['sample'])]
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
    # cell_index = [f'{i}_{j}' for i,j in zip(cell_index, nrb_sample)]
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
# nrb_celltype_df_all = pd.concat(nrb_celltype_df_list, axis = 0)
# nrb_celltype_df_all.to_csv('./prop_adata_files/nrb/nrb_15.csv')

ct = nrb_expr_cts[0].replace(' ', '_')
save_base_path = '/import/home2/xwanaf/Img2Expr/data/lung_HLCA/pretrain_v4/anno_V7_Xenium_spatial_nrb'
save_base_path = Path(save_base_path)

save_file_name = f'nrb_expr_raw_ManuSplit_allgenes_{ct}.h5ad'
adata_all_raw.write(save_base_path / save_file_name)

save_file_name = f'nrb_expr_inte_ManuSplit_allgenes_{ct}.h5ad'
adata_all_inte.write(save_base_path / save_file_name)
