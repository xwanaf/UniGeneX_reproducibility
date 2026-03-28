#!/bin/bash 

set -o pipefail
set -exu


######### test sample VUHD113
python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/Reference_data/atlas_Normal.h5ad \
--adata_inte_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/Vannan_inte_pretrainV4/adata_inte_VUHD113.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_reference_nn_normal \
--recompute_pca \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_spatial_nn/VUHD113_Normal \
--atlas_assign_label_col ct_anno




declare -a arr=('VUHD116A' 'VUHD069' 'VUHD049' 'VUHD090' 'VUHD095' 'VUHD038' 'VUHD116B')
##################### 
# spImp
##################### 

for adata in "${arr[@]}"
do

echo ${adata%.h5ad}


# map raw
python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/Reference_data/atlas_Normal.h5ad \
--adata_inte_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/Vannan_inte_pretrainV4/adata_inte_${adata}.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_reference_nn_normal \
--use_existing_nn \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_spatial_nn/${adata}_Normal \
--atlas_assign_label_col ct_anno



done
