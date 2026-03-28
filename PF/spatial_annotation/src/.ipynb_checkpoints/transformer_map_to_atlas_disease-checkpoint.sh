#!/bin/bash 

set -o pipefail
set -exu


######### test sample VUILD107MA
python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/Reference_data/atlas_Disease.h5ad \
--adata_inte_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/Vannan_inte_pretrainV4/adata_inte_VUILD107MA.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_reference_nn_disease \
--recompute_pca \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_spatial_nn/VUILD107MA_Disease \
--atlas_assign_label_col ct_anno




declare -a arr=('VUILD91MA' 'VUILD91LA' 'VUILD48LA2' 'VUILD102LA' 'VUILD48LA1' 'VUILD106MA' 'VUILD105MA1' 'VUILD96MA' 'VUILD104MA1' 'VUILD102MA' 'VUILD142MA' 'VUILD96LA' 'VUILD141MA' 'VUILD58MA' 'VUILD49LA' 'VUILD78LA' 'VUILD110LA' 'VUILD105MA2' 'VUILD115MA' 'VUILD78MA' 'VUILD104MA2')
##################### 
# spImp
##################### 

for adata in "${arr[@]}"
do

echo ${adata%.h5ad}


# map raw
python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/Reference_data/atlas_Disease.h5ad \
--adata_inte_path /home/share/xwanaf/Img2Expr/data/lung_HLCA/Xenium_Vannan/Vannan_inte_pretrainV4/adata_inte_${adata}.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_reference_nn_disease \
--use_existing_nn \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/spatial_annotation/fitted_spatial_nn/${adata}_Disease \
--atlas_assign_label_col ct_anno



done

