#!/bin/bash 

set -o pipefail
set -exu


python transformer_map_to_atlas.py \
--atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/UGE_atlas_sub.h5ad \
--adata_inte_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/UGE_sp_sub.h5ad \
--fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/results \
--recompute_pca \
--save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/results \
--n_neighbors 30




# python transformer_map_to_atlas.py \
# --atlas_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/UGE_atlas_sub.h5ad \
# --adata_inte_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/UGE_sp_sub.h5ad \
# --fitted_NN_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/results_NN300 \
# --recompute_pca \
# --save_nn_path /import/home2/xwanaf/Img2Expr/data/Reproducibility_data/PF/sp_map_to_sc_Epi/results_NN300 \
# --n_neighbors 300


