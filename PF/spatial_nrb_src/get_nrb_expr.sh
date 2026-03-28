#!/bin/bash 

set -o pipefail
set -exu


python get_nrb_expr.py \
--nrb_celltype "Monocyte-derived Mph" &


python get_nrb_expr.py \
--nrb_celltype "FB ECM"