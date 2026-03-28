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
import scipy.sparse as sp


from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal
from scipy.sparse import issparse
import matplotlib.pyplot as plt

import random
import shutil
import glob
import sys
import glob as gb

import warnings
import numpy as np
from scipy.sparse import issparse, linalg, spdiags


from typing import Union

import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix




# TODO: Add docstrings
def scale(X, min=0, max=1):
    """TODO."""
    idx = np.isfinite(X)
    if any(idx):
        X = X - X[idx].min() + min
        xmax = X[idx].max()
        X = X / xmax * max if xmax != 0 else X * max
    return X

# TODO: Add docstrings
def select_distances(dist, n_neighbors=None):
    """TODO."""
    D = dist.copy()
    n_counts = (D > 0).sum(1).A1 if issparse(D) else (D > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()
    return D


# TODO: Add docstrings
def select_connectivities(connectivities, n_neighbors=None):
    """TODO."""
    C = connectivities.copy()
    n_counts = (C > 0).sum(1).A1 if issparse(C) else (C > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = C.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[::-1][n_neighbors:]
        dat[rm_idx] = 0
    C.eliminate_zeros()
    return C


# TODO: Add docstrings
def get_neighs(adata, mode="distances", key_added = None):
    """TODO."""
    if key_added is not None:
        mode = key_added + f'_{mode}'
    if hasattr(adata, "obsp") and mode in adata.obsp.keys():
        print(f'use adata.obsp["{mode}"]')
        return adata.obsp[mode]
    elif "neighbors" in adata.uns.keys() and mode in adata.uns["neighbors"]:
        print(f'use adata.uns["neighbors"]["{mode}"]')
        return adata.uns["neighbors"][mode]
    else:
        raise ValueError("The selected mode is not valid.")


# TODO: Add docstrings
def get_n_neighs(adata):
    """TODO."""
    return adata.uns.get("neighbors", {}).get("params", {}).get("n_neighbors", 0)


# TODO: Add docstrings
def verify_neighbors(adata):
    """TODO."""
    valid = "neighbors" in adata.uns.keys() and "params" in adata.uns["neighbors"]
    if valid:
        n_neighs = (get_neighs(adata, "distances") > 0).sum(1)
        # test whether the graph is corrupted
        valid = n_neighs.min() * 2 > n_neighs.max()
    if not valid:
        logg.warn(
            "The neighbor graph has an unexpected format "
            "(e.g. computed outside scvelo) \n"
            "or is corrupted (e.g. due to subsetting). "
            "Consider recomputing with `pp.neighbors`."
        )


# TODO: Add docstrings
def neighbors_to_be_recomputed(adata, n_neighbors=None):  # deprecated
    """TODO."""
    # check whether neighbors graph is disrupted or has insufficient number of neighbors
    invalid_neighs = (
        "neighbors" not in adata.uns.keys()
        or "params" not in adata.uns["neighbors"]
        or (n_neighbors is not None and n_neighbors > get_n_neighs(adata))
    )
    if invalid_neighs:
        return True
    else:
        n_neighs = (get_neighs(adata, "distances") > 0).sum(1)
        return n_neighs.max() * 0.1 > n_neighs.min()


# TODO: Add docstrings
def get_connectivities(
    adata, mode="connectivities", key_added = None, n_neighbors=None, recurse_neighbors=False
):
    """TODO."""
    if "neighbors" in adata.uns.keys():
        C = get_neighs(adata, mode, key_added = key_added)
        if n_neighbors is not None and n_neighbors < get_n_neighs(adata):
            if mode == "connectivities":
                C = select_connectivities(C, n_neighbors)
            else:
                C = select_distances(C, n_neighbors)
        connectivities = C > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            connectivities.setdiag(1)
            if recurse_neighbors:
                connectivities += connectivities.dot(connectivities * 0.5)
                connectivities.data = np.clip(connectivities.data, 0, 1)
            connectivities = connectivities.multiply(1.0 / connectivities.sum(1))
        return connectivities.tocsr().astype(np.float32)
    else:
        return None
