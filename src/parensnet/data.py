import abc
import typing as t
#
import anndata
import h5py
import numpy as np
import os
import scanpy as sc
import scipy
import pandas as pd
#
from anndata import AnnData
from pydantic import BaseModel
from .types import DataArray, NDFloatArray


def idx_dict(slist: list[str]) -> dict[str, int]:
    return dict(zip(slist, range(len(slist))))


def matrix_sub_row(
    in_matrix: DataArray,
    row : int
) -> DataArray:
    if isinstance(in_matrix, (scipy.sparse.csr_matrix,
                              scipy.sparse.csc_matrix)):
        return scipy.sparse.hstack(
            (
                in_matrix[:, :row], 
                in_matrix[:, row + 1 :]
            )
        )
    elif isinstance(
        in_matrix,
        np.ndarray
    ):
        return np.delete(in_matrix, row, 1)
    else:
        return in_matrix


def matrix_column(
    in_matrix: DataArray,
    col : int
) -> DataArray:
    if isinstance(in_matrix, (scipy.sparse.csr_matrix,
                              scipy.sparse.csc_matrix)):
        return in_matrix[:, col].toarray().flatten() # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(
        in_matrix,
        np.ndarray
    ):
        return in_matrix[:, col].flatten()


def target_gene_matrix(
    exp_matrix: NDFloatArray,
    tf_exp_matrix: NDFloatArray,
    gene_map: dict[str, int],
    tf_map: dict[str, int],
    target_gene: str,
):
    if target_gene not in gene_map :
        return None, None
    if target_gene not in tf_map:
        sidx = gene_map[target_gene]
        return tf_exp_matrix, matrix_column(exp_matrix, sidx)
    tidx = tf_map[target_gene]
    return (matrix_sub_row(tf_exp_matrix, tidx),
            matrix_column(tf_exp_matrix, tidx))



def read_h5ad_xcol(h5ad_file: str, cindex: int):
    with h5py.File(h5ad_file) as hfx:
        mat_col: NDFloatArray = hfx["X"][:, cindex] # pyright: ignore[reportIndexIssue, reportAssignmentType]
        return mat_col


def read_h5ad_submatrix(h5ad_file: str, col_indexes: list[int]):
    with h5py.File(h5ad_file) as hfx:
        #k
        indexes = np.array(col_indexes)
        indexes_asort = np.argsort(col_indexes)
        indexes_srtd = indexes[indexes_asort]
        submat_srtd: NDFloatArray = hfx["X"][:, indexes_srtd] # pyright: ignore[reportIndexIssue, reportAssignmentType]
        submat = np.zeros(shape=submat_srtd.shape,
                          dtype=submat_srtd.dtype)
        submat[:, indexes_asort] = submat_srtd
        return submat


def target_gene_matrix_h5ad_index(
    h5ad_file: str,
    tf_exp_matrix: NDFloatArray,
    gene_index: int,
    tf_index: int,
):
    if gene_index < 0:
        return None, None
    if tf_index < 0:
        return tf_exp_matrix, read_h5ad_xcol(h5ad_file, gene_index)
    return (
        matrix_sub_row(tf_exp_matrix, tf_index),
        matrix_column(tf_exp_matrix, tf_index)
    )


def target_gene_matrix_h5ad(
    h5ad_file: str,
    tf_exp_matrix: NDFloatArray,
    gene_map: dict[str, int],
    tf_map: dict[str, int],
    target_gene: str,
):
    return target_gene_matrix_h5ad_index(
        h5ad_file,
        tf_exp_matrix,
        gene_map[target_gene] if target_gene in gene_map else -1,
        tf_map[target_gene] if target_gene in tf_map else -1,
    )


class SCDataArgs(BaseModel):
    format: t.Literal['ad'] = 'ad'
    data : t.Literal['sparse', 'dense'] = 'dense'
    tf_file : str = "./trrust_tf.txt"
    h5ad_file : str = "./../rsc/h5/adata.raw.h5ad"
    select_hvg : int = True
    ntop_genes : int = 2000
    scale_regress : bool=True
    nsub_cells : int  = 0
    skip_preprocess: bool = False
    digits: int | None = None


class CSVDataArgs(BaseModel):
    format: t.Literal['csv'] = 'csv'
    csv_file : str = ""
    nsub_cells : int  = 0
    tf_file : str = ""
    digits: int | None = None


class NetworkGenes(BaseModel):
    ntfs : int = 0
    ngenes : int = 0
    gene_map : dict[str, int] = {}
    tf_map : dict[str, int] = {}
    gene_list : list[str] = []
    tf_list : list[str] = []
    tf_indices : list[int] = []

    @classmethod
    def from_list(cls, gene_list: list[str], full_tf_list: list[str]):
        net_genes = cls()
        #
        # Gene list
        net_genes.gene_list = gene_list
        net_genes.gene_map = idx_dict(net_genes.gene_list)
        net_genes.ngenes = len(net_genes.gene_list)
        #
        # TF list
        all_tf_set = set(full_tf_list)
        net_genes.tf_list  = list(
            tx for tx in all_tf_set if tx in net_genes.gene_map
        )
        net_genes.ntfs = len(net_genes.tf_list)
        net_genes.tf_map = idx_dict(net_genes.tf_list)
        net_genes.tf_indices = list(
            net_genes.gene_map[tfx] for tfx in net_genes.tf_list
        )
        return net_genes

    @classmethod
    def from_adata(cls, adata: AnnData, tf_file: str):
        tfdf = pd.read_csv(tf_file)
        full_tf_list : list[str] = list(tfdf.gene)
        return NetworkGenes.from_list(
            list(adata.var.index),
            full_tf_list
        )

    @classmethod
    def from_h5ad(
        cls,
        h5ad_file: os.PathLike[str] | str,
        tf_file: os.PathLike[str] | str,
        backed: t.Literal['r', 'r+'] | bool | None = "r"
    ):
        tfdf = pd.read_csv(tf_file)
        full_tf_list : list[str] = list(tfdf.gene)
        adata = anndata.read_h5ad(h5ad_file, backed=backed)
        adata.file.close()
        gene_list = list(adata.var.index)
        return NetworkGenes.from_list(gene_list, full_tf_list)


class ExpDataProcessor(abc.ABC):
    def __init__(self) -> None:
        self.net_genes_ : NetworkGenes = NetworkGenes()

    @property
    @abc.abstractmethod
    def exp_matrix(self) -> NDFloatArray:
        pass

    @property
    @abc.abstractmethod
    def tf_exp_matrix(self) -> NDFloatArray:
        pass

    @property
    def ntfs(self) -> int:
        return self.net_genes_.ntfs

    @property
    def ngenes(self) -> int:
        return self.net_genes_.ngenes

    @property
    def gene_map(self) -> dict[str, int]:
        return self.net_genes_.gene_map

    @property
    def tf_map(self) -> dict[str, int]:
        return self.net_genes_.tf_map

    @property
    def gene_list(self) -> list[str]:
        return self.net_genes_.gene_list

    @property
    def tf_list(self) -> list[str]:
        return self.net_genes_.tf_list

    def print(self):
        print(f"""
            No. Genes             : {self.ngenes}
            No. TF                : {self.ntfs}
            Expt Matrix shape     : {self.exp_matrix.shape}
            TF Expt Matrix  shape : {self.tf_exp_matrix.shape}
        """)

    def target_gene_matrix(self, target_gene: str):
        return target_gene_matrix(self.exp_matrix,
                                  self.tf_exp_matrix, 
                                  self.gene_map,
                                  self.tf_map,
                                  target_gene)


class CSVDataProcessor(ExpDataProcessor):
    def __init__(self, sargs: CSVDataArgs) -> None:
        super().__init__()
        self.adata_ : pd.DataFrame = pd.read_csv(
            sargs.csv_file, header=0, index_col=0
        )
        self.ematrix_ : NDFloatArray = self.adata_.T.to_numpy()
        if sargs.digits:
            self.ematrix_ = np.round(self.ematrix_, sargs.digits)
        #
        self.net_genes_.gene_list  = list(self.adata_.index)
        self.net_genes_.gene_map = idx_dict(self.net_genes_.gene_list)
        self.net_genes_.ngenes = len(self.net_genes_.gene_list)
        #
        self.net_genes_.tf_list  = list(self.adata_.index)
        self.net_genes_.tf_map = idx_dict(self.net_genes_.tf_list)
        self.net_genes_.ntfs  = len(self.net_genes_.tf_list)
        self.net_genes_.tf_indices  = list(
            self.net_genes_.gene_map[tfx] for tfx in self.net_genes_.tf_list
        )

    @property
    @t.override
    def exp_matrix(self) -> NDFloatArray:
        return self.ematrix_

    @property
    @t.override
    def tf_exp_matrix(self) -> NDFloatArray:
        return self.ematrix_


class SCDataProcessor(ExpDataProcessor):
    def __init__(self, sargs: SCDataArgs) -> None:
        super().__init__()
        tfdf = pd.read_csv(sargs.tf_file)
        self.all_tf_list_ : list[str] = list(tfdf.gene)
        self.adata_ : AnnData = sc.read_h5ad(sargs.h5ad_file)
        if sargs.skip_preprocess:
            print("Skipping Pre-process")
        else:
            self.__pre_process(sargs)
        #
        self.net_genes_ : NetworkGenes = NetworkGenes.from_list(
            list(self.adata_.var.index),
            self.all_tf_list_
        )
        #
        # Round 
        if sargs.digits is not None:
            if self.adata_.X is not None:
                self.adata_.X = np.round( # pyright:ignore[reportCallIssue]
                    self.adata_.X, # pyright:ignore[reportArgumentType]
                    sargs.digits
                )
        #
        # TF Anndata
        self.tf_adata_ : AnnData = (
            self.adata_[:, self.net_genes_.tf_list]
        )


    # def __init_genes_meta(self, glist):
    #     #
    #     # Gene list
    #     self.gene_list_ : list[str] = glist
    #     self.gene_map_ : dict[str, int]= idx_dict(self.gene_list_)
    #     self.ngenes_ : int = len(self.gene_list_)
    #     #
    #     # TF list
    #     all_tf_set = set(self.all_tf_list_)
    #     self.tf_list_ : list[str] = list(
    #         tx for tx in all_tf_set if tx in self.gene_map_
    #     )
    #     self.ntfs_ : int = len(self.tf_list_)
    #     self.tf_map_ : dict[str, int]  = idx_dict(self.tf_list_)
    #     self.tf_indices_ : list[int] = list(
    #         self.gene_map_[tfx] for tfx in self.tf_list_
    #     )
        
    def __pre_process(self, sargs: SCDataArgs):
        # QC Metrics
        self.adata_.var["mt"] = self.adata_.var_names.str.startswith("MT-")
        self.adata_.var["rb"] = self.adata_.var_names.str.startswith("RPS")
        sc.pp.calculate_qc_metrics(
            self.adata_,
            qc_vars=["mt", "rb"],
            percent_top=None,
            log1p=False,
            inplace=True
        )
        #  Subset cells,
        ncells = self.adata_.shape[0]
        nsub_cells = sargs.nsub_cells
        if nsub_cells > 0 and nsub_cells < ncells:
            np.random.seed(0)
            self.adata_ = self.adata_[np.random.choice(ncells, nsub_cells), :]
        # Normalization
        sc.pp.normalize_total(self.adata_, target_sum=int(1e4))
        sc.pp.log1p(self.adata_)
        # Highly variable genes
        sc.pp.highly_variable_genes(self.adata_, n_top_genes=sargs.ntop_genes)
        if sargs.select_hvg:
            self.adata_.raw = self.adata_
            # Restrict to highly variable genes
            self.adata_ = self.adata_[:, self.adata_.var.highly_variable]  # pyright:ignore[reportAttributeAccessIssue]
            if sargs.scale_regress:
                # Regress and Scale data
                sc.pp.regress_out(self.adata_,
                                  keys=["total_counts", "pct_counts_mt"])
                sc.pp.scale(self.adata_, max_value=10)
        else:
            print("Not selecting highly_variable_genes")

    @property
    @t.override
    def exp_matrix(self) -> NDFloatArray:
        return self.adata_.X  # pyright:ignore[reportReturnType]

    @property
    @t.override
    def tf_exp_matrix(self) -> NDFloatArray:
        return self.tf_adata_.X  # pyright: ignore[reportReturnType]

    def save(self, scdata_file: str):
        self.adata_.write_h5ad(scdata_file)
