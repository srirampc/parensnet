import time, itertools
import typing as t
import pydantic
import dask.delayed, dask.bag as db
import numpy as np, anndata as an

from collections.abc import Iterable
from .pidc import PIDCNode, PIDCPair, PIDCPairListData
from .misi import MISIData, MISIDataH5
from .types import DiscretizerMethod, LogBase, NDFloatArray, NPDType
from .util import NDIntArray
#

class InputArgs(pydantic.BaseModel):
    h5ad_file: str = "./data/pbmc20k/adata.20k.5k.h5ad"
    misi_data: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.misidata.h5"
    puc_output: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.puc.h5"
    nrounds: int = 8
    nsamples: int = 200
    nvariables: int = 5000 
    npairs: int | None = None


def compute_nodes(
    data_list: list[NDFloatArray],
    dmethod: DiscretizerMethod = "bayesian_blocks",
    ftype: NPDType = np.float32,
    itype: NPDType = np.int32
) -> list[PIDCNode]:
    return db.from_sequence(data_list).map(
        lambda x: PIDCNode(x, None, dmethod, ftype, itype)
        ).compute()


def subset_puc(i: int, j: int, h5_file: str, by_list: Iterable[int]):
    with MISIDataH5(h5_file) as misihx:
        return misihx.accumulate_redundancies(i, j, by_list)


def prep_subc_pair_batches(
    h5_file: str,
    sub_indices: NDIntArray,
    nvars: int,
    npartitions:int,
):
    plist = [
        (i, j, h5_file, sub_indices)
        for i,j in itertools.combinations(range(nvars), 2)
    ]
    npairs = len(plist)
    plist_arange = np.arange(npairs, dtype=np.int32)
    arange_splits = np.array_split(plist_arange, npartitions)
    sbegin = [int(x[0]) for x in arange_splits]
    sends = sbegin[1:] + [npairs]
    assert npartitions == len(sbegin)
    assert npartitions == len(sends)
    return plist, sbegin, sends


def run_subc_pair_batches(
    plist: list[tuple[int, int, str, Iterable[int]]],
    sbegin: list[int],
    sends: list[int],
) -> tuple[list[list[PIDCPair]], dict[int, t.Any]]:
    npartitions = len(sbegin)
    batch_pairs = [[] for _ in range(npartitions)]
    batch_pairs_time = {x: 0.0 for x in range(npartitions)}
    for ix, (bstart, bend)  in enumerate(zip(sbegin, sends)):
        batch_start = time.time()
        print(f"Start Batch {ix} : Range {bstart} : {bend}")
        batch_pairs[ix] = db.from_sequence(plist[bstart:bend]).map(
            lambda x: subset_puc(x[0], x[1], x[2], x[3])
        ).compute()
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_pairs_time[ix] = batch_time
        print(f"End Batch {ix} : Range {bstart} : {bend} :: Time : {batch_time}")
    return batch_pairs, batch_pairs_time


def prep_pair_batches(
    data_list: list[NDFloatArray],
    nodes: list[PIDCNode],
    nvars: int,
    npartitions:int,
):
    plist = [
        (nodes[i], nodes[j], data_list[i], data_list[j])
        for i,j in itertools.combinations(range(nvars), 2)
    ]
    npairs = len(plist)
    plist_arange = np.arange(npairs, dtype=np.int32)
    arange_splits = np.array_split(plist_arange, npartitions)
    sbegin = [int(x[0]) for x in arange_splits]
    sends = sbegin[1:] + [npairs]
    assert npartitions == len(sbegin)
    assert npartitions == len(sends)
    return plist, sbegin, sends

def run_pair_batches(
    plist: list[tuple[PIDCNode, PIDCNode, NDFloatArray, NDFloatArray]],
    sbegin: list[int],
    sends: list[int],
    lbase: LogBase,
    ftype: NPDType,
    itype: NPDType,
) -> tuple[list[list[PIDCPair]], dict[int, t.Any]]:
    npartitions = len(sbegin)
    batch_pairs = [[] for _ in range(npartitions)]
    batch_pairs_time = {x: 0.0 for x in range(npartitions)}
    for ix, (bstart, bend)  in enumerate(zip(sbegin, sends)):
        batch_start = time.time()
        print(f"Start Batch {ix} : Range {bstart} : {bend}")
        batch_pairs[ix] = db.from_sequence(plist[bstart:bend]).map(
            lambda x: PIDCPair.from_nodes(
                (x[0], x[1]), (x[2], x[3]), lbase, ftype, itype
            )
        ).compute()
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_pairs_time[ix] = batch_time
        print(f"End Batch {ix} : Range {bstart} : {bend} :: Time : {batch_time}")
    return batch_pairs, batch_pairs_time
 

def merge_batches(
    sbegin: list[int],
    sends: list[int],
    batch_pairs: list[list[PIDCPair]],
) -> list[PIDCPair]:
    # npairs = sum([len(px) for px in batch_pairs])
    # node_pairs = [None for _ in range(npairs)]
    # for ix, (bstart, _)  in enumerate(zip(sbegin, sends)):
    #      ridx = bstart
    #      for pairn in batch_pairs[ix]:
    #          node_pairs[ridx] = pairn
    #          ridx += 1
    # return node_pairs
    return [pairn for batch_list in batch_pairs for pairn in batch_list]


def pldata_from_pairs(
    nodes: list[PIDCNode],
    node_pairs: list[PIDCPair],
    nobs: int,
    nvars: int,
):
    np_keys = [(i, j) for i,j in itertools.combinations(range(nvars), 2)]
    npairs_dict = {(i, j): npx for (i, j), npx in zip(np_keys, node_pairs)}
    return  PIDCPairListData(
        nobs=nobs,
        nvars=nvars,
        npairs=len(node_pairs),
        nodes=nodes,
        node_pairs=npairs_dict
    )

class DaskWorkflow:
    def __init__(
        self,
        np_data: NDFloatArray,
        nobs:int | None,
        nvars: int | None,
        nroundup:int = 4,
        npartitions: int = 100,
    ):
        self.adata: NDFloatArray = np.round(np_data, nroundup)
        self.rftype: NPDType = np.float32
        self.ritype : NPDType = np.int32
        self.tbase: LogBase = '2'
        #
        self.ndata: int = 0
        self.nvariables: int = 0
        self.ndata, self.nvariables = self.adata.shape
        if nobs is not None:
            self.ndata = nobs
        if nvars is not None:
            self.nvariables = nvars
        self.npartitions: int =  npartitions
        self.dask_run_times: dict[str, t.Any] = {}
        self.cprun_times: dict[str, t.Any] = {}
        #
        self.data_list: list[NDFloatArray] = [
            self.adata[:, ix] for ix in range(self.nvariables)
        ]
        #
        self.nodes: list[PIDCNode] = []
        self.node_pairs: list[PIDCPair] = []

    @classmethod
    def from_h5ad(
        cls,
        h5ad_file: str,
        nobs:int | None,
        nvars: int | None,
        nroundup:int = 4,
        npartitions: int = 100,
    ):
        exp_data = an.read_h5ad(h5ad_file)
        return cls(exp_data.X, nroundup, nobs, nvars, npartitions)  # pyright: ignore[ reportArgumentType]

    def __compute_pair_batches(self):
        start_time = time.time()
        plist, sbegin, sends = prep_pair_batches(
            self.data_list, self.nodes, self.nvariables, self.npartitions
        )
        end_time = time.time()
        self.cprun_times['pairs_prep'] = (end_time - start_time, start_time, end_time)
        start_time = time.time()
        batch_pairs, batch_pairs_time = run_pair_batches(
            plist, sbegin, sends, self.tbase, self.rftype, self.ritype
        )
        end_time = time.time()
        self.cprun_times['pairs_build'] = (end_time - start_time, start_time, end_time)
        start_time = time.time()
        self.node_pairs = merge_batches(sbegin, sends, batch_pairs)
        end_time = time.time()
        self.cprun_times['pairs_merge'] = (end_time - start_time, start_time, end_time)
        self.cprun_times['batch_times'] = batch_pairs_time
        self.dask_run_times = self.dask_run_times | self.cprun_times

    def build_nodes(self):
        #
        start_tx = time.time()
        self.nodes = compute_nodes(self.data_list)
        end_tx = time.time()
        self.dask_run_times['nodes'] = (end_tx - start_tx, start_tx, end_tx)

    def build_node_pairs(self):
        #
        self.__compute_pair_batches()
    
    def construct_plist_data(self):
        #
        start_tx = time.time()
        pdpairsl = pldata_from_pairs(
            self.nodes,
            self.node_pairs,
            self.ndata,
            self.nvariables
        )
        end_tx = time.time()
        self.dask_run_times['pairs_list'] = (end_tx - start_tx, start_tx, end_tx) 
        return pdpairsl

# exp_data = an.read_h5ad(H5AD_FILE)
# adata = np.round(exp_data.X, 4) # pyright: ignore[reportCallIssue, reportArgumentType]
# ndata, nvariables = adata.shape
# data_list = [adata[:, ix] for ix in range(adata.shape[1])]
# pair_list = [(i, j) for i, j in itertools.combinations(range(nvariables), 2)]

