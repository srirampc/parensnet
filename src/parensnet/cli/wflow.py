import argparse
import logging
import time, itertools
import typing as t
import h5py
import dask.bag as db
import numpy as np, anndata as an

from collections.abc import Iterable
from devtools import pformat
from codetiming import Timer

from ..types import (
    DiscretizerMethod, IDDTuple, LogBase, DataPair,
    NDIntArray, NDFloatArray, NPDType 
)
from ..comm_interface import IDComm
from ..workflow.util import InputArgs, iddtuple_to_h5
from ..mvim.context import get_clr_weight

from ..mvim.rv import RVNode, RVNodePair, MRVNodePairs
from ..mvim.misi import MISIDataH5

#

def _log():
    return logging.getLogger(__name__)

def compute_nodes(
    data_list: list[NDFloatArray],
    dmethod: DiscretizerMethod = "bayesian_blocks",
    ftype: NPDType = np.float32,
    itype: NPDType = np.int32
) -> list[RVNode]:
    def create_node(x: NDFloatArray):
        return RVNode(x, None, dmethod, ftype, itype)
    return db.from_sequence(data_list).map(
        create_node
        # lambda x: RVNode(x, None, dmethod, ftype, itype)
        ).compute()


#def subset_puc(i: int, j: int, h5_file: str, by_list: Iterable[int]):
def subset_puc(subset_desc: tuple[int, int, str, Iterable[int]]):
    i, j, h5_file, by_list = subset_desc
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
) -> tuple[list[list[RVNodePair]], dict[int, t.Any]]:
    npartitions = len(sbegin)
    batch_pairs = [[] for _ in range(npartitions)]
    batch_pairs_time = {x: 0.0 for x in range(npartitions)}
    for ix, (bstart, bend)  in enumerate(zip(sbegin, sends)):
        batch_start = time.time()
        print(f"Start Batch {ix} : Range {bstart} : {bend}")
        batch_pairs[ix] = db.from_sequence(plist[bstart:bend]).map(
            #lambda x: subset_puc(x[0], x[1], x[2], x[3])
            subset_puc
        ).compute()
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_pairs_time[ix] = batch_time
        print(f"End Batch {ix} : Range {bstart} : {bend} :: Time : {batch_time}")
    return batch_pairs, batch_pairs_time


def prep_pair_batches(
    data_list: list[NDFloatArray],
    nodes: list[RVNode],
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
    plist: list[tuple[RVNode, RVNode, NDFloatArray, NDFloatArray]],
    sbegin: list[int],
    sends: list[int],
    lbase: LogBase,
    ftype: NPDType,
    itype: NPDType,
) -> tuple[list[list[RVNodePair]], dict[int, t.Any]]:
    npartitions = len(sbegin)
    batch_pairs = [[] for _ in range(npartitions)]
    batch_pairs_time = {x: 0.0 for x in range(npartitions)}
    def build_rv_node(x: tuple[RVNode, RVNode, NDFloatArray, NDFloatArray]):
        return RVNodePair.from_nodes(
                (x[0], x[1]), (x[2], x[3]), lbase, ftype, itype
            )
    #
    for ix, (bstart, bend)  in enumerate(zip(sbegin, sends)):
        batch_start = time.time()
        print(f"Start Batch {ix} : Range {bstart} : {bend}")
        batch_pairs[ix] = db.from_sequence(plist[bstart:bend]).map(
            build_rv_node
        ).compute()
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_pairs_time[ix] = batch_time
        print(f"End Batch {ix} : Range {bstart} : {bend} :: Time : {batch_time}")
    return batch_pairs, batch_pairs_time
 

def merge_batches(
    _sbegin: list[int],
    _sends: list[int],
    batch_pairs: list[list[RVNodePair]],
) -> list[RVNodePair]:
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
    nodes: list[RVNode],
    node_pairs: list[RVNodePair],
    nobs: int,
    nvars: int,
):
    np_keys = [(i, j) for i,j in itertools.combinations(range(nvars), 2)]
    npairs_dict = {(i, j): npx for (i, j), npx in zip(np_keys, node_pairs)}
    return  MRVNodePairs(
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
        self.nodes: list[RVNode] = []
        self.node_pairs: list[RVNodePair] = []

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

def misi_workflow(mxargs: InputArgs):
    dflow = DaskWorkflow.from_h5ad(
        mxargs.h5ad_file, mxargs.nobs, mxargs.nvars, mxargs.nroundup
    )
    dflow.build_nodes()
    dflow.build_node_pairs()
    # TODO:: save the node pairs as misi

    # exp_data = an.read_h5ad(mxargs.h5ad_file)
    # adata = np.round(exp_data.X, mxargs.nroundup)
    # ndata, nvariables = adata.shape
    # data_list = [adata[:, ix] for ix in range(adata.shape[1])]
    # pair_list = [(i, j) for i, j in itertools.combinations(range(nvariables), 2)]

class PUCUnionWorkflow:
    mxargs: InputArgs
    union_mat: NDFloatArray
    dtype: NPDType
    int_dtype: NPDType
    nindices: NDIntArray

    @Timer(name="PUCUnionWorkflow::__init__", logger=None)
    def __init__(self, mxargs: InputArgs) -> None:
        self.mxargs = mxargs
        first_file = mxargs.sub_net_files[0]
        #
        with h5py.File(first_file) as hfptr:
            rindex: NDIntArray = hfptr["/data/index"] # pyright: ignore[reportAssignmentType]
            rpuc: NDFloatArray = hfptr["/data/puc"] # pyright: ignore[reportAssignmentType]
            self.nindices = rindex[:]
            self.dtype = rpuc.dtype
            self.int_dtype = rindex.dtype
        #
        self.union_mat = np.zeros((mxargs.nvars, mxargs.nvars),
                                  self.dtype)
 
    @Timer(name="PUCUnionWorkflow::run", logger=None)
    def run(self):
        if mxargs.sub_net_files is None:
            return
        #
        for puc_file in mxargs.sub_net_files:
            with h5py.File(puc_file) as hfptr:
                rindex: NDIntArray = hfptr["/data/index"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
                rpuc: NDFloatArray = hfptr["/data/puc"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
                self.union_mat[rindex[:, 0], rindex[:, 1]] += rpuc
                self.union_mat[rindex[:, 1], rindex[:, 0]] += rpuc
        nfiles = len(mxargs.sub_net_files)
        self.union_mat = self.union_mat/np.float64(nfiles).astype(self.dtype)
        union_puc = self.union_mat[self.nindices[:, 0], self.nindices[:, 1]]   
        iddtuple_to_h5(
            IDDTuple(self.nindices, union_puc),
            self.mxargs.puc_file,
            DataPair[str]("index", "puc"),
            "data"
        )

class ContextWorkflow:
    mxargs: InputArgs
    puc_mat: NDFloatArray
    nindices: NDIntArray
    dtype: NPDType
    int_dtype: NPDType

    @Timer(name="ContextWorkflow::__init__", logger=None)
    def __init__(self, mxargs: InputArgs) -> None:
        self.mxargs = mxargs

        with h5py.File(self.mxargs.puc_file) as hfptr:
            rindex: NDIntArray = hfptr["/data/index"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            rpuc: NDFloatArray = hfptr["/data/puc"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            self.dtype = rpuc.dtype
            self.int_dtype = rindex.dtype
            self.nindices = rindex[:]
            self.puc_mat = np.zeros((mxargs.nvars, mxargs.nvars),
                                    self.dtype)
            self.puc_mat[rindex[:, 0], rindex[:, 1]] = rpuc
            self.puc_mat[rindex[:, 1], rindex[:, 0]] = rpuc

    def compute_context(self) -> IDDTuple:
        bats_pidc = IDDTuple(-np.ones((1,2), dtype=np.int32),
                                 np.zeros(1, np.float32))
        px_pidc = np.zeros(self.nindices.shape[0], self.dtype)
        rindex = 0
        for rx, cx in self.nindices:
            px_pidc[rindex] = get_clr_weight(self.puc_mat, rx, cx)
            rindex += 1
        bats_pidc = IDDTuple(self.nindices, px_pidc)
        return bats_pidc

    @Timer(name="ContextWorkflow::run", logger=None)
    def run(self):
        pidc_tuple = self.compute_context()
        #
        iddtuple_to_h5(pidc_tuple, self.mxargs.pidc_file,
                       DataPair[str]("index", "pidc"), "data")
        


def main(mxargs: InputArgs):
    comm_ifx  = IDComm()
    print(
        pformat(
            mxargs.model_dump(exclude=["h5_fptr"])  # pyright: ignore[reportArgumentType]
        )
    )
    for rxmode in mxargs.mode:
        match rxmode:
            case 'misi':
                misi_workflow(mxargs)
            case 'puc_union':
                PUCUnionWorkflow(mxargs).run()
            case 'puc2pidc':
                ContextWorkflow(mxargs).run()
            case _:
                print("not implmented")
    if mxargs.enable_timers:
        comm_ifx.barrier()
        comm_ifx.log_profile_summary(_log(), logging.DEBUG)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       prog="misi_data",
       description="Generate GRNs w. PIDC for Single Cell Data"
   )
   parser.add_argument(
       "yaml_file",
       default="./config/networks/pbmc20k_5k.yaml",
       help=f"Yaml Input file with a given configuration."
   )
   run_args = parser.parse_args()
   print("Run Arguments :: ", run_args)
   with InputArgs.from_yaml(run_args.yaml_file) as mxargs:
       main(mxargs)
