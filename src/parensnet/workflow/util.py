import logging
import os
import typing as t

import h5py
import numpy as np
import pydantic
import anndata as an

from enum import Enum
from codetiming import Timer, TimerConfig

from ..types import (
    NDIntArray, NDFloatArray, 
    DiscretizerMethod, LogBase, DataPair, IDDTuple, NPDType
)
from ..util import parse_yaml, create_h5ds
from ..util import block_range, block_2d_all_ranges, diag_distribution
from ..comm_interface import CommInterface


LogLevel : t.TypeAlias = t.Literal[
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
]

RunMode : t.TypeAlias = t.Literal[
    'misi',
    'sampled_puc_pairs',
    'samples_ranges',
    'samples_input', 
    'samples_lmr_ranges',
    'samples_lmr_input', 
    'puc_ranges',
    'puc_lmr',
    'puc2pidc',
    'puc_union',
    'cluster_union',
    'cluster_lmr_union',
]

LOG_LEVEL_MAP: dict[LogLevel, int] = {
    'NOTSET': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class PUCMethod(Enum):
    REDUNDANCY = 0
    LMR = 1

def getenv(env_name: str):
    if env_name in os.environ:
        return os.environ[env_name]
    return 'N/A'

@Timer(name="collect_iddtuples", logger=None)
def collect_iddtuples(
    comm_ifx: CommInterface,
    local_idds:list[IDDTuple]
) -> IDDTuple:
    pindices = np.vstack([px.first for px in local_idds])
    pvalues = np.concatenate([px.second for px in local_idds])
    # 
    all_indices: None | list[NDIntArray] = comm_ifx.collect_objects_at_root(pindices)
    all_values : None | list[NDFloatArray] = comm_ifx.collect_objects_at_root(pvalues)
    del pindices
    del pvalues
    full_indices : NDIntArray = np.array(())
    full_values : NDFloatArray = np.array(())
    if comm_ifx.rank == 0:
        if (all_indices is not None) and (all_values is not None):
            full_indices = np.vstack(all_indices) 
            full_values = np.concatenate(all_values)
    comm_ifx.barrier()
    return IDDTuple(full_indices, full_values)

def iddtuple_to_h5(
    idd_tuple: IDDTuple,
    output_file: str,
    names: DataPair[str],
    group_name: str = "data"
):
    with h5py.File(output_file, 'w') as fptr:
        data_grp = fptr.create_group(group_name)
        create_h5ds(data_grp, names.first, idd_tuple.first)
        create_h5ds(data_grp, names.second, idd_tuple.second)

def init_range_input(
    st_ranges: tuple[range, range],
    int_dtype: NPDType
) -> NDIntArray:
    row_range, col_range = st_ranges
    nrpairs = 0
    for rx in row_range:
        for cx in col_range:
            if rx >= cx:
                continue
            nrpairs += 1
    nrindex = 0
    pairs_mtx = np.zeros((nrpairs, 2), int_dtype)
    for rx in row_range:
        for cx in col_range:
            if rx >= cx:
                continue
            pairs_mtx[nrindex] = (rx, cx)
            nrindex += 1
    return pairs_mtx

class InputArgs(pydantic.BaseModel):
    h5ad_file: str = "./data/pbmc20k/adata.20k.5k.h5ad"
    nodes_pickle: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.nodes.pickle"
    nodes_pairs_pickle: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.node_pairs.pickle"
    misi_data_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.np_par_misidata.h5"
    puc_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.par_puc.h5"
    pidc_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.par_pidc.h5"
    samples_file: str | None = None
    sub_net_files: list[str] | None = None
    wflow_dir: str = '/tmp/'
    gene_id_col: str = "gene_ids"
    save_nodes: bool = False
    save_node_pairs: bool = False
    enable_timers: bool = True
    nroundup: int = 4
    nrounds: int = 8
    nsamples: int = 200
    nobs: int = 0
    nvars: int = 0
    npairs: int = 0
    disc_method: DiscretizerMethod = 'bayesian_blocks'
    tbase: LogBase = '2'
    mode: list[RunMode] = ['misi']
    h5_fptr: h5py.File | None = None
    log_level: LogLevel = 'DEBUG'
    quantile_filter: float | None = None

    class Config:
        arbitrary_types_allowed:bool=True

    @classmethod
    def from_yaml(cls, yaml_file: str):
        cfg_dict = parse_yaml(yaml_file)
        run_args = cls.model_validate(cfg_dict)
        return run_args

    def __init__(self, /, **data: t.Any) -> None:
        super().__init__(**data)
        #
        hfx = an.read_h5ad(self.h5ad_file, 'r')
        nobs, nvars = hfx.shape
        if self.nobs == 0:
            self.nobs = nobs
        if self.nvars == 0:
            self.nvars = nvars
        if self.npairs == 0:
            self.npairs = (self.nvars * (self.nvars - 1)) // 2
        #
        logging.basicConfig(level=self.get_level())
        if self.enable_timers:
            TimerConfig.enable_timers()
        else:
            TimerConfig.disable_timers()

    def get_level(self):
        return LOG_LEVEL_MAP[self.log_level]

    def load_range_data(
        self,
        data_range: range
    ) -> NDFloatArray:
        return np.round( # pyright: ignore[reportCallIssue]
            self.h5_fptr["/X"][:self.nobs, data_range], # pyright: ignore[reportArgumentType, reportIndexIssue]
            self.nroundup
        )

    def open(self):
        self.h5_fptr = h5py.File(self.h5ad_file)
        return self

    def close(self):
        if self.h5_fptr:
            self.h5_fptr.close()

    def __enter__(self):
        return self.open()

    def __exit__(
        self,
        exc_type: object,
        exc_value: IOError,
        exc_traceback: object
    ):
        self.close()

class WorkDistributor(pydantic.BaseModel):
    mxargs: InputArgs
    nproc: int = 1
    rank: int = 0
    #
    var_blocks: list[range] = []
    pair_blocks1d: list[range] = []
    #
    pair_blocks2d: list[list[tuple[range, range]]] = []
    pair_batches2d: list[list[tuple[int, int]]] = []
    pair_batches2d_ranges: list[list[tuple[range, range]]] = []
    pair_nbatches: int = 0
    #

    class Config:
        arbitrary_types_allowed:bool=True

    def __init__(self, mxargs:InputArgs, nproc: int, rank: int, **data: t.Any) -> None:
        super().__init__(mxargs=mxargs, nproc=nproc, rank=rank, **data)
        # nodes = construct_all_nodes(hfx, nvariables, nvariables, mxargs.nroundup)
        #
        self.var_blocks = [range(0, mxargs.nvars)]
        if (self.nproc > 1) and (mxargs.nvars > self.nproc):
            self.var_blocks = [
                block_range(rx, self.nproc, mxargs.nvars)
                for rx in range(self.nproc)
            ]
        #
        self.pair_blocks1d= [range(0, self.mxargs.npairs)]
        if (self.nproc > 1) and (self.mxargs.npairs > self.nproc):
            self.pair_blocks1d = [
                block_range(rx, self.nproc, mxargs.npairs)
                for rx in range(self.nproc)
            ]
        #
        self.pair_blocks2d = block_2d_all_ranges(
            self.nproc, self.mxargs.nvars
        )
        self.pair_batches2d = diag_distribution(self.nproc)
        self.pair_nbatches = len(self.pair_batches2d)
        self.pair_batches2d_ranges = [
            [
                self.range_for(bid, pid, prow, pcol)
                for pid,(prow, pcol) in enumerate(batch_lst)
            ]
            for bid, batch_lst in enumerate(self.pair_batches2d)
        ]

    def range_for(self, bid: int, pid: int, prow: int, pcol: int):
        if self.nproc % 2 == 0 and bid == self.pair_nbatches - 1:
            row_range, col_range = self.pair_blocks2d[prow][pcol]
            rowr_mid = row_range.start + (
                (row_range.stop - row_range.start) // 2
            )
            if pid * 2 < self.nproc:
                return (
                    range(row_range.start, rowr_mid),
                    col_range
                )
            else:
                return (
                    range(rowr_mid, row_range.stop),
                    col_range
                )

        else:
            return self.pair_blocks2d[prow][pcol]

    @Timer(name="load_pair_block_data", logger=None)
    def load_pair_block_data(
        self,
        batch_id:int,
        pid:int,
    ) -> tuple[NDFloatArray, NDFloatArray]:
        row_range, col_range = self.pairs_blocks2d_ranges(batch_id, pid)
        # world_comm.log_comm(
        #     _log(), logging.DEBUG, f"Block :: {row_range} ;{col_range} "
        # )
        row_data = self.mxargs.load_range_data(row_range)
        col_data = self.mxargs.load_range_data(col_range)
        return row_data, col_data


    def pair_row_range(self, batch_id: int, pid: int):
        # curr_batch = self.pair_batches2d[batch_id]
        # block_row, block_col = curr_batch[self.rank]
        # return self.pair_blocks2d[block_row][block_col][0]
        return self.pair_batches2d_ranges[batch_id][pid][0]

    def pair_col_range(self, batch_id: int, pid:int):
        # world_comm = default_comm()
        # curr_batch = self.pair_batches2d[batch_id]
        # block_row, block_col = curr_batch[self.rank]
        # return self.pair_blocks2d[block_row][block_col][1]
        return self.pair_batches2d_ranges[batch_id][pid][1]

    def var_range(self, pid: int):
        return self.var_blocks[pid]

    def pair_blocks1d_range(self, pid: int):
        return self.var_blocks[pid]

    def pairs_blocks2d_ranges(self, bid: int, pid: int):
        return self.pair_batches2d_ranges[bid][pid]
