import argparse
import logging
import os
import pickle
import typing as t
#
import pandas as pd
import anndata as an
import h5py
import numpy as np
import pydantic
#
from devtools import pformat
from codetiming import Timer, TimerConfig

from parensnet.distributed import MISIDataDistributed
#
from .types import NDFloatArray, NDIntArray, DataPair, DataPair2, NPDType
from .types import DiscretizerMethod, LogBase, NPFloat
from .comm_interface import CommInterface, NPArray, default_comm
from .context import get_clr_weight
from .pidc import PIDCNode, PIDCPair, PIDCPairData, PIDCPairListData
from .misi import MISIData, MISIDataH5, MISIPair, MISIRangePair 
from .util import parse_yaml, create_h5ds, triu_pair_to_index, triu_index_to_pair
from .util import block_range, block_2d_all_ranges, diag_distribution

PUCTuple: t.TypeAlias = tuple[int, int, NPFloat]
IDDTuple: t.TypeAlias =  DataPair2[NDIntArray, NDFloatArray] # tuple[NDFloatArray, NDIntArray]

LogLevel : t.TypeAlias = t.Literal[
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
]

RunMode : t.TypeAlias = t.Literal[
    'sampled_puc_pairs', 'samples_ranges', 'samples_input', 
    'misi', 'puc_ranges', 'puc_lmr', 'puc2pidc'
]

LOG_LEVEL_MAP: dict[LogLevel, int] = {
    'NOTSET': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def _log():
    return logging.getLogger(__name__)

def getenv(env_name: str):
    if env_name in os.environ:
        return os.environ[env_name]
    return 'N/A'

def write_iddtuples(
    idd_tuple: IDDTuple,
    output_file: str,
    names: DataPair[str],
    group_name: str = "data"
):
    with h5py.File(output_file, 'w') as fptr:
        data_grp = fptr.create_group(group_name)
        create_h5ds(data_grp, names.first, idd_tuple.first)
        create_h5ds(data_grp, names.second, idd_tuple.second)

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

def init_range_input(st_ranges: tuple[range, range],  int_dtype: NPDType):
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
    misi_data_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.np_par_misidata.h5"
    puc_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.par_puc.h5"
    pidc_file: str = "/localscratch/schockalingam6/tmp/adata.20k.5k.par_pidc.h5"
    samples_file: str | None = None
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
            self.npairs = (self.nvars * self.nvars) // 2
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


class FullPUCWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor

    def __init__(self, mxargs: InputArgs, wdistr: WorkDistributor) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr

    # def _init_range_input(self, misi_range: MISIRangePair):
    #     row_range, col_range = misi_range.st_ranges
    #     nrpairs = 0
    #     for rx in row_range:
    #         for cx in col_range:
    #             if rx >= cx:
    #                 continue
    #             nrpairs += 1
    #     nrindex = 0
    #     pairs_mtx = np.zeros((nrpairs, 2), misi_range.int_dtype())
    #     for rx in row_range:
    #         for cx in col_range:
    #             if rx >= cx:
    #                 continue
    #             pairs_mtx[nrindex] = (rx, cx)
    #             nrindex += 1
    #     return pairs_mtx

    @Timer(name="FullPUCWorkflow::_compute_lmr_puc_with_misi_range", logger=None)
    def _compute_lmr_puc_with_misi_range(
        self,
        misi_range:MISIRangePair,
    ) -> list[PUCTuple]:
        pairs_list = init_range_input(misi_range.st_ranges,
                                      misi_range.int_dtype()) 
        pairs_puc = [(-1,-1,np.float32(0.0))] * len(pairs_list)
        rindex = 0
        for rx, cx in pairs_list:
            s_puc = misi_range.compute_lmr_puc(int(rx), int(cx))
            pairs_puc[rindex] = (rx, cx, np.float32(s_puc))
            rindex += 1
        return pairs_puc

    @Timer(name="FullPUCWorkflow::_batch_misi_range", logger=None)
    def _batch_misi_range(self, bidx:int, pid:int, mi_cache: NDFloatArray | None):
        st_ranges = self.wdistr.pairs_blocks2d_ranges(bidx, pid)
        return MISIRangePair(mxargs.misi_data_file, st_ranges, mi_cache, True)
    
    # @Timer(name="FullPUCWorkflow::_batch_misi_load_ds", logger=None)
    # def _batch_misi_load_ds(self, misi_range: MISIRangePair):
    #     misi_range.load_ds()
    #     return misi_range
    
    @Timer(name="FullPUCWorkflow::_compute_with_ranges", logger=None)
    def _compute_with_ranges(self, mi_cache: NDFloatArray | None) -> list[PUCTuple]:
        world_comm = default_comm()
        bsamples_puc: list[list[PUCTuple] | None] = [None] * self.wdistr.pair_nbatches
        for bidx in range(self.wdistr.pair_nbatches):
            row_range, col_range = self.wdistr.pairs_blocks2d_ranges(
                bidx,
                world_comm.rank
            )
            #
            misi_range = self._batch_misi_range(bidx, world_comm.rank, mi_cache)
            # misi_range = self._batch_misi_load_ds(misi_range)
            #
            world_comm.log_comm(
                _log(), logging.DEBUG,
                f"B{bidx} :: R: {row_range} ; C: {col_range}"
            )
            #
            bsamples_puc[bidx] = self._compute_lmr_puc_with_misi_range(
                misi_range,
            )
            del misi_range
            # if bidx == 2:
            #     break
        return [px for gplx in bsamples_puc if gplx is not None for px in gplx]

    @Timer(name="FullPUCWorkflow::_load_mi_cache", logger=None)
    def _load_mi_cache(self) -> NDFloatArray:
        with h5py.File(self.mxargs.misi_data_file, 'r') as fptr:
            data_grp : h5py.Group = t.cast(h5py.Group, fptr["data"])
            return data_grp["mi"][:]  # pyright: ignore[reportReturnType,reportIndexIssue]

    @Timer(name="FullPUCWorkflow::run_with_ranges", logger=None)
    def run_with_ranges(self) -> list[PUCTuple]:
        mi_cache = self._load_mi_cache()
        slist = self._compute_with_ranges(mi_cache)
        #
        return slist


class SPUCWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface

    def __init__(self, mxargs: InputArgs, wdistr: WorkDistributor) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()

    @Timer(name="SPUCWorkflow::generate_samples", logger=None)
    def generate_samples(self):
        rsamples = np.array([])
        if self.comm.rank == 0:
            rsamples = np.random.choice(
                range(mxargs.nvars),
                self.mxargs.nsamples*self.mxargs.nrounds
            )
        rsamples = self.comm.broadcast(rsamples)
        return np.array_split(rsamples, self.mxargs.nrounds)
        # world_comm.log_comm(_log(), logging.DEBUG, ",".join([str(x) for x in r0]))

    @Timer(name="SPUCWorkflow::_generate_pair_list", logger=None)
    def _generate_pair_list(
        self,
        misihx: MISIDataH5,
    ) -> list[tuple[int, int]]:
        nprange = block_range(
            self.comm.rank, self.comm.size, self.mxargs.npairs
        )
        pair_list = [
            (i, j) for i,j in misihx.h5_fptr["/data/pair_index"][nprange] # pyright: ignore[reportIndexIssue]
        ]
        self.comm.log_comm(_log(), logging.DEBUG, f"{str(nprange)}" )
        return pair_list

    @Timer(name="SPUCWorkflow::_compute_with_pairs", logger=None)
    def _compute_with_pairs(
        self,
        misihx: MISIDataH5,
        pair_lst: list[tuple[int,int]],
        samples_lst: list[NDIntArray]
    ):
        samples_puc = np.zeros(len(pair_lst), misihx.float_dtype())
        for px, (i, j) in enumerate(pair_lst):
            samples_puc[px] = 0
            mpair = MISIPair.from_misidata(misihx, int(i), int(j))
            s_puc = np.array(
                [
                    mpair.accumulate_redundancies(
                        int(mpair.idx.first),
                        int(mpair.idx.second),
                        by_nodes
                    )
                    for by_nodes in samples_lst
                ],
                misihx.float_dtype()
            )
            samples_puc[px] = np.mean(s_puc)
            del mpair
        return samples_puc

    @Timer(name="SPUCWorkflow::puc_to_h5", logger=None)
    def puc_to_h5(self, samples_puc: NPArray):
        gpuc_list = self.comm.gather_at_root_by_snd_rcv(samples_puc)
        if self.comm.rank != 0:
            return
        if not gpuc_list:
            return
        gpuc_samples = np.concatenate(gpuc_list)
        self.comm.log_at_root(_log(), logging.DEBUG, f"{gpuc_samples.shape}" )
        import h5py
        with h5py.File(self.mxargs.puc_file, 'w') as fptr:
            data_grp = fptr.create_group("data")
            create_h5ds(data_grp, "samples_puc", gpuc_samples)

    def _init_range_input(self, misi_range: MISIRangePair):
        row_range, col_range = misi_range.st_ranges
        nrpairs = 0
        for rx in row_range:
            for cx in col_range:
                if rx >= cx:
                    continue
                nrpairs += 1
        nrindex = 0
        pairs_mtx = np.zeros((nrpairs, 2), misi_range.int_dtype())
        for rx in row_range:
            for cx in col_range:
                if rx >= cx:
                    continue
                pairs_mtx[nrindex] = (rx, cx)
                nrindex += 1
        return pairs_mtx

    @Timer(name="SPUCWorkflow::_compute_with_misi_range", logger=None)
    def _compute_with_misi_range(
        self,
        misi_range:MISIRangePair,
        rsamples: list[NDIntArray]
    ) -> IDDTuple:
        pairs_mtx = self._init_range_input(misi_range)
        pairs_puc = np.zeros(len(pairs_mtx), misi_range.float_dtype())
        for rindex, (rx, cx) in enumerate(pairs_mtx):
            s_puc = np.zeros(len(rsamples), misi_range.float_dtype())
            for sidx, by_nodes in enumerate(rsamples):
                s_puc[sidx] = misi_range.accumulate_redundancies(
                    int(rx), int(cx), by_nodes
                )
            pairs_puc[rindex] = np.mean(s_puc)
            rindex += 1
        return IDDTuple(pairs_mtx, pairs_puc)

    @Timer(name="SPUCWorkflow::_batch_misi_range", logger=None)
    def _batch_misi_range(self, bidx:int, pid:int, mi_cache: NDFloatArray):
        st_ranges = self.wdistr.pairs_blocks2d_ranges(bidx, pid)
        return MISIRangePair(mxargs.misi_data_file, st_ranges, mi_cache, False)
    
    @Timer(name="SPUCWorkflow::_load_mi_cache", logger=None)
    def _load_mi_cache(self) -> NDFloatArray:
        with h5py.File(self.mxargs.misi_data_file, 'r') as fptr:
            data_grp : h5py.Group = t.cast(h5py.Group, fptr["data"])
            return data_grp["mi"][:]  # pyright: ignore[reportReturnType,reportIndexIssue]

    @Timer(name="SPUCWorkflow::_compute_with_ranges", logger=None)
    def _compute_with_ranges(
        self,
        rsamples: list[NDIntArray],
        mi_cache: NDFloatArray,
    ) -> list[IDDTuple]:
        empty_tuple = IDDTuple(-np.ones((1,2), dtype=np.int32),
                                 np.zeros(1, np.float32))
        bsamples_puc: list[IDDTuple] = [empty_tuple] * self.wdistr.pair_nbatches
        for bidx in range(self.wdistr.pair_nbatches):
            row_range, col_range = self.wdistr.pairs_blocks2d_ranges(
                bidx,
                self.comm.rank
            )
            #
            misi_range = self._batch_misi_range(bidx, self.comm.rank, mi_cache)
            #
            # self.comm.log_comm(
            #     _log(), logging.DEBUG,
            #     f"B{bidx} :: R: {row_range} ; C: {col_range}"
            # )
            #
            bsamples_puc[bidx] = self._compute_with_misi_range(
                misi_range,
                rsamples
            )
            del misi_range
            # if bidx == 2:
            #      break
            # break
        return bsamples_puc

    def input_sample(self, samples_input_file: str) -> list[NDIntArray]:
        adx = an.read_h5ad(self.mxargs.h5ad_file)
        in_genes = pd.read_csv(samples_input_file, header=None)[0]
        isin_ind = adx.var[self.mxargs.gene_id_col].isin(in_genes)
        return [np.where(isin_ind)[0]]

    @Timer(name="SPUCWorkflow::run_with_ranges", logger=None)
    def run_with_ranges(self, gen_samples:bool=True):
        if gen_samples:
            rsamples = self.generate_samples()
        else:
            if self.mxargs.samples_file is not None:
                rsamples = self.input_sample(self.mxargs.samples_file)
            else:
                rsamples =  [np.array(range(self.mxargs.nvars))] 
        #
        self.comm.log_at_root(
            _log(),
            logging.DEBUG,
            f"nsamples : {[sx.size for sx in rsamples]}"
        )
        mi_cache = self._load_mi_cache()
        slist = self._compute_with_ranges(rsamples, mi_cache)
        # self.comm.barrier()
        #self.comm.log_comm(_log(), logging.DEBUG, f"NSLIST : {len(slist)}")
        #
        all_samples = collect_iddtuples(self.comm, slist)
        if self.comm.rank == 0:
            write_iddtuples(all_samples, self.mxargs.puc_file,
                            DataPair[str]("index", "puc"), "data")

    @Timer(name="SPUCWorkflow::run_with_pairs", logger=None)
    def run_with_pairs(self):
        vsamples = self.generate_samples()
        samples_puc = []
        with MISIDataH5(self.mxargs.misi_data_file).open() as misihx:
            plist = self._generate_pair_list(misihx)
            samples_puc = self._compute_with_pairs(misihx, plist, vsamples)
        self.puc_to_h5(samples_puc)


class MISIWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface

    def __init__(self, mxargs: InputArgs, wdistr: WorkDistributor) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()

    @Timer(name="SIWorkflow::construct_nodes_range", logger=None)
    def construct_nodes_range(self, range_itr:range) -> list[PIDCNode]:
        node_data = self.mxargs.load_range_data(range_itr)
        return [
            PIDCNode(
                node_data[:, ix],
                self.mxargs.nobs,
                self.mxargs.disc_method,
                np.float32,
                np.int32
            )
            for ix, _ in enumerate(range_itr)
        ]

    @Timer(name="SIWorkflow::construct_nodes", logger=None)
    def construct_nodes(self, pid:int) -> list[PIDCNode]:
        nodes = self.construct_nodes_range(self.wdistr.var_range(pid))
        return nodes

    @Timer(name="SIWorkflow::construct_all_nodes", logger=None)
    def construct_all_nodes(self) -> list[PIDCNode]:
        return self.construct_nodes_range(range(self.mxargs.nvars))

    @Timer(name="SIWorkflow::collect_nodes", logger=None)
    def collect_nodes(self, nodes:list[PIDCNode]) -> list[PIDCNode]:
        rnodes = self.comm.collect_merge_lists(nodes)
        return t.cast(list[PIDCNode], rnodes)

    @Timer(name="SIWorkflow::save_nodes_to_h5", logger=None)
    def save_nodes_to_h5(self, rnodes:list[PIDCNode]):
        if self.comm.rank == 0:
            ppld = PIDCPairListData.from_pairs(rnodes, [])
            misd = MISIData.from_pair_list_data(ppld)
            misd.to_h5(self.mxargs.misi_data_file)
            del misd
            del ppld
        self.comm.barrier()

    @Timer(name="SIWorkflow::collect_node_pairs", logger=None)
    def collect_node_pairs(self, node_pairs:list[PIDCPairData]) -> list[PIDCPairData]:
        return self.comm.collect_merge_lists_at_root(node_pairs, True)

    @Timer(name="SIWorkflow::build_misi", logger=None)
    def build_misi(self,
        nodes:list[PIDCNode],
        node_pairs:list[PIDCPairData]
    ):
        return MISIData.from_nodes_and_pairs(
            nodes,
            node_pairs,
            (self.mxargs.nobs, self.mxargs.nvars),
            self.mxargs.disc_method,
            self.mxargs.tbase,
        )

    @Timer(name="SIWorkflow::save_misi", logger=None)
    def save_misi(self, misi_data: MISIData):
        misi_data.to_h5(self.mxargs.misi_data_file)

    @Timer(name="SIWorkflow::construct_pairs_for_range", logger=None)
    def construct_pairs_for_range(
        self,
        nodes: list[PIDCNode],
        px_range: tuple[range, range],
        px_data: tuple[NDFloatArray, NDFloatArray],
    ) -> list[PIDCPairData]:
        row_range, col_range = px_range
        row_data, col_data = px_data
        return [
            PIDCPairData(
                triu_pair_to_index(self.mxargs.nvars, rx, cx),
                rx,
                cx,
                PIDCPair.from_nodes(
                    (nodes[rx], nodes[cx]),
                    (row_data[:, i], col_data[:, j]),
                    mxargs.tbase,
                    np.float32,
                    np.int32,
                ),
            )
            for j, cx in enumerate(col_range) for i, rx in enumerate(row_range)
            if rx < cx
        ]

    @Timer(name="SIWorkflow::construct_node_pairs", logger=None)
    def construct_node_pairs(
        self,
        nodes: list[PIDCNode]
    ) -> list[PIDCPairData]:
        self.comm.barrier()
        gpairs: list[list[PIDCPairData] | None] = [None] * self.wdistr.pair_nbatches
        #
        for bidx in range(self.wdistr.pair_nbatches):
            row_range = self.wdistr.pair_row_range(bidx, self.comm.rank)
            col_range = self.wdistr.pair_col_range(bidx, self.comm.rank)
            row_data, col_data = self.wdistr.load_pair_block_data(bidx, self.comm.rank)
            self.comm.log_comm(
                _log(), logging.DEBUG,
                f"B{bidx} :: R:{row_range} ; C:{col_range}"
            )
            gpairs[bidx] = self.construct_pairs_for_range(
                nodes,
                (row_range, col_range),
                (row_data, col_data)
            )
            del row_data
            del col_data
            # if bidx == 2:
            #     break
        return [px for gplx in gpairs if gplx is not None for px in gplx]

    @Timer(name="save_misi_distributed", logger=None)
    def save_misi_distributed(
        self,
        nodes: list[PIDCNode],
        node_pairs: list[PIDCPairData]
    ):
        distr_misi =  MISIDataDistributed(
            nodes,
            node_pairs,
            (self.mxargs.nobs, self.mxargs.nvars),
            self.mxargs.disc_method,
            self.mxargs.tbase
        )
        distr_misi.to_h5(self.mxargs.misi_data_file)

    @Timer(name="SIWorkflow::save_misi_at_root", logger=None)
    def save_misi_at_root(
        self,
        nodes: list[PIDCNode],
        node_pairs: list[PIDCPairData]
    ):
        rnpairs = self.collect_node_pairs(node_pairs)
        self.comm.log_at_root(_log(), logging.DEBUG, f"rnpairs ::  {len(rnpairs)}" )
        del node_pairs
        rnpairs.sort(key=lambda x: x.pindex)
        if self.comm.rank == 0:
            mdata = self.build_misi(nodes, rnpairs)
            self.save_misi(mdata)

    @Timer(name="SIWorkflow::save_misih5", logger=None)
    def save_misih5(self, node_pairs:list[PIDCPairData]):
        for ix in range(self.comm.size):
            self.comm.barrier()
            if ix == self.comm.rank:
                for pindex, i, j, npair in node_pairs:
                    with MISIDataH5(self.mxargs.misi_data_file, open_mode='a') as misih5:
                        misih5.set_pair_data(pindex, i, j, npair)
            self.comm.barrier()

    def log_list(self, lprefix: str, lst: list[t.Any]):
        tlst = self.comm.collect_counts(len(lst))
        self.comm.log_at_root(
            _log(), logging.DEBUG, "%s    :: %s ",
            lprefix, pformat({"distr": tlst, "count": sum(tlst)})
        )

    def run(self):
        self.comm.log_at_root(_log(), logging.DEBUG,
                              f"Data : {mxargs.nobs} x {mxargs.nvars}")
        #
        nodes = []
        nodes = self.construct_nodes(self.comm.rank)
        nodes = self.collect_nodes(nodes)
        if mxargs.save_nodes and not mxargs.save_node_pairs:
            self.save_nodes_to_h5(nodes)
        # nodes = t.cast(list[PIDCNode], nodes)
        #
        # with open(mxargs.nodes_pickle, 'rb') as fx:
        #      nodes =  pickle.load(fx)
        #
        self.log_list("Nodes", nodes) 
        #
        node_pairs = self.construct_node_pairs(nodes)
        self.log_list("Node Pairs", node_pairs) 
        node_pairs.sort(key=lambda x: x.pindex)
        #
        if mxargs.save_node_pairs:
            self.save_misi_at_root(nodes, node_pairs)
            # self.save_misi_distributed(nodes, node_pairs)


class ContextWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface
    puc_mat: NDFloatArray
    dtype: NPDType
    int_dtype: NPDType

    @Timer(name="ContextWorkflow::__init__", logger=None)
    def __init__(self, mxargs: InputArgs, wdistr: WorkDistributor) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()

        with h5py.File(self.mxargs.puc_file) as hfptr:
            rindex: NDIntArray = hfptr["/data/index"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            rpuc: NDFloatArray = hfptr["/data/puc"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            self.dtype = rpuc.dtype
            self.int_dtype = rindex.dtype
            self.puc_mat = np.zeros((mxargs.nvars, mxargs.nvars),
                                    self.dtype)
            self.puc_mat[rindex[:, 0], rindex[:, 1]] = rpuc
            self.puc_mat[rindex[:, 1], rindex[:, 0]] = rpuc

    @Timer(name="ContextWorkflow::compute_context", logger=None)
    def compute_context(self) -> list[IDDTuple]:
        
        empty_tuple = IDDTuple(-np.ones((1,2), dtype=np.int32),
                                 np.zeros(1, np.float32))
        bats_pidc: list[IDDTuple] = [empty_tuple] * self.wdistr.pair_nbatches
        for bid in range(self.wdistr.pair_nbatches):
            st_ranges = self.wdistr.pairs_blocks2d_ranges(
                bid, self.comm.rank
            )
            px_list = init_range_input(st_ranges, self.int_dtype) 
            px_pidc = np.zeros(len(px_list), self.dtype)
            rindex = 0
            for rx, cx in px_list:
                px_pidc[rindex] = get_clr_weight(self.puc_mat, rx, cx)
                rindex += 1
            bats_pidc[bid] = IDDTuple(px_list, px_pidc)
        return bats_pidc

        nsize = pd_range.stop - pd_range.start
        px_indices: NDIntArray = np.zeros((nsize, 2), np.int32)
        px_pidc: NDFloatArray = np.zeros(nsize, self.puc_mat)
        for prix in pd_range:
            ix, iy = triu_index_to_pair(self.mxargs.npairs, prix)
            px_indices[prix, :] = (ix, iy)
            px_pidc[prix] = get_clr_weight(self.puc_mat, ix, iy)
        return IDDTuple(px_indices, px_pidc)

    @Timer(name="ContextWorkflow::collect_pidc_tuples", logger=None)
    def collect_pidc_tuples(
        self,
        pidc_tuple:IDDTuple
    ) -> IDDTuple:
        all_indices: None | list[NDIntArray] = self.comm.collect_objects_at_root(pidc_tuple.first)
        all_values : None | list[NDFloatArray] = self.comm.collect_objects_at_root(pidc_tuple.second)
        full_indices : NDIntArray = np.array(())
        full_values : NDFloatArray = np.array(())
        if self.comm.rank == 0:
            if (all_indices is not None) and (all_values is not None):
                full_indices = np.vstack(all_indices) 
                full_values = np.concatenate(all_values)
        self.comm.barrier()
        return IDDTuple(full_indices, full_values)


    def run(self):
        pidc_tuples = self.compute_context()
        #
        all_tuples = collect_iddtuples(self.comm, pidc_tuples)
        # all_tuples = self.collect_pidc_tuples(pidc_tuples)
        if self.comm.rank == 0:
            write_iddtuples(all_tuples, self.mxargs.pidc_file,
                            DataPair[str]("index", "pidc"), "data")
            # import h5py
            # with h5py.File(self.mxargs.pidc_file, 'w') as fptr:
            #     data_grp = fptr.create_group("data")
            #     create_h5ds(data_grp, "index", all_tuples.first)
            #     create_h5ds(data_grp, "pidc", all_tuples.second)


def main(mxargs: InputArgs):
    env_vars = ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
    #
    comm_ifx = default_comm()
    wdistr = WorkDistributor(mxargs, comm_ifx.size, comm_ifx.rank)
    comm_ifx.log_at_root(_log(), logging.INFO, "ENV:: %s ",
                         pformat({vx: getenv(vx) for vx in env_vars }))
    comm_ifx.log_at_root(
        _log(),
        logging.INFO, "MXARGS : %s",
        pformat(
            mxargs.model_dump(exclude=["h5_fptr"])  # pyright: ignore[reportArgumentType]
        )
    )
    # comm_ifx.log_at_root(
    #     _log(),
    #     logging.DEBUG, "WDISTR : %s",
    #     pformat(
    #         wdistr.model_dump(exclude=["mxargs"])  # pyright: ignore[reportArgumentType]
    #     )
    # )
    #
    for rxmode in mxargs.mode:
        match rxmode:
            case 'misi':
                MISIWorkflow(mxargs, wdistr).run()
            case 'sampled_puc_pairs':
                SPUCWorkflow(mxargs, wdistr).run_with_pairs()
            case 'samples_ranges':
                SPUCWorkflow(mxargs, wdistr).run_with_ranges(True)
            case 'samples_input':
                SPUCWorkflow(mxargs, wdistr).run_with_ranges(False)
            case 'puc_ranges':
                SPUCWorkflow(mxargs, wdistr).run_with_ranges(False)
            case 'puc_lmr':
                FullPUCWorkflow(mxargs, wdistr).run_with_ranges()
            case 'puc2pidc':
                ContextWorkflow(mxargs, wdistr).run()
        comm_ifx.barrier()
    #
    if mxargs.enable_timers:
        comm_ifx.barrier()
        comm_ifx.log_profile_summary(_log(), logging.DEBUG)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       prog="misi_data",
       description="Generate GRNs w. XGB/lightgbm for Single Cell Data"
   )
   parser.add_argument(
       "yaml_file",
       default="./config/networks/pbmc20k_5k.yaml",
       help=f"Yaml Input file with a given configuration."
   )
   run_args = parser.parse_args()
   if default_comm().rank == 0:
       print("Run Arguments :: ", run_args)
   with InputArgs.from_yaml(run_args.yaml_file) as mxargs:
       main(mxargs)
