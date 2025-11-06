import logging
import h5py
import typing as t
import numpy as np
import pandas as pd
import anndata as an

#
from codetiming import Timer

from ..types import NDIntArray, NDFloatArray, DataPair, PUCTuple, IDDTuple
from ..comm_interface import CommInterface, default_comm
from ..misi import MISIRangePair, MISIDataH5, MISIPair
from ..util import create_h5ds, block_range
from .util import (
    InputArgs, WorkDistributor, init_range_input,
    collect_iddtuples, iddtuple_to_h5
)

def _log():
    return logging.getLogger(__name__)


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
        return MISIRangePair(self.mxargs.misi_data_file, st_ranges, mi_cache, True)
    
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
    misi_data_file: str
    puc_file: str

    def __init__(
        self,
        mxargs: InputArgs,
        wdistr: WorkDistributor,
        data_file: str | None = None,
        puc_file: str | None = None,
    ) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()
        if data_file:
            self.misi_data_file = data_file
        else:
            self.misi_data_file = mxargs.misi_data_file
        if puc_file:
            self.puc_file = puc_file
        else:
            self.puc_file = mxargs.puc_file

    @Timer(name="SPUCWorkflow::generate_samples", logger=None)
    def generate_samples(self) -> list[NDIntArray]:
        rsamples = np.array([])
        if self.comm.rank == 0:
            rsamples = np.random.choice(
                range(self.mxargs.nvars),
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
    def puc_to_h5(self, samples_puc: NDFloatArray):
        gpuc_list = self.comm.gather_at_root_by_snd_rcv(samples_puc)
        if self.comm.rank != 0:
            return
        if not gpuc_list:
            return
        gpuc_samples = np.concatenate(gpuc_list)
        self.comm.log_at_root(_log(), logging.DEBUG, f"{gpuc_samples.shape}" )
        import h5py
        with h5py.File(self.puc_file, 'w') as fptr:
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
        return MISIRangePair(self.misi_data_file, st_ranges, mi_cache, False)
    
    @Timer(name="SPUCWorkflow::_load_mi_cache", logger=None)
    def _load_mi_cache(self) -> NDFloatArray:
        with h5py.File(self.misi_data_file, 'r') as fptr:
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
            _row_range, _col_range = self.wdistr.pairs_blocks2d_ranges(
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

    @Timer(name="SPUCWorkflow::range_puc_to_h5", logger=None)
    def range_puc_to_h5(self, slist: list[IDDTuple]):
        all_samples = collect_iddtuples(self.comm, slist)

        self.comm.barrier()
        self.comm.log_at_root(_log(), logging.DEBUG, f"FULL LIST : {len(all_samples)}")
        if self.comm.rank == 0:
            iddtuple_to_h5(all_samples, self.puc_file,
                            DataPair[str]("index", "puc"), "data")

    @Timer(name="SPUCWorkflow::range_puc_to_par_h5", logger=None)
    def range_puc_to_par_h5(self, slist: list[IDDTuple]):
        all_samples = collect_iddtuples(self.comm, slist)
        if self.comm.rank == 0:
            iddtuple_to_h5(all_samples, self.puc_file,
                            DataPair[str]("index", "puc"), "data")


    @Timer(name="SPUCWorkflow::range_puc_to_zarr", logger=None)
    def range_puc_to_zarr(self, idds_list: list[IDDTuple]):
        import zarr
        pindices = np.vstack([px.first for px in idds_list])
        pvalues = np.concatenate([px.second for px in idds_list])
        nlocal = len(pvalues)
        ilcounts = self.comm.collect_counts(nlocal)
        range_start = int(np.sum(ilcounts[:self.comm.rank]))
        range_end = range_start + nlocal

        if self.comm.rank == 0:
            zindex = zarr.create_array(
                store=self.puc_file,
                name="data/index",
                shape=(self.mxargs.npairs, 2),
                dtype=pindices.dtype,
                overwrite=True,
            )
            zvalues = zarr.create_array(
                store=self.puc_file,
                name="data/puc",
                shape=self.mxargs.npairs,
                dtype=pvalues.dtype,
                overwrite=True,
            )
        self.comm.barrier()
        self.comm.log_comm(_log(), logging.DEBUG,
                           f"R:: {range_start}, {range_end}")
        self.comm.barrier()
        zindex = zarr.open_array(
            store=self.puc_file,
            path="data/index",
            mode='r+',
        )
        zvalues = zarr.open_array(
            store=self.puc_file,
            path="data/puc",
            mode='r+',
        )
        zindex[range_start:range_end, :] = pindices
        zvalues[range_start:range_end] = pvalues

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
        #
        if self.puc_file.endswith("zarr"):
            self.range_puc_to_zarr(slist)
        else:
            self.range_puc_to_h5(slist)
        
    @Timer(name="SPUCWorkflow::run_with_pairs", logger=None)
    def run_with_pairs(self):
        vsamples = self.generate_samples()
        samples_puc = []
        with MISIDataH5(self.misi_data_file).open() as misihx:
            plist = self._generate_pair_list(misihx)
            samples_puc = self._compute_with_pairs(misihx, plist, vsamples)
        self.puc_to_h5(samples_puc)
