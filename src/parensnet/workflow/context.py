import h5py
import numpy as np

#
from codetiming import Timer

from .util import (
    InputArgs, WorkDistributor, init_range_input,
    collect_iddtuples, iddtuple_to_h5
)
from ..types import NPDType, NDIntArray, NDFloatArray, DataPair, IDDTuple
from ..comm_interface import CommInterface, default_comm
from ..mvim.context import get_clr_weight


class ContextWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface
    puc_file: str
    pidc_file: str
    puc_mat: NDFloatArray
    dtype: NPDType
    int_dtype: NPDType

    @Timer(name="ContextWorkflow::__init__", logger=None)
    def __init__(
        self,
        mxargs: InputArgs,
        wdistr: WorkDistributor,
        puc_file: str | None = None,
        pidc_file: str | None = None,
    ) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()
        #
        if puc_file:
            self.puc_file = puc_file
        else:
            self.puc_file = mxargs.puc_file
        #
        if pidc_file:
            self.pidc_file = pidc_file
        else:
            self.pidc_file = mxargs.pidc_file

        with h5py.File(self.puc_file) as hfptr:
            rindex: NDIntArray = hfptr["/data/index"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            rpuc: NDFloatArray = hfptr["/data/puc"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
            self.dtype = rpuc.dtype
            self.int_dtype = rindex.dtype
            self.puc_mat = np.zeros((mxargs.nvars, mxargs.nvars),
                                    self.dtype)
            if self.mxargs.quantile_filter is not None:
                qfilter = np.quantile(rpuc, self.mxargs.quantile_filter)
                rpuc[rpuc < qfilter] = 0.0
            self.puc_mat[rindex[:, 0], rindex[:, 1]] = rpuc
            self.puc_mat[rindex[:, 1], rindex[:, 0]] = rpuc

    @Timer(name="ContextWorkflow::compute_context", logger=None)
    def compute_context(self) -> list[IDDTuple]:
        # 
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

        # nsize = pd_range.stop - pd_range.start
        # px_indices: NDIntArray = np.zeros((nsize, 2), np.int32)
        # px_pidc: NDFloatArray = np.zeros(nsize, self.puc_mat)
        # for prix in pd_range:
        #     ix, iy = triu_index_to_pair(self.mxargs.npairs, prix)
        #     px_indices[prix, :] = (ix, iy)
        #     px_pidc[prix] = get_clr_weight(self.puc_mat, ix, iy)
        # return IDDTuple(px_indices, px_pidc)

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
            iddtuple_to_h5(all_tuples, self.pidc_file,
                            DataPair[str]("index", "pidc"), "data")
            # import h5py
            # with h5py.File(self.mxargs.pidc_file, 'w') as fptr:
            #     data_grp = fptr.create_group("data")
            #     create_h5ds(data_grp, "index", all_tuples.first)
            #     create_h5ds(data_grp, "pidc", all_tuples.second)



