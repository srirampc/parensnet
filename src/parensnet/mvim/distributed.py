import numpy as np
import h5py

from codetiming import Timer

from ..types import DiscretizerMethod, LogBase, NPDType, NDIntArray, NDFloatArray
from ..util import create_h5ds, flatten_npalist
from ..comm_interface import default_comm
from .rv import RVNode, RVPairData

class MISIDataDistributed:
    node_pairs:list[RVPairData]
    disc_method: DiscretizerMethod = "bayesian_blocks"
    tbase: LogBase = '2'
    #
    nvars: int = 0
    nobs: int = 0
    npairs: int = 0
    local_npairs: int = 0
    local_nsjv_dim: int = 0
    nsjv_dim: int = 0
    nsi: int = 0
    #
    hist_dim:  NDIntArray = np.array(())
    hist_dim_pfxe: NDIntArray = np.array(())
    hist_start:  NDIntArray = np.array(())
    bins_dim: NDIntArray = np.array(())
    bins_start:  NDIntArray = np.array(())
    #
    pair_index: NDIntArray = np.array(())
    #
    jv_dim: NDIntArray = np.array(())
    jv_start: NDIntArray = np.array(())
    jv_row_start: NDIntArray = np.array(())
    si_start: NDIntArray = np.array(())
    #
    hist: NDFloatArray = np.array(())
    bins: NDFloatArray = np.array(())
    pair_hist_index: NDIntArray = np.array(())
    pair_hist: NDFloatArray = np.array(())
    pair_jvir: NDFloatArray = np.array(())
    pindices: NDIntArray = np.array(())
    mi: NDFloatArray = np.array(())
    si: NDFloatArray = np.array(())
    lmr: NDFloatArray = np.array(())
    #
    ftype: NPDType = np.float32
    itype: NPDType = np.int32
    idx_type: NPDType = np.int64

    def float_dtype(self) -> NPDType:
        return self.ftype

    def int_dtype(self) -> NPDType:
        return self.itype

    def index_dtype(self) -> NPDType:
        return self.idx_type

    def set_nodes_data(
        self,
        bins_list: list[NDFloatArray],
        hist_list: list[NDFloatArray]
    ):
        self.bins_dim, self.bins_start, self.bins = flatten_npalist(
            bins_list, self.float_dtype(), self.int_dtype(), self.idx_type
        )
        self.hist_dim, self.hist_start, self.hist = flatten_npalist(
            hist_list, self.float_dtype(), self.int_dtype(), self.idx_type
        )

    def __init_jv_index_auxds(self):
        # hist_dim = self.hist_dim
        hist_rsum = np.zeros(self.nvars, self.index_dtype())
        for i in range(1, self.nvars):
            hist_rsum[-i] = hist_rsum[-i+1] + self.hist_dim[-i]
        hist_rsum[1:] = hist_rsum[1:] * self.hist_dim[:-1]
        self.jv_row_start = np.zeros(self.nvars, self.index_dtype())
        for i in range(1, self.nvars):
            self.jv_row_start[i] = self.jv_row_start[i-1] + hist_rsum[i]
        self.hist_dim_pfxe = np.zeros(self.nvars, self.index_dtype())
        for i in range(1, self.nvars):
            self.hist_dim_pfxe[i] = self.hist_dim_pfxe[i-1] + self.hist_dim[i-1]

    def ___init_node_data(self, nodes: list[RVNode]):
        self.set_nodes_data(
            [nx.bins for nx in nodes],
            [nx.hist for nx in nodes],
        )
        self.si_start = np.zeros(self.hist_dim.size, dtype=self.idx_type)
        for index in range(1, self.nvars):
            self.si_start[index] = (
                self.si_start[index - 1] +
                self.nvars * self.hist_dim[index - 1]
            )
        self.nsi = int(np.sum(self.hist_dim)) * (self.nvars)

    def __init_node_pairs_data(self, node_pairs: list[RVPairData]):
        self.__init_jv_index_auxds()
        comm_ifx = default_comm()
        self.local_npairs = len(node_pairs)
        self.npairs = sum(comm_ifx.collect_counts(self.local_npairs))
        self.pindices = np.zeros(self.local_npairs, dtype=self.idx_type)
        self.mi = np.zeros(self.local_npairs, dtype=self.float_dtype())
        self.jv_dim = np.zeros(self.local_npairs, dtype=self.int_dtype())
        self.jv_start = np.zeros(self.local_npairs, dtype=self.index_dtype())
        self.local_nsjv_dim = int(np.sum([
            int(px.pidc_pair.sthist.size) for px in node_pairs
        ]))
        self.nsjv_dim = sum(comm_ifx.collect_counts(self.local_nsjv_dim))
        self.pair_index = np.zeros((self.local_npairs, 2),
                                   dtype=self.int_dtype())
        self.pair_hist = np.zeros(self.local_nsjv_dim, dtype=self.float_dtype())
        self.pair_jvir = np.zeros(self.local_nsjv_dim, dtype=self.float_dtype())
        self.pair_hist_index = np.zeros(self.local_nsjv_dim,
                                        dtype=self.index_dtype())
        lcl_jv_start = 0
        for ridx, npx in enumerate(node_pairs):
            jv_size = npx.pidc_pair.sthist.size
            self.pindices[ridx] = npx.pindex
            self.mi[ridx] = npx.pidc_pair.mi
            self.jv_dim[ridx] = jv_size
            self.jv_start[ridx] = self.jv_start_of(npx.i, npx.j)
            self.pair_index[ridx] = npx.i, npx.j
            npair = npx.pidc_pair
            jv_start = self.jv_start[ridx]
            jv_size = self.jv_dim[ridx]
            jv_stop = jv_start + jv_size
            lcl_jv_stop = lcl_jv_start + jv_size
            lcl_range = range(lcl_jv_start, lcl_jv_stop)
            self.pair_hist_index[lcl_range] = list(range(jv_start, jv_stop))
            self.pair_hist[lcl_range] = npair.sthist.reshape(jv_size)
            self.pair_jvir[lcl_range] = npair.ljvi.reshape(jv_size)
            lcl_jv_start += jv_size

    @Timer(name="MISIDataDistributed::root_to_h5", logger=None)
    def root_to_h5(self, h5_file: str):
        with h5py.File(h5_file, 'w') as fptr:
            data_grp = fptr.create_group("data")
            data_grp.attrs["disc_method"] = self.disc_method
            data_grp.attrs["tbase"] = self.tbase
            data_grp.attrs["nvars"] = self.nvars
            data_grp.attrs["nobs"] = self.nobs
            data_grp.attrs["npairs"] = self.npairs
            data_grp.attrs["nsjv_dim"] = self.nsjv_dim
            data_grp.attrs["nsi"] = self.nsi
            #
            create_h5ds(data_grp, "hist_dim", self.hist_dim)
            create_h5ds(data_grp, "hist_start", self.hist_start)
            create_h5ds(data_grp, "hist", self.hist)
            create_h5ds(data_grp, "bins_dim", self.bins_dim)
            create_h5ds(data_grp, "bins_start", self.bins_start)
            create_h5ds(data_grp, "bins", self.bins)
            create_h5ds(data_grp, "si_start", self.si_start)
            #
            data_grp.create_dataset("pair_index", (self.npairs, 2), 'i4')
            data_grp.create_dataset("jv_dim", (self.npairs, ), 'i4')
            data_grp.create_dataset("jv_start", (self.npairs, ), 'i8')
            data_grp.create_dataset("pair_hist", (self.nsjv_dim, ), 'f4')
            data_grp.create_dataset("pair_jvir", (self.nsjv_dim, ), 'f4')
            data_grp.create_dataset("mi", (self.npairs, ), 'f4')
            # data_grp.create_dataset("si", (self.nsi, ), 'f4')
            # data_grp.create_dataset("lmr", (self.nsi, ), 'f4')

    @Timer(name="MISIDataDistributed::local_to_h5", logger=None)
    def local_to_h5(self, h5_file: str): 
        with h5py.File(h5_file, 'a') as fptr:
            data_grp = fptr["data"]
            data_grp["pair_index"][self.pindices] = self.pair_index   # pyright: ignore[reportIndexIssue]
            data_grp["jv_dim"][self.pindices] = self.jv_dim   # pyright: ignore[reportIndexIssue]
            data_grp["jv_start"][self.pindices] = self.jv_start   # pyright: ignore[ reportIndexIssue]
            data_grp["mi"][self.pindices] = self.mi   # pyright: ignore[reportIndexIssue]
            data_grp["pair_hist"][self.pair_hist_index] =  self.pair_hist # pyright: ignore[reportIndexIssue]
            #TODO:: Distributed
            # for ridx, npx in enumerate(self.node_pairs):
            #     npair = npx.pidc_pair
            #     jv_start = self.jv_start[ridx]
            #     jv_size = self.jv_dim[ridx]
            #     jv_stop = jv_start + jv_size
            #     data_grp["pair_hist"][jv_start:jv_stop] = npair.sthist.reshape(jv_size)   
            #     data_grp["pair_jvir"][jv_start:jv_stop] = npair.ljvi.reshape(jv_size)
            #     npx.pidc_pair.x_lmr
            #     npx.pidc_pair.y_lmr
            #     npx.pidc_pair.x_si
            #     npx.pidc_pair.y_si

    @Timer(name="MISIDataDistributed::to_h5", logger=None)
    def to_h5(self, h5_file: str):
        comm_ifx = default_comm()
        if comm_ifx.rank == 0:
            self.root_to_h5(h5_file)
        comm_ifx.barrier()
        for i in range(comm_ifx.size):
            if comm_ifx.rank == i:
                self.local_to_h5(h5_file)
            comm_ifx.barrier()

    @Timer(name="MISIDataDistributed::__init__", logger=None)
    def __init__(
        self,
        nodes: list[RVNode],
        node_pairs:list[RVPairData],
        dshape: tuple[int, int],
        disc_method: DiscretizerMethod,
        tbase: LogBase,
    ) -> None:
        self.node_pairs = node_pairs
        self.nobs, self.nvars = dshape
        self.disc_method = disc_method
        self.tbase = tbase
        #
        self.___init_node_data(nodes)
        #
        self.__init_node_pairs_data(node_pairs)

    def jv_start_of(self, i: int, j: int):
        return (
            self.jv_row_start[i] + (
                (self.hist_dim_pfxe[j] - self.hist_dim_pfxe[i+1]) *
                self.hist_dim[i]
            )
        )
