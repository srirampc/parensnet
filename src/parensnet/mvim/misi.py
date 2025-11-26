import itertools
import typing as t
import numpy as np
import h5py

from collections.abc import Iterable
from ..types import DiscretizerMethod, LogBase, FloatT, IntegerT, DataPair
from ..types import NPDType, NPFloat, NDIntArray, NDFloatArray
from ..util import create_h5ds, triu_pair_to_index, flatten_npalist
from .rv import (
  RVNode, RVNodePair,RVPairData, MRVInterface, MRVNodePairs,
  LMRSubsetDataStructure, LMRDataStructure
)
from .imeasures import LMR, redundancy

class MISIData(MRVInterface):
    disc_method: DiscretizerMethod = "bayesian_blocks"
    tbase: LogBase = '2'
    #
    nvars: int = 0
    nobs: int = 0
    npairs: int = 0
    nsjv_dim: int = 0
    nsi: int = 0
    #
    hist_dim:  NDIntArray = np.array(())
    hist_start:  NDIntArray = np.array(())
    bins_dim: NDIntArray = np.array(())
    bins_start:  NDIntArray = np.array(())
    #
    pair_index: NDIntArray = np.array(())
    pair_lookup: dict[tuple[int, int], int] = {}
    #
    jv_dim: NDIntArray = np.array(())
    jv_start: NDIntArray = np.array(())
    si_start: NDIntArray = np.array(())
    #
    hist: NDFloatArray = np.array(())
    bins: NDFloatArray = np.array(())
    pair_hist: NDFloatArray  | None = None
    pair_jvir: NDFloatArray  | None = None
    mi: NDFloatArray = np.array(())
    si: NDFloatArray = np.array(())
    lmr: NDFloatArray = np.array(())
    #
    ftype: NPDType = np.float32
    itype: NPDType = np.int32
    idx_type: NPDType = np.int64

    @t.override
    def float_dtype(self) -> NPDType:
        return self.ftype

    @t.override
    def int_dtype(self) -> NPDType:
        return self.itype

    @t.override
    def index_dtype(self) -> NPDType:
        return self.idx_type

    @t.override
    def get_mi(self, i:int, j:int) -> float | NPFloat:
        return self.mi[self.pair_lookup[(i, j)]]

    @t.override
    def get_si(self, about:int, by:int) -> NDFloatArray:
        return self.si[self.si_bounds(about=about, by=by)]

    @property
    @t.override
    def ndata(self) -> float | NPFloat:
        return float(self.nobs)

    @property
    @t.override
    def nvariables(self):
        return self.nvars

    @t.override
    def get_hist_dim(self, i: int) -> IntegerT:
        return self.hist_dim[i]

    @t.override
    def get_hist(self, i: int) -> NDFloatArray:
        assert i < self.hist_start.size
        hstart = self.hist_start[i]
        hend = self.hist_start[i] + self.hist_dim[i]
        assert hstart < self.hist.size
        assert hend <= self.hist.size
        return self.hist[hstart:hend]

    def set_pair_data(self, pindex: int, i: int, j: int, npair: RVNodePair):
        self.mi[pindex] = npair.mi
        #
        jvsize = self.jv_dim[pindex]
        jvbounds = range(self.jv_start[pindex], self.jv_start[pindex] + jvsize)
        if self.pair_hist is not None and npair.sthist is not None:
            self.pair_hist[jvbounds] = npair.sthist.reshape(jvsize)
        if self.pair_jvir is not None and npair.ljvi is not None:
            self.pair_jvir[jvbounds] = npair.ljvi.reshape(jvsize)
        #
        isi_bounds, jsi_bounds = self.pair_si_bounds(i, j)
        self.si[isi_bounds] = npair.x_si
        self.si[jsi_bounds] = npair.y_si
        self.lmr[isi_bounds] = npair.x_lmr
        self.lmr[jsi_bounds] = npair.y_lmr

    def set_nodes_data(
        self,
        bins_list: list[NDFloatArray],
        hist_list: list[NDFloatArray]
    ):
        self.bins_dim, self.bins_start, self.bins = flatten_npalist(
            bins_list, self.float_dtype(),
            self.int_dtype(), self.index_dtype()
        )
        self.hist_dim, self.hist_start, self.hist = flatten_npalist(
            hist_list, self.float_dtype(),
            self.int_dtype(), self.index_dtype()
        )

    def get_bins(self, i: int) -> NDFloatArray:
        assert i < self.bins_start.size
        bstart = self.bins_start[i]
        bend = self.bins_start[i] + self.bins_dim[i]
        assert bstart < self.bins.size
        assert bend <= self.bins.size
        return self.hist[bstart:bend]

    def get_node_data(self, i:int):
        return self.get_hist(i),  self.get_bins(i)

    def get_si_start(self, about: int):
        return self.si_start[about]

    def get_si_range(self, from_idx:int, to_idx:int) -> NDFloatArray:
        return self.si[from_idx:to_idx]

    def get_lmr_range(self, from_idx:int, to_idx:int) -> NDFloatArray:
        return self.lmr[from_idx:to_idx]

    def si_begin(self, about: int, by: int):
        return self.si_start[about] + (by * self.hist_dim[about])

    def si_bounds(self, about: int, by: int):
        bstart = self.si_begin(about, by)
        bend = bstart + self.hist_dim[about]
        return range(bstart, bend)

    def jv_bounds(self, i: int, j: int):
        assert i < j
        pindex = self.pair_lookup[(i, j)]
        jv_start = self.jv_start[pindex]
        jv_end = jv_start + self.jv_dim[pindex]
        return jv_start, jv_end

    def pair_si_bounds(self, i: int, j: int):
        return (
            self.si_bounds(about=i, by=j),
            self.si_bounds(about=j, by=i),
        )

    def __get_ljvr(self, i:int, j:int) -> NDFloatArray:
        assert i < j
        idim, jdim = self.get_hist_dim(i), self.get_hist_dim(j)
        jv_start, jv_end = self.jv_bounds(i, j)
        return self.pair_jvir[jv_start:jv_end].reshape((idim, jdim))

    def get_ljvr(self, i:int, j:int):
        if i < j:
            return self.__get_ljvr(i, j)
        else:
            return self.__get_ljvr(j, i)

    def __get_joint_hist(self, i:int, j:int) -> NDFloatArray:
        assert i < j
        idim, jdim = self.get_hist_dim(i), self.get_hist_dim(j)
        jv_start, jv_end = self.jv_bounds(i, j)
        return self.pair_hist[jv_start:jv_end].reshape((idim, jdim))

    def get_joint_hist(self, i:int, j:int):
        if i < j:
            return self.__get_joint_hist(i, j)
        else:
            return self.__get_joint_hist(j, i)

    @t.override
    def si_value(self, about: int, by: int, rstate:int) -> FloatT:
        bstart = self.si_begin(about, by)
        return self.si[bstart + rstate] # pyright: ignore[reportReturnType]

    @t.override
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        return self.lmr[self.si_bounds(about=about, by=by)]

    @t.override
    def lmr_value(self, about: int, by: int, rstate:int) -> FloatT:
        bstart = self.si_begin(about, by)
        return self.lmr[bstart + rstate] # pyright: ignore[reportReturnType]

    def compute_lmr(self, about: int, by: int):
        ljv_ratio = self.get_ljvr(about, by)
        jv_hist = self.get_joint_hist(about, by)
        if about < by:
            return LMR.about_x_from_ljvi(ljv_ratio, jv_hist, self.nobs)
        else:
            return LMR.about_y_from_ljvi(ljv_ratio, jv_hist, self.nobs)

    def set_lmr_data(self, about:int, by:int, lmr_data: NDFloatArray):
        lmr_bounds = self.si_bounds(about, by)
        self.lmr[lmr_bounds] = lmr_data

    def populate_lmr(self):
        self.lmr = np.zeros(self.nsi, dtype=self.float_dtype())
        for pindex in range(self.npairs):
            i, j = self.pair_index[pindex]
            isi_bounds, jsi_bounds = self.pair_si_bounds(i, j)
            self.lmr[isi_bounds] = self.compute_lmr(about=i, by=j)
            self.lmr[jsi_bounds] = self.compute_lmr(about=j, by=i)

    def prep_pair_data(self, hist_dim: NDIntArray, nvars: int, save_hist:bool):
        self.npairs = nvars * (nvars - 1)//2
        self.nsi = int(np.sum(hist_dim)) * (nvars)
        self.si_start = np.zeros(hist_dim.size, dtype=self.idx_type)
        for index in range(1, nvars):
            self.si_start[index] = (
                self.si_start[index - 1] +
                nvars * hist_dim[index - 1]
            )
        #
        self.pair_lookup = dict(zip(
            itertools.combinations(range(nvars), 2),
            range(self.npairs),
        ))
        self.pair_index = t.cast(NDIntArray, np.fromiter(
            itertools.combinations(range(self.nvars), 2),
            dtype=np.dtype((self.int_dtype(), 2)),
            count=self.npairs
        ))
        #
        self.jv_dim = np.multiply(
            self.hist_dim[self.pair_index[:, 0]],
            self.hist_dim[self.pair_index[:, 1]]
        )
        self.jv_start = np.zeros(self.npairs, dtype=self.idx_type)
        for ijx in range(1, self.npairs):
            self.jv_start[ijx] = self.jv_dim[ijx - 1] + self.jv_start[ijx - 1]
        #
        self.nsjv_dim = int(np.sum(self.jv_dim))
        if save_hist:
            self.pair_hist = np.zeros(self.nsjv_dim, dtype=self.float_dtype())
            self.pair_jvir = np.zeros(self.nsjv_dim, dtype=self.float_dtype())
        else:
            self.pair_hist = None
            self.pair_jvir = None
        self.mi = np.zeros(self.npairs, dtype=self.float_dtype())
        self.si = np.zeros(self.nsi, dtype=self.float_dtype())
        self.lmr = np.zeros(self.nsi, dtype=self.float_dtype())

    def assert_equal(self, lobject: "MISIData") -> bool:
        assert self.disc_method == lobject.disc_method
        assert self.tbase == lobject.tbase
        #
        assert lobject.nvars    == self.nvars
        assert lobject.nobs     == self.nobs
        assert lobject.npairs   == self.npairs
        assert lobject.nsjv_dim == self.nsjv_dim
        assert lobject.nsi      == self.nsi
        #
        assert np.all(lobject.hist_dim    ==   self.hist_dim  )
        assert np.all(lobject.hist_start  ==   self.hist_start)
        assert np.all(lobject.bins_dim    ==   self.bins_dim )
        assert np.all(lobject.bins_start  ==   self.bins_start)
        #
        assert np.all(lobject.mi          == self.mi        )
        assert np.all(lobject.pair_index  == self.pair_index )
        assert np.all(lobject.pair_lookup == self.pair_lookup)
        assert np.all(lobject.jv_dim      == self.jv_dim     )
        assert np.all(lobject.jv_start    == self.jv_start   )
        assert np.all(lobject.si_start    == self.si_start   )
        assert np.all(lobject.hist        == self.hist       )
        assert np.all(lobject.bins        == self.bins       )
        assert np.all(lobject.pair_hist   == self.pair_hist  )
        assert np.all(lobject.pair_jvir   == self.pair_jvir  )
        assert np.all(lobject.si          == self.si         )
        assert np.all(lobject.lmr         == self.lmr        )
        return True

    def to_h5(self, h5_file: str):
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
            create_h5ds(data_grp, "pair_index", self.pair_index)
            # pair_lookup: dict[tuple[int, int], int] = {}
            create_h5ds(data_grp, "jv_dim", self.jv_dim)
            create_h5ds(data_grp, "jv_start", self.jv_start)
            if self.pair_hist is not None:
                create_h5ds(data_grp, "pair_hist", self.pair_hist)
            if self.pair_jvir is not None:
                create_h5ds(data_grp, "pair_jvir", self.pair_jvir)
            create_h5ds(data_grp, "mi", self.mi)
            create_h5ds(data_grp, "si", self.si)
            create_h5ds(data_grp, "lmr", self.lmr)


    @classmethod
    def from_h5(cls, h5_file: str):
        with h5py.File(h5_file, 'r') as fptr:
            misiobj = cls()
            data_grp : h5py.Group = t.cast(h5py.Group, fptr["data"])
            misiobj.disc_method = data_grp.attrs["disc_method"] # pyright: ignore[reportAttributeAccessIssue]
            misiobj.tbase       = data_grp.attrs["tbase"] # pyright: ignore[reportAttributeAccessIssue]
            misiobj.nvars       = int(data_grp.attrs["nvars"])  # pyright: ignore[reportArgumentType]
            misiobj.nobs        = int(data_grp.attrs["nobs"])  # pyright: ignore[reportArgumentType]
            misiobj.npairs      = int(data_grp.attrs["npairs"])  # pyright: ignore[reportArgumentType]
            misiobj.nsjv_dim    = int(data_grp.attrs["nsjv_dim"])  # pyright: ignore[reportArgumentType]
            misiobj.nsi         = int(data_grp.attrs["nsi"])  # pyright: ignore[reportArgumentType]
            misiobj.hist_dim   = data_grp["hist_dim"][:]    # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.hist_start = data_grp["hist_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.hist       = data_grp["hist"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.bins_dim   = data_grp["bins_dim"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.bins_start = data_grp["bins_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.bins       = data_grp["bins"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            #
            misiobj.mi = data_grp["mi"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.pair_index = data_grp["pair_index"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.jv_dim    = data_grp["jv_dim"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.jv_start  = data_grp["jv_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.si_start  = data_grp["si_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            if "pair_hist" in data_grp.keys():
                misiobj.pair_hist = data_grp["pair_hist"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            if "pair_jvir" in data_grp.keys():
                misiobj.pair_jvir = data_grp["pair_jvir"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.si        = data_grp["si"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.lmr       = data_grp["lmr"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
            misiobj.pair_lookup = {
                (int(i), int(j)): int(index)
                for index, (i, j) in enumerate(misiobj.pair_index)
            }
        return misiobj

    @classmethod
    def from_data(
        cls,
        data: NDFloatArray,
        data_dims: tuple[int, int] | None,
        tbase: LogBase,
        disc_method: DiscretizerMethod,
        save_hist: bool = True,
    ):
        mobj = cls()
        mobj.disc_method = disc_method
        mobj.tbase = tbase
        mobj.nobs, mobj.nvars= data.shape if data_dims is None else data_dims
        mobj.npairs = mobj.nvars * (mobj.nvars - 1)//2
        #
        bins_list, hist_list = RVNode.compute_hist(
            data, mobj.nobs, mobj.nvars, mobj.disc_method, mobj.ftype, mobj.itype
        )
        mobj.set_nodes_data(bins_list, hist_list)
        mobj.prep_pair_data(mobj.hist_dim, mobj.nvars, save_hist)
        for pindex, (i, j) in enumerate(
            itertools.combinations(range(mobj.nvars), 2)
        ):

            idata, jdata = data[:mobj.nobs, i], data[:mobj.nobs, j]
            inode_data = mobj.get_node_data(i)
            jnode_data = mobj.get_node_data(j)
            npair = RVNodePair.from_node_data(
                (idata, *inode_data),
                (jdata, *jnode_data),
                mobj.nobs,
                tbase,
                disc_method,
                mobj.ftype,
                mobj.itype
            )
            mobj.set_pair_data(pindex, i, j, npair)
        return mobj

    @classmethod
    def from_pair_list_data(cls, npld: MRVNodePairs, save_hist: bool = True):
        mobj = cls()
        mobj.nobs, mobj.nvars = npld.nobs, npld.nvars
        mobj.disc_method = npld.disc_method
        mobj.tbase = npld.tbase
        mobj.set_nodes_data(
            [nx.bins for nx in npld.nodes],
            [nx.hist for nx in npld.nodes],
        )
        mobj.prep_pair_data(mobj.hist_dim, mobj.nvars, save_hist)
        for (i, j), npair in npld.node_pairs.items():
            pindex = mobj.pair_lookup[(i,j)]
            mobj.set_pair_data(pindex, i, j, npair)
        return mobj

    @classmethod
    def from_nodes_and_pairs(
        cls,
        nodes: list[RVNode],
        node_pairs: list[RVPairData],
        dshape: tuple[int, int],
        disc_method: DiscretizerMethod,
        tbase: LogBase,
        save_hist: bool = True
    ):
        mobj = cls()
        mobj.nobs, mobj.nvars = dshape
        mobj.disc_method = disc_method
        mobj.tbase = tbase
        mobj.set_nodes_data(
            [nx.bins for nx in nodes],
            [nx.hist for nx in nodes],
        )
        mobj.prep_pair_data(mobj.hist_dim, mobj.nvars, save_hist)
        for pindex, i, j, npair in node_pairs:
            mobj.set_pair_data(pindex, i, j, npair)
        return mobj


class MISIDataH5(MRVInterface):
    h5_file: str
    h5_fptr: h5py.File | None
    disc_method: DiscretizerMethod = "bayesian_blocks"
    tbase: LogBase = '2'
    #
    nvars: int = 0
    nobs: int = 0
    npairs: int = 0
    nsjv_dim: int = 0
    nsi: int = 0
    #
    hist_dim_cache:  NDIntArray | None = None
    hist_start_cache:  NDIntArray | None = None
    si_start_cache:  NDIntArray | None = None
    #

    def __load_cache(self, data_grp: h5py.Group):
        self.hist_dim_cache   = data_grp["hist_dim"][:]    # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
        self.hist_start_cache = data_grp["hist_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
        self.si_start_cache = data_grp["si_start"][:]  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]

    def __init__(
        self,
        h5_file: str,
        *args: t.Any,
        load_cache: bool=True,
        open_mode: t.Literal['r','a'] | None = None,
        **kwargs: t.Any
    ):
        super().__init__(*args, **kwargs)
        self.h5_file = h5_file
        self.h5_fptr = None
        with h5py.File(h5_file, 'r') as fptr:
            data_grp : h5py.Group = t.cast(h5py.Group, fptr["data"])
            self.disc_method = data_grp.attrs["disc_method"] # pyright: ignore[reportAttributeAccessIssue]
            self.tbase       = data_grp.attrs["tbase"] # pyright: ignore[reportAttributeAccessIssue]
            self.nvars       = int(data_grp.attrs["nvars"])  # pyright: ignore[reportArgumentType]
            self.nobs        = int(data_grp.attrs["nobs"])   # pyright: ignore[reportArgumentType]
            self.npairs      = int(data_grp.attrs["npairs"])  # pyright: ignore[reportArgumentType]
            self.nsjv_dim    = int(data_grp.attrs["nsjv_dim"]) # pyright: ignore[reportArgumentType]
            self.nsi         = int(data_grp.attrs["nsi"])  # pyright: ignore[reportArgumentType]
            if load_cache:
                self.__load_cache(data_grp)
        if open_mode:
            self.open(open_mode)

    def hist_start(self, i: int) -> IntegerT:
        if self.hist_start_cache is not None:
            return self.hist_start_cache[i]
        return self.h5_fptr["/data/hist_start"][i] # pyright: ignore[reportReturnType, reportIndexIssue]

    @t.override
    def get_hist_dim(self, i: int) -> IntegerT:
        return self.hist_dim(i)

    def hist_dim(self, i: int) -> IntegerT:
        if self.hist_dim_cache is not None:
            return self.hist_dim_cache[i]
        return self.h5_fptr["/data/hist_dim"][i] # pyright: ignore[reportReturnType, reportIndexIssue]

    def si_start(self, i: int) -> IntegerT:
        if self.si_start_cache is not None:
            return self.si_start_cache[i]
        return self.h5_fptr["/data/si_start"][i] # pyright: ignore[reportReturnType, reportIndexIssue]

    def get_si_start(self, about: int):
        return self.si_start(about)

    def get_si_range(self, from_idx:int, to_idx:int) -> NDFloatArray:
        return self.h5_fptr["/data/si"][from_idx:to_idx] # pyright: ignore[reportReturnType, reportIndexIssue]

    def get_lmr_range(self, from_idx:int, to_idx:int) -> NDFloatArray:
        return self.h5_fptr["/data/lmr"][from_idx:to_idx] # pyright: ignore[reportReturnType, reportIndexIssue]

    def si_begin(self, about: int, by: int):
        return self.si_start(about) + (by * self.hist_dim(about))

    def si_bounds(self, about: int, by: int):
        bstart = self.si_begin(about, by)
        bend = bstart + self.hist_dim(about)
        return bstart, bend

    def si_range(self, about:int, by: int):
        bstart, bend = self.si_bounds(about, by)
        return range(bstart, bend)

    def jv_start(self, i: int) -> IntegerT:
        return self.h5_fptr["/data/jv_start"][i] # pyright: ignore[reportReturnType, reportIndexIssue]

    def jv_dim(self, i: int) -> IntegerT:
        return self.h5_fptr["/data/jv_dim"][i] # pyright: ignore[reportReturnType, reportIndexIssue]

    def pair_si_bounds(self, i: int, j: int):
        return (
            self.si_bounds(about=i, by=j),
            self.si_bounds(about=j, by=i),
        )

    def pair_si_range(self, i: int, j: int):
        return (
            self.si_range(about=i, by=j),
            self.si_range(about=j, by=i),
        )

    def set_pair_data(self, pindex: int, i: int, j: int, npair: RVNodePair):
        self.h5_fptr["/data/mi"][pindex] = npair.mi    # pyright: ignore[reportIndexIssue]

        #
        jvstart = self.jv_start(pindex)
        jvsize = self.jv_dim(pindex)
        jvstop = jvstart + jvsize
        if npair.sthist is not None:
            if jvsize != npair.sthist.size:
                print(i, j, pindex, jvsize, npair.sthist.size)
            assert jvsize == npair.sthist.size
            self.h5_fptr["/data/pair_hist"][jvstart:jvstop] = npair.sthist.reshape(jvsize)  # pyright: ignore[reportIndexIssue]
        if npair.ljvi is not None:
            self.h5_fptr["/data/pair_jvir"][jvstart:jvstop] = npair.ljvi.reshape(jvsize)  # pyright: ignore[reportIndexIssue]
        #
        isi_range, jsi_range = self.pair_si_range(i, j)
        self.h5_fptr["/data/si"][isi_range] = npair.x_si    # pyright: ignore[reportIndexIssue]
        self.h5_fptr["/data/si"][jsi_range] = npair.y_si    # pyright: ignore[reportIndexIssue]
        self.h5_fptr["/data/lmr"][isi_range] = npair.x_lmr    # pyright: ignore[reportIndexIssue]
        self.h5_fptr["/data/lmr"][jsi_range] = npair.y_lmr    # pyright: ignore[reportIndexIssue]



    def open(self, open_mode: t.Literal['r','a'] = 'r' ):
        self.h5_fptr = h5py.File(self.h5_file, open_mode)
        return self

    def close(self):
        if self.h5_fptr:
            self.h5_fptr.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type: type[Exception], exc_value: Exception, exc_traceback: t.Any):
        self.close()

    @t.override
    def float_dtype(self) -> NPDType:
        return np.float32

    @t.override
    def int_dtype(self) -> NPDType:
        return np.int32

    @t.override
    def index_dtype(self) -> NPDType:
        return np.int64

    @t.override
    def get_mi(self, i:int, j:int) -> float | NPFloat:
        idx = triu_pair_to_index(self.nvars, i, j)
        return self.h5_fptr["/data/mi"][idx] # pyright: ignore[reportReturnType, reportIndexIssue]

    @t.override
    def get_si(self, about:int, by:int) -> NDFloatArray:
        bstart, bend = self.si_bounds(about=about, by=by)
        return self.h5_fptr["/data/si"][bstart:bend] # pyright: ignore[reportReturnType, reportIndexIssue]

    @t.override
    def si_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self.si_begin(about=about, by=by) + rstate
        return self.h5_fptr["/data/si"][si_idx] # pyright: ignore[reportReturnType, reportIndexIssue]

    @t.override
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        bstart, bend = self.si_bounds(about=about, by=by)
        return self.h5_fptr["/data/lmr"][bstart:bend] # pyright: ignore[reportReturnType, reportIndexIssue]

    @t.override
    def lmr_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self.si_begin(about=about, by=by) + rstate
        return self.h5_fptr["/data/lmr"][si_idx] # pyright: ignore[reportReturnType, reportIndexIssue]

    @property
    @t.override
    def ndata(self) -> float | NPFloat:
        return float(self.nobs)

    @property
    @t.override
    def nvariables(self):
        return self.nvars

    @t.override
    def get_hist(self, i: int) -> NDFloatArray:
        hstart = self.hist_start(i)
        hend = self.hist_start(i) + self.hist_dim(i)
        return self.h5_fptr["/data/hist"][hstart:hend] # pyright: ignore[reportReturnType, reportIndexIssue]


@t.final
class MISIPair(MRVInterface):
    nobs: int
    nvars: int
    mi : FloatT
    #
    idx: DataPair[IntegerT]
    hist_dim: DataPair[IntegerT]
    #
    hist: DataPair[NDFloatArray]
    si: DataPair[NDFloatArray]
    lmr: DataPair[NDFloatArray]
    #
    lmr_ds: DataPair[LMRDataStructure] | DataPair[LMRSubsetDataStructure] 
    #
    ftype: NPDType = np.float32
    itype: NPDType = np.int32
    idx_type: NPDType = np.int64
    #
    subset_var: list[int] = []
    subset_map: dict[int, int] = {}


    def __init__(
        self,
        rshape : tuple[int, int],
        pidx : DataPair[IntegerT],
        mi: FloatT,
        phist: DataPair[NDFloatArray],
        phist_dim: DataPair[IntegerT],
        psi: DataPair[NDFloatArray],
        plmr: DataPair[NDFloatArray],
        subset_var: list[int] | NDIntArray | None = None,
    ):
        super().__init__()
        self.nobs, self.nvars = rshape
        self.idx = pidx
        self.mi = mi
        self.hist = phist
        self.hist_dim = phist_dim
        self.si = psi
        self.lmr = plmr
        self.subset_var =  []
        self.subset_map =  {}
        self._set_subset_var(subset_var)
        self._init_lmr_ds()

    def _set_subset_var(self, subset_var: list[int] | NDIntArray | None):
        if subset_var is not None:
            self.subset_var = [int(x) for x in subset_var]
            self.subset_map = dict(zip(self.subset_var,
                                       range(len(self.subset_var))))

    def _init_lmr_ds(self):
        if self.subset_var:
            self.lmr_ds  = DataPair[LMRSubsetDataStructure](
                LMRSubsetDataStructure(self, int(self.idx.first),
                                       self.subset_var, self.subset_map),
                LMRSubsetDataStructure(self, int(self.idx.second),
                                        self.subset_var, self.subset_map)
            )
        else:
            self.lmr_ds  = DataPair[LMRDataStructure](
                LMRDataStructure(self, int(self.idx.first)),
                LMRDataStructure(self, int(self.idx.second))
            )

    def init_subset_var(self, subset_var: list[int] | NDIntArray | None):
        self._set_subset_var(subset_var)
        self._init_lmr_ds()

    def _select_first_si(self): # pyright: ignore[reportUnusedFunction]
        return self.si.first

    def _select_second_si(self): # pyright: ignore[reportUnusedFunction]
        return self.si.second

    def _select_si(self, selector:int):
        return self.si.first if selector == self.idx.first else self.si.second

    def _select_lmr(self, selector:int):
        return self.lmr.first if selector == self.idx.first else self.lmr.second

    def _si_begin(self, about: int, by: int):
        return by * self.get_hist_dim(about)

    def _si_bounds(self, about: int, by: int):
        bstart = self._si_begin(about, by)
        bend = bstart + self.get_hist_dim(about)
        return range(bstart, bend)

    @t.override
    def index_dtype(self) -> NPDType:
        return self.idx_type

    @t.override
    def float_dtype(self) -> NPDType:
        return self.ftype

    @t.override
    def int_dtype(self) -> NPDType:
        return self.itype

    @t.override
    def get_mi(self, i:int, j:int) -> float | NPFloat:
        return self.mi

    @t.override
    def get_si(self, about:int, by:int) -> NDFloatArray:
        return self._select_si(about)[self._si_bounds(about=about, by=by)]

    @t.override
    def si_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self._si_begin(about, by) + rstate
        return self._select_si(about)[si_idx]

    @t.override
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        return self._select_lmr(about)[self._si_bounds(about=about, by=by)]

    @t.override
    def lmr_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self._si_begin(about, by) + rstate
        return self._select_lmr(about)[si_idx]

    @property
    @t.override
    def ndata(self) -> float | NPFloat:
        return float(self.nobs)

    @property
    @t.override
    def nvariables(self):
        return self.nvars
 
    @t.override
    def get_lmr_minsum(self, about:int, target:int) -> FloatT:
        if about == self.idx.first:
            return self.lmr_ds.first.lmr_minsum(target)
        return self.lmr_ds.second.lmr_minsum(target)

    @t.override
    def get_hist(self, i: int) -> NDFloatArray:
        return self.hist.first if i == self.idx.first else self.hist.second

    @t.override
    def get_hist_dim(self, i: int) -> IntegerT:
        return self.hist_dim.first if i == self.idx.first else self.hist_dim.second

    @classmethod
    def from_misidata(cls, misd: MISIData | MISIDataH5, i: int, j:int):
        ihist_dim = misd.get_hist_dim(i)
        ibstart = int(misd.get_si_start(i))
        ibend = ibstart + misd.nvariables * ihist_dim
        #
        jhist_dim = misd.get_hist_dim(j)
        jbstart = int(misd.get_si_start(j))
        jbend = jbstart + misd.nvariables * jhist_dim
        return cls(
            (misd.nobs, misd.nvars),
            DataPair[IntegerT](i, j),
            misd.get_mi(i, j),
            DataPair[NDFloatArray](misd.get_hist(i), misd.get_hist(j)),
            DataPair[IntegerT](ihist_dim, jhist_dim),
            DataPair[NDFloatArray](
                misd.get_si_range(ibstart, ibend),
                misd.get_si_range(jbstart, jbend)
            ),
            DataPair[NDFloatArray](
                misd.get_lmr_range(ibstart, ibend),
                misd.get_lmr_range(jbstart, jbend)
            )
        )

    @t.override
    def get_puc_factor(self, about: int, target:int):
        if about == self.idx.first:
            return self.lmr_ds.first.mi_factor(target)
        return self.lmr_ds.second.mi_factor(target)

    @t.override
    def compute_lmr_puc(self, i:int, j:int):
        if not self.subset_var:
            return super().compute_lmr_puc(i, j)
        mij = self.get_mi(i, j)
        return (
            (self.get_puc_factor(i, j) - (self.get_lmr_minsum(i, j) / mij) ) +
            (self.get_puc_factor(j, i) - (self.get_lmr_minsum(j, i) / mij) )
        )


@t.final
class MISIRangePair(MRVInterface):
    nobs: int
    nvars: int
    npairs: int
    nsi: int
    nsjv_dim: int
    #
    hist_dim: DataPair[NDIntArray]
    hist_start: DataPair[NDIntArray]
    # si_start: DataPair[NDIntArray]
    range_hist_start: DataPair[NDIntArray]
    range_si_start: DataPair[NDIntArray]
    #
    pair_lookup: dict[tuple[int, int], int]
    #
    mi_cache : NDFloatArray | None
    mi : NDFloatArray | None
    #
    st_ranges: DataPair[range]
    #
    hist: DataPair[NDFloatArray]
    si: DataPair[NDFloatArray]
    lmr: DataPair[NDFloatArray]
    #
    lmr_ds: DataPair[list[LMRDataStructure] | list[LMRSubsetDataStructure]]
    #
    ftype: NPDType = np.float32
    itype: NPDType = np.int32
    idx_type: NPDType = np.int64
    #
    subset_var: list[int] = []
    subset_map: dict[int, int] = {}


    def __init__(
        self,
        h5_file: str,
        st_ranges: tuple[range, range],
        mi_cache: NDFloatArray | None,
        load_ds_flag: bool = False,
        subset_var: list[int] | None = None,
    ):
        super().__init__()
        src_range, tgt_range = st_ranges
        self.st_ranges = DataPair(src_range, tgt_range)
        self.h5_file = h5_file
        with h5py.File(h5_file) as fptr:
            data_grp : h5py.Group = t.cast(h5py.Group, fptr["data"])
            self.nvars       = int(data_grp.attrs["nvars"])   # pyright: ignore[reportArgumentType]
            self.nobs        = int(data_grp.attrs["nobs"])   # pyright: ignore[reportArgumentType]
            self.npairs      = int(data_grp.attrs["npairs"])  # pyright: ignore[reportArgumentType]
            self.nsjv_dim    = int(data_grp.attrs["nsjv_dim"]) # pyright: ignore[reportArgumentType]
            self.nsi         = int(data_grp.attrs["nsi"]) # pyright: ignore[reportArgumentType]

            self.hist_dim  = DataPair(
                data_grp["hist_dim"][src_range],    # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                data_grp["hist_dim"][tgt_range],    # pyright: ignore[reportIndexIssue]
            )
            self.hist_start = DataPair(
                data_grp["hist_start"][src_range],  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                data_grp["hist_start"][tgt_range],  # pyright: ignore[reportIndexIssue]
            )
            si_start : DataPair[NDIntArray] = DataPair(  # pyright: ignore[reportAssignmentType]
                data_grp["si_start"][src_range],  # pyright: ignore[reportIndexIssue]
                data_grp["si_start"][tgt_range]   # pyright: ignore[reportIndexIssue]
            )

            src_hist_sum = sum(self.hist_dim.first)
            src_start = self.hist_start.first[0]
            src_stop = src_start + src_hist_sum
            tgt_hist_sum = sum(self.hist_dim.second)
            tgt_start = self.hist_start.second[0]
            tgt_stop = tgt_start + tgt_hist_sum
            self.hist = DataPair(
                data_grp["hist"][src_start:src_stop],   # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                data_grp["hist"][tgt_start:tgt_stop]    # pyright: ignore[reportIndexIssue]
            )

            src_start = si_start.first[0]
            tgt_start = si_start.second[0]
            src_stop = src_start + (src_hist_sum * self.nvars)
            tgt_stop = tgt_start + (tgt_hist_sum * self.nvars)
            self.si = DataPair(
                data_grp["si"][src_start:src_stop],  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                data_grp["si"][tgt_start:tgt_stop]   # pyright: ignore[reportIndexIssue]
            )

            self.lmr = DataPair(
                data_grp["lmr"][src_start:src_stop],  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                data_grp["lmr"][tgt_start:tgt_stop]   # pyright: ignore[reportIndexIssue]
            )

            if mi_cache is None:
                 pair_list = sorted([
                    (s, t) for t in tgt_range for s in src_range if s < t
                 ])
                 npairs = len(pair_list)
                 pair_indices = [triu_pair_to_index(self.nvars, s, t) for s,t in pair_list]
                 self.pair_lookup = {x: y for x, y in zip(pair_list, range(npairs))}
                 self.mi = data_grp["mi"][pair_indices]    # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
                 self.mi_cache = None
            else:
                self.mi_cache = mi_cache
                self.pair_lookup = {}
                self.mi = None
        #
        fi_hdim = np.zeros(self.hist_dim.first.size, self.index_dtype())
        for i in range(1, self.hist_dim.first.size):
            fi_hdim[i] = fi_hdim[i-1] + self.hist_dim.first[i-1]
        # fi_hdim = fi_hdim * self.nvars
        se_hdim = np.zeros(self.hist_dim.second.size, self.index_dtype())
        for i in range(1, self.hist_dim.second.size):
            se_hdim[i] = se_hdim[i-1] + self.hist_dim.second[i-1]
        # se_hdim = se_hdim * self.nvars
        #
        self.range_hist_start = DataPair(fi_hdim, se_hdim)
        self.range_si_start = DataPair(fi_hdim * self.nvars,
                                       se_hdim * self.nvars)
        self.lmr_ds  = DataPair([],[])
        self.init_subset_var(subset_var, load_ds_flag)

    def _set_subset_var(self, subset_var: list[int] | NDIntArray | None):
        if subset_var is not None:
            self.subset_var = [int(x) for x in subset_var]
            self.subset_map = dict(zip(self.subset_var,
                                       range(len(self.subset_var))))

    def _init_lmr_ds(self):
        if self.subset_var:
            self.lmr_ds  = DataPair(
                [
                    LMRSubsetDataStructure(self, int(ix),
                                           self.subset_var, self.subset_map)
                    for ix in self.st_ranges.first
                ],
                [
                    LMRSubsetDataStructure(self, int(jx),
                                           self.subset_var, self.subset_map)
                    for jx in self.st_ranges.second
                ]
            )
        else:
            self.lmr_ds  = DataPair(
                [LMRDataStructure(self, int(ix)) for ix in self.st_ranges.first],
                [LMRDataStructure(self, int(jx)) for jx in self.st_ranges.second]
            )

    def init_subset_var(
            self,
            subset_var: list[int] | NDIntArray | None, 
            load_ds_flag: bool
    ):
        self._set_subset_var(subset_var)
        if load_ds_flag: 
            self._init_lmr_ds()

    @t.override
    def float_dtype(self) -> NPDType:
        return self.ftype

    @t.override
    def int_dtype(self) -> NPDType:
        return self.itype

    @t.override
    def index_dtype(self) -> NPDType:
        return self.idx_type

    @property
    @t.override
    def ndata(self) -> float | NPFloat:
        return float(self.nobs)

    @property
    @t.override
    def nvariables(self):
        return self.nvars

    def get_first_hist(self, i: int) -> NDFloatArray:
        di  = i - self.st_ranges.first.start
        hist_start = self.range_hist_start.first[di]
        hist_stop = hist_start + self.hist_dim.first[di]
        return self.hist.first[hist_start:hist_stop]

    def get_second_hist(self, i: int) -> NDFloatArray:
        di  = i - self.st_ranges.second.start
        hist_start = self.range_hist_start.second[di]
        hist_stop = hist_start + self.hist_dim.second[di]
        return self.hist.second[hist_start:hist_stop]

    @t.override
    def get_hist(self, i: int) -> NDFloatArray:
        src_flag, di = self._var_loc(i)
        if src_flag:
            hist_start = self.range_hist_start.first[di]
            hist_stop = hist_start + self.hist_dim.first[di]
            return self.hist.first[hist_start:hist_stop]
        hist_start = self.range_hist_start.second[di]
        hist_stop = hist_start + self.hist_dim.second[di]
        return self.hist.second[hist_start:hist_stop]

    def _select_si(self, selector:int):
        return self.si.first if self._is_first(selector) else self.si.second

    def _select_lmr(self, selector:int):
        return self.lmr.first if self._is_first(selector) else self.lmr.second

    def _is_first(self, i:int):
        if (self.st_ranges.first.start <= i) and (i < self.st_ranges.first.stop):
            return True
        return False

    def _var_loc(self, i:int) -> tuple[bool, IntegerT]:
        if self._is_first(i):
            return True, i - self.st_ranges.first.start
        return False, i - self.st_ranges.second.start

    def get_first_hist_dim(self, i: int) -> IntegerT:
        di = i - self.st_ranges.first.start
        return self.hist_dim.first[di]

    def get_second_hist_dim(self, i: int) -> IntegerT:
        di = i - self.st_ranges.second.start
        return self.hist_dim.second[di]

    @t.override
    def get_hist_dim(self, i: int) -> IntegerT:
        src_flag, di = self._var_loc(i)
        return self.hist_dim.first[di] if src_flag else self.hist_dim.second[di]

    def _si_begin(self, about: int, by: int) -> IntegerT:
        frst_flag, vloc = self._var_loc(about)
        if frst_flag:
            si_offset =  self.range_si_start.first[vloc]
        else:
            si_offset =  self.range_si_start.second[vloc]
        return si_offset + (by * self.get_hist_dim(about))

    def _first_si_bounds(self, about: int, by: int):
        vloc = about - self.st_ranges.first.start
        si_offset = self.range_si_start.first[vloc]
        bstart = si_offset + (by * self.get_first_hist_dim(about))
        bend = bstart + self.get_first_hist_dim(about)
        return range(bstart, bend)

    def _second_si_bounds(self, about: int, by: int):
        vloc = about - self.st_ranges.second.start
        si_offset = self.range_si_start.second[vloc]
        bstart = si_offset + (by * self.get_second_hist_dim(about))
        bend = bstart + self.get_second_hist_dim(about)
        return range(bstart, bend)

    def _si_bounds(self, about: int, by: int):
        bstart = self._si_begin(about, by)
        bend = bstart + self.get_hist_dim(about)
        return range(bstart, bend)

    @t.override
    def get_mi(self, i:int, j:int) -> float | NPFloat:
        if self.mi_cache is not None:
            pindex = triu_pair_to_index(self.nvars, i, j)
            return self.mi_cache[pindex]
        return self.mi[self.pair_lookup[(i,j)]]

    @t.override
    def redundancy_update_for(self, i: int, j: int, by: int) -> FloatT:
        assert i < j
        assert i != by
        assert j != by
        ndata = self.ndata
        iby_si = self.get_si(about=i, by=by)
        jby_si = self.get_si(about=j, by=by)
        ri = redundancy(self.get_hist(i), self.get_si(about=i, by=j), iby_si, ndata)
        rj = redundancy(self.get_hist(j), self.get_si(about=j, by=i), jby_si, ndata)
        return self.mpuc(i, j, ri) + self.mpuc(i, j, rj)

    @t.override
    def accumulate_redundancies(
        self,
        i:int,
        j:int,
        by_nodes: Iterable[int] | None
    ) -> FloatT:
        if by_nodes is None:
            by_nodes = range(self.nvariables)
        ihist = self.get_first_hist(i)
        jhist = self.get_second_hist(j)
        #
        fsi_vec = self.si.first
        ssi_vec = self.si.second
        ibj_bounds = self._first_si_bounds(about=i, by=j)
        jbi_bounds = self._second_si_bounds(about=j, by=i)
        isi_byj = fsi_vec[ibj_bounds.start: ibj_bounds.stop]
        jsi_byi = ssi_vec[jbi_bounds.start: jbi_bounds.stop]
        #
        red_value = np.float64(0.0).astype(self.float_dtype())
        for kby in by_nodes:
            if kby == i or kby == j:
                continue
            ikby_bounds = self._first_si_bounds(about=i, by=kby)
            jkby_bounds = self._second_si_bounds(about=j, by=kby)
            isi_byk = fsi_vec[ikby_bounds.start: ikby_bounds.stop]
            jsi_byk = ssi_vec[jkby_bounds.start: jkby_bounds.stop]
            ri = redundancy(ihist, isi_byj, isi_byk, self.ndata)
            rj = redundancy(jhist, jsi_byi, jsi_byk, self.ndata)
            rupdate = self.mpuc(i, j, ri) + self.mpuc(i, j, rj)
            red_value = red_value + rupdate
        return red_value

    @t.override
    def get_si(self, about:int, by:int) -> NDFloatArray:
        return self._select_si(about)[self._si_bounds(about=about, by=by)]

    @t.override
    def si_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self._si_begin(about, by) + rstate
        return self._select_si(about)[si_idx]

    @t.override
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        return self._select_lmr(about)[self._si_bounds(about=about, by=by)]

    @t.override
    def lmr_value(self, about:int, by:int, rstate:int) -> FloatT:
        si_idx = self._si_begin(about, by) + rstate
        return self._select_lmr(about)[si_idx]

    @t.override
    def get_lmr_minsum(self, about:int, target:int) -> FloatT:
        src_flag, dxa = self._var_loc(about)
        return (
            self.lmr_ds.first[dxa].lmr_minsum(target)
            if src_flag else self.lmr_ds.second[dxa].lmr_minsum(target)
        )

    @t.override
    def get_puc_factor(self, about: int, target:int):
        src_flag, dxa = self._var_loc(about)
        return np.float32(
            self.lmr_ds.first[dxa].mi_factor(target)
            if src_flag else self.lmr_ds.second[dxa].mi_factor(target)
        ).astype(self.float_dtype())

    @t.override
    def compute_lmr_puc(self, i:int, j:int):
        if not self.subset_var:
            return super().compute_lmr_puc(i, j)
        mij = self.get_mi(i, j)
        return (
            (self.get_puc_factor(i, j) - (self.get_lmr_minsum(i, j) / mij) ) +
            (self.get_puc_factor(j, i) - (self.get_lmr_minsum(j, i) / mij) )
        )
