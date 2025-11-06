import numpy as np
import itertools
import typing as t
import h5py

# from pydantic import BaseModel

from collections.abc import Generator, Iterable
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from .types import NDIntArray, NPFloat, NPInteger, NDFloatArray, NPDType
from .types import FloatT, IntegerT, DiscretizerMethod, LogBase
from .bayesian_blocks import block_bins
from .imeasures import LMR, MI, SI, log_jvi_ratio, redundancy


class PIDCNode:
    nobs: int = 0
    method: DiscretizerMethod
    bins: NDFloatArray = np.array(())
    hist: NDFloatArray = np.array(())

    def __init__(
        self,
        data: NDFloatArray,
        nobs: int | None,
        method: DiscretizerMethod,
        dtype: NPDType,
        int_dtype: NPDType,
    ):
        self.nobs=data.size if nobs is None else nobs
        self.method=method
        self.bins, self.hist = PIDCNode.compute_hist_for(
            data, nobs, method, dtype, int_dtype
        )

    @staticmethod
    def compute_hist_for(
        data: NDFloatArray,
        nobs: int | None,
        method: DiscretizerMethod,
        dtype: NPDType,
        int_dtype: NPDType,
    ):
        nobs=data.size if nobs is None else nobs
        dbins = block_bins(data[:nobs], dtype)
        dhist = np.histogram(data[:nobs], bins=dbins)[0]
        return dbins, dhist.astype(int_dtype)

    @staticmethod
    def compute_hist(
        data: NDFloatArray,
        nobs: int | None,
        nvars: int | None,
        method: DiscretizerMethod,
        dtype: NPDType,
        int_dtype: NPDType,
    ):
        nobs = nobs if nobs else data.shape[0]
        nvars = nvars if nvars else data.shape[1]
        bins_list = [
            block_bins(data[:nobs, vx], dtype)
            for vx in range(nvars)
        ]
        hist_list = [
            (np.histogram(data[:nobs, vx], bins=bins_list[vx])[0]).astype(int_dtype)
            for vx in range(nvars)
        ]
        return bins_list, hist_list

    @classmethod
    def from_data(
        cls,
        exp_data: NDFloatArray,
        nvars: int,
        nobs: int | None,
        method: DiscretizerMethod,
        dtype: NPDType,
        int_dtype: NPDType,
    ) -> list["PIDCNode"]:
        return [
            cls(exp_data[:, ix], nobs, method, dtype, int_dtype)
            for ix in range(nvars)
        ]


class PIDCPair:
    method: DiscretizerMethod = "bayesian_blocks"
    tbase: LogBase = '2'
    sthist: NDFloatArray = np.array(())
    ljvi: NDFloatArray = np.array(())
    x_si: NDFloatArray = np.array(())
    y_si: NDFloatArray = np.array(())
    x_lmr: NDFloatArray = np.array(())
    y_lmr: NDFloatArray = np.array(())
    mi: float | NPFloat = 0.0

    def __init__(
        self,
        method: DiscretizerMethod,
        tbase: LogBase,
        sthist: NDFloatArray,
        ljvi: NDFloatArray,
        mi: float | NPFloat,
        sis: tuple[NDFloatArray, NDFloatArray],
        lmrs: tuple[NDFloatArray, NDFloatArray],
    ) -> None:
        self.method=method
        self.tbase=tbase
        self.sthist=sthist
        self.ljvi=ljvi
        self.mi=mi
        self.x_si, self.y_si = sis
        self.x_lmr, self.y_lmr = lmrs


    def compute_x_lmr(self, nobs: FloatT | None):
        return LMR.about_x_from_ljvi(self.ljvi, self.sthist, nobs)

    def compute_y_lmr(self, nobs: FloatT | None):
        return LMR.about_y_from_ljvi(self.ljvi, self.sthist, nobs)

    @classmethod
    def from_nodes(
        cls,
        fnodes: tuple[PIDCNode, PIDCNode],
        fdata: tuple[NDFloatArray, NDFloatArray],
        tbase: LogBase,
        dtype: NPDType,
        int_dtype: NPDType,
    ) -> "PIDCPair":
        xnode, ynode = fnodes
        xdata, ydata = fdata
        assert xnode.nobs == ynode.nobs
        nobs = xnode.nobs
        return cls.from_node_data(
            (xdata[:nobs], xnode.bins, xnode.hist),
            (ydata[:nobs], ynode.bins, ynode.hist),
            nobs = xnode.nobs,
            tbase = tbase,
            method=xnode.method,
            dtype=dtype,
            int_dtype=int_dtype,
        )

    @classmethod
    def from_node_data(
        cls,
        ixnode: tuple[NDFloatArray, NDFloatArray, NDFloatArray],
        iynode: tuple[NDFloatArray, NDFloatArray, NDFloatArray],
        nobs: int,
        tbase: LogBase,
        method: DiscretizerMethod,
        dtype: NPDType,
        int_dtype: NPDType,
    ):
        xdata, xbins, xhist = ixnode
        ydata, ybins, yhist = iynode
        fwt = float(nobs)
        #
        sthist = np.histogram2d(xdata, ydata, bins=(xbins, ybins))[0]
        sthist = sthist.astype(int_dtype)
        #
        with np.errstate(divide = 'ignore'):
            ljvi = log_jvi_ratio(sthist, xhist, yhist, tbase, fwt)
            mi = MI.from_ljvi(ljvi, sthist, fwt)
            x_si, y_si = SI.from_ljvi(ljvi, sthist, xhist, yhist)
            x_lmr, y_lmr = LMR.from_ljvi(ljvi, sthist, fwt)
        #
        return cls(
            method=method,
            tbase=tbase,
            sthist=sthist.astype(int_dtype),
            ljvi=ljvi.astype(dtype),
            mi=mi.astype(dtype),
            sis=(x_si.astype(dtype), y_si.astype(dtype)),
            lmrs=(x_lmr.astype(dtype), y_lmr.astype(dtype)),
        )

    @classmethod
    def from_data(
        cls,
        nodes: list[PIDCNode],
        exp_data: NDFloatArray,
        nvars: int | None,
        tbase: LogBase,
        dtype: NPDType,
        int_dtype: NPDType,
    ) -> dict[tuple[int, int], "PIDCPair"]:
        nvars = len(nodes) if nvars is None else nvars
        assert nvars <= len(nodes)
        return {
            (ix, iy): cls.from_nodes(
                (nodes[ix], nodes[iy]),
                (exp_data[:, ix], exp_data[:, iy]),
                tbase,
                dtype,
                int_dtype,
            )
            for ix, iy in itertools.combinations(range(nvars), 2)
        }


class PIDCPairData(t.NamedTuple):
    pindex: int
    i: int
    j: int
    pidc_pair: PIDCPair


class PIDCInterface(ABC):

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def float_dtype(self) -> NPDType:
        pass

    @abstractmethod
    def int_dtype(self) -> NPDType:
        pass

    @abstractmethod
    def index_dtype(self) -> NPDType:
        pass

    @abstractmethod
    def get_hist(self, i: int) -> NDFloatArray:
        pass

    @abstractmethod
    def get_hist_dim(self, i: int) -> IntegerT:
        pass

    @abstractmethod
    def get_mi(self, i:int, j:int) -> FloatT:
        pass

    @abstractmethod
    def get_si(self, about:int, by:int) -> NDFloatArray:
        pass

    @abstractmethod
    def si_value(self, about: int, by: int, rstate:int) -> FloatT:
        pass

    @abstractmethod
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        pass

    @abstractmethod
    def lmr_value(self, about: int, by: int, rstate:int) -> FloatT:
        pass

    @property
    @abstractmethod
    def ndata(self) -> float | NPFloat:
        pass

    @property
    @abstractmethod
    def nvariables(self) -> int:
        pass

    def get_redundancies(
        self, i: int, j: int, k: int
    ) -> tuple[FloatT, FloatT, FloatT]:
        return (
            redundancy(
                self.get_hist(i),
                self.get_si(about=i, by=j),
                self.get_si(about=i, by=k),
                self.ndata,
            ),
            redundancy(
                self.get_hist(j),
                self.get_si(about=j, by=i),
                self.get_si(about=j, by=k),
                self.ndata,
            ),
            redundancy(
                self.get_hist(k),
                self.get_si(about=k, by=i),
                self.get_si(about=k, by=j),
                self.ndata
            )
        )

    def mpuc(self, i: int, j: int, redundancy: float | NPFloat):
        mi: float | NPFloat = self.get_mi(i, j)
        puc_score = (mi - redundancy) / mi
        return puc_score if np.isfinite(puc_score) and puc_score >= 0 else 0.0

    def redundancy_updates(self, i: int, j: int , k: int):
        assert i < j
        ri, rj, rk = self.get_redundancies(i, j, k)
        return (
            self.mpuc(i, j, ri) + self.mpuc(i, j, rj),
            self.mpuc(i, k, ri) + self.mpuc(i, k, rk),
            self.mpuc(j, k, rj) + self.mpuc(j, k, rk)
        )

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

    def accumulate_redundancies(
        self,
        i:int,
        j:int,
        by_nodes: Iterable[int] | None
    ) -> FloatT:
        if by_nodes is None:
            by_nodes = range(self.nvariables)
        red_val = np.float32(0).astype(self.float_dtype())
        for kby in by_nodes:
            if kby == i or kby == j:
                continue
            red_val = red_val + self.redundancy_update_for(i, j, kby)
        return red_val

    def compute_puc_matrix(self, dtype:NPDType) -> NDFloatArray:
        if self.nvariables == 0:
            return np.array(())
        nvars = self.nvariables
        puc_network = np.zeros((nvars, nvars), dtype=dtype)
        #
        for i, j, k in itertools.combinations(range(nvars), 3):
            rij, rik, rjk = self.redundancy_updates(i, j, k)
            puc_network[i, j] += rij
            puc_network[i, k] += rik
            puc_network[j, k] += rjk
        #
        for i, j in itertools.combinations(range(nvars), 2):
            puc_network[j, i] = puc_network[i, j]
        return puc_network

    def compute_puc_matrix_for(
        self,
        by_nodes: list[int],
        dtype:NPDType
    ) -> NDFloatArray:
        if self.nvariables == 0:
            return np.array(())
        nvars = self.nvariables
        puc_network = np.zeros((nvars, nvars), dtype=dtype)
        #
        for i, j in itertools.combinations(range(nvars), 2):
            puc_network[i, j]  = self.accumulate_redundancies(i, j, by_nodes)
        #
        for i, j in itertools.combinations(range(nvars), 2):
            puc_network[j, i] = puc_network[i, j]
        return puc_network

    def compute_redundancies(self) -> dict[tuple[int, int, int], FloatT]:
        nvars = self.nvariables
        if nvars == 0:
            return {}
        red_dict = {}
        for i, j, k in itertools.combinations(range(nvars), 3):
            rij, rik, rjk = self.get_redundancies(i, j, k)
            red_dict[(i, j, k)] = rij
            red_dict[(i, k, j)] = rik
            red_dict[(j, k, i)] = rjk
        return red_dict

    def minsum_list(self, about: int, target: int) -> NDFloatArray:
        by_nodes: list[int] = list(
            x for x in range(self.nvariables) if x != about and x != target
        )
        lmr_abtgt = self.get_lmr(about=about, by=target)
        min_sum_list = np.array([
            np.sum(np.minimum(lmr_abtgt, self.get_lmr(about=about, by=by)))
            for by in by_nodes
        ], self.float_dtype())
        return min_sum_list

    def get_lmr_minsum(self, about:int, target:int) -> FloatT:
        by_nodes: list[int] = list(
            x for x in range(self.nvariables) if x != about and x != target
        )
        lmr_abtgt = self.get_lmr(about=about, by=target)
        min_sum_list = np.array([
            np.sum(np.minimum(lmr_abtgt, self.get_lmr(about=about, by=by)))
            for by in by_nodes
        ], self.float_dtype())
        return np.sum(min_sum_list)

    def compute_lmr_puc(self, i:int, j:int):
        mij = self.get_mi(i, j)
        mi_factor = mij * (self.nvariables - 2)
        return (
            ( (mi_factor - self.get_lmr_minsum(i, j)) / mij ) +
            ( (mi_factor - self.get_lmr_minsum(j, i)) / mij )
        )
    

class PIDCPairListData(PIDCInterface):
    disc_method: DiscretizerMethod = "bayesian_blocks"
    tbase: LogBase = '2'
    #
    nvars: int
    nobs: int
    npairs: int
    #
    data: NDFloatArray = np.array(())
    nodes: list[PIDCNode] = []
    node_pairs: dict[tuple[int, int], PIDCPair] = {}

    def __init__(
        self,
        nvars: int,
        nobs: int,
        npairs: int,
        nodes: list[PIDCNode],
        node_pairs: dict[tuple[int, int], PIDCPair],
        data: NDFloatArray | None = None,
        tbase: LogBase = '2',
        disc_method: DiscretizerMethod = 'bayesian_blocks',
    ):
        super().__init__()
        self.disc_method = disc_method
        self.tbase = tbase
        #
        self.nvars  =  nvars
        self.nobs   =  nobs
        self.npairs =  npairs
        #
        self.data = np.array(()) if data is None else data 
        self.nodes = nodes
        self.node_pairs = node_pairs

    @t.override
    def get_mi(self, i:int, j:int) -> FloatT:
        return self.node_pairs[(i, j)].mi

    @t.override
    def get_hist(self, i: int) -> NDFloatArray:
        return self.nodes[i].hist

    @t.override
    def get_hist_dim(self, i: int) -> IntegerT:
        return self.nodes[i].hist.size

    @t.override
    def get_si(self, about:int, by:int) -> NDFloatArray:
        if about < by:
            return self.node_pairs[(about, by)].x_si
        else:
            return self.node_pairs[(by, about)].y_si

    @t.override
    def si_value(self, about: int, by: int, rstate:int) -> FloatT:
        return self.get_si(about, by)[ rstate]

    @t.override
    def get_lmr(self, about:int, by:int) -> NDFloatArray:
        if about < by:
            return self.node_pairs[(about, by)].x_lmr
        else:
            return self.node_pairs[(by, about)].y_lmr

    @t.override
    def lmr_value(self, about: int, by: int, rstate:int) -> FloatT:
        return self.get_lmr(about, by)[rstate]

    def get_ljvr(self, i:int, j:int):
        assert i < j
        return self.node_pairs[(i, j)].ljvi

    def get_joint_hist(self, i:int, j:int):
        assert i < j
        return self.node_pairs[(i, j)].sthist

    def compute_lmr(self, about:int, by:int):
        if about < by:
            return self.node_pairs[(about, by)].compute_x_lmr(self.nobs)
        else:
            return self.node_pairs[(by, about)].compute_y_lmr(self.nobs)

    @property
    @t.override
    def ndata(self):
        return float(self.nobs)

    @property
    @t.override
    def nvariables(self):
        return len(self.nodes) if self.nvars == 0 else self.nvars

    @t.override
    def float_dtype(self) -> NPDType:
        return self.data.dtype

    @t.override
    def int_dtype(self) -> NPDType:
        return np.int32

    @t.override
    def index_dtype(self) -> NPDType:
        return np.int64

    @classmethod
    def from_data(
        cls,
        data: NDFloatArray,
        shape: tuple[int, int],
        dmethod: DiscretizerMethod,
        tbase: LogBase,
        dtype: NPDType,
        int_dtype: NPDType,
    ):
        nobs, nvars = shape
        nodes = PIDCNode.from_data(data, nvars, nobs, dmethod, dtype, int_dtype)
        node_pairs = PIDCPair.from_data(nodes, data, nvars, tbase, dtype, int_dtype)
        return PIDCPairListData(
            disc_method=dmethod,
            tbase=tbase,
            nvars=nvars,
            nobs=nobs,
            npairs=len(node_pairs),
            data=data,
            nodes=nodes,
            node_pairs=node_pairs,
        )

    @classmethod
    def from_pairs(cls, nodes: list[PIDCNode], node_pairs: list[PIDCPair]):
        nvars = len(nodes)
        np_keys = [(i, j) for i,j in itertools.combinations(range(nvars), 2)]
        npairs_dict = {(i, j): npx for (i, j), npx in zip(np_keys, node_pairs)}
        return  cls(
            nobs=nodes[0].nobs if nodes else 0,
            nvars=nvars,
            npairs=len(node_pairs),
            nodes=nodes,
            node_pairs=npairs_dict
        )

@t.final
class LMRDataStructure:
    nvars: int
    about: int
    dim: int
    lmr_sorted: NDFloatArray
    lmr_rank: NDIntArray 
    lmr_pfxsum: NDFloatArray

    def __init__(self, pidata: PIDCInterface, about:int):
        self.nvars = pidata.nvariables
        self.about = about
        self.dim = int(pidata.get_hist_dim(about))
        lmr_size: int = self.dim * self.nvars
        self.lmr_sorted = np.zeros(lmr_size, dtype=pidata.float_dtype())
        self.lmr_pfxsum = np.zeros(lmr_size, dtype=pidata.float_dtype())
        self.lmr_rank = np.zeros(lmr_size, dtype=pidata.int_dtype())
        self.__init_ds(pidata, about)

    def _build_ds(self, si_values_list: list[NDArray[t.Any]]):
        for rs, si_values in enumerate(si_values_list):
            # print(si_values, rs, self.about_dim, rsbegin)
            rsbegin = rs * self.nvars
            curr_sum = 0.0
            for ix, (svx, by) in enumerate(si_values):
                curr_sum += svx
                self.lmr_rank[rsbegin + by] = ix
                self.lmr_sorted[rsbegin + ix] = svx
                self.lmr_pfxsum[rsbegin + ix] = curr_sum

    def _si_values_list(self, pidata: PIDCInterface, about: int):
        vl_dtype = np.dtype([
            ('v', self.lmr_sorted.dtype), ('i', self.lmr_rank.dtype)
        ])
        dvalue = np.float32(0).astype(self.lmr_sorted.dtype)
        svlist = [
            np.fromiter(
                zip(itertools.repeat(dvalue), range(self.nvars)),
                dtype=vl_dtype
            )
            for _ in range(self.dim)
        ]
        for by in range(self.nvars):
            lmr_ax = pidata.get_lmr(about, by)
            for rs in range(self.dim):
                svlist[rs][by] = (lmr_ax[rs], by)
        return svlist
    
    def _sort_si_values(self, si_values_list: list[NDArray[t.Any]]):
        for si_values in si_values_list:
            si_values.sort()
        return si_values_list

    def __init_ds(self, pidata: PIDCInterface, about: int):
        #
        si_values_list = self._si_values_list(pidata, about)
        si_values_list = self._sort_si_values(si_values_list)
        self._build_ds(si_values_list)

    def minsum_list(self, src: int) -> NDFloatArray:
        rstates = np.arange(self.dim)
        rs_begin = np.multiply(rstates, self.nvars)
        src_ranks = self.lmr_rank[np.add(rs_begin, src)]
        src_locs = np.add(rs_begin, src_ranks)
        lmr_lows = self.lmr_pfxsum[np.subtract(src_locs, 1)]
        lmr_highs = np.multiply(
            self.lmr_sorted[src_locs],
            np.subtract(self.nvars - 1, src_ranks)
        )
        lmr_states = lmr_lows + lmr_highs
        return lmr_states
 
    def lmr_minsum_vec(self, src:int):
        rstates = np.arange(self.dim)
        rs_begin = np.multiply(rstates, self.nvars)
        src_ranks = self.lmr_rank[np.add(rs_begin, src)]
        src_locs = np.add(rs_begin, src_ranks)
        lmr_lows = self.lmr_pfxsum[np.subtract(src_locs, 1)]
        lmr_highs = np.multiply(
            self.lmr_sorted[src_locs],
            np.subtract(self.nvars - 1, src_ranks)
        )
        lmr_states = lmr_lows + lmr_highs
        #return src_ranks, lmr_lows, lmr_highs, lmr_states
        return np.sum(lmr_states)

    def lmr_minsum_iter(self, src:int):
        rdsum = np.float32(0).astype(self.lmr_sorted.dtype)
        lmvalues = np.zeros(self.dim, self.lmr_sorted.dtype)
        lmsums = np.zeros(self.dim, self.lmr_sorted.dtype)
        for rs in range(self.dim):
            rsbegin = rs * self.nvars
            lmrank = self.lmr_rank[rsbegin + src]
            lmlow = self.lmr_pfxsum[rsbegin + lmrank- 1]
            lmhigh = (self.nvars - 1 - lmrank) * self.lmr_sorted[rsbegin + lmrank]
            lmvalues[rs] = lmhigh
            lmsums[rs] = lmlow
            rdsum += lmlow + lmhigh
        # print(lmvalues, lmsums)
        return rdsum

    def lmr_minsum(self, src:int):
        # print(lmvalues, lmsums)
        return self.lmr_minsum_vec(src)

@t.final
class LMRSubsetDataStructure:
    nvars: int
    about: int
    dim: int
    lmr_sorted: NDFloatArray
    lmr_pfxsum: NDFloatArray
    lmr_rank: NDIntArray
    subset_var: list[int]
    subset_map: dict[int, int]
    pidata: PIDCInterface

    def __init__(
        self,
        pidata: PIDCInterface,
        about:int,
        subset_var: list[int],
        subset_map: dict[int, int],
    ):
        self.pidata = pidata
        self.about = about
        self.subset_var = subset_var
        self.subset_map = subset_map
        self.nvars = len(subset_map)
        self.dim = int(pidata.get_hist_dim(about))
        lmr_size: int = self.dim * len(subset_map)
        self.lmr_sorted = np.zeros(lmr_size, dtype=pidata.float_dtype())
        self.lmr_pfxsum = np.zeros(lmr_size, dtype=pidata.float_dtype())
        self.lmr_rank = np.zeros(lmr_size, dtype=pidata.int_dtype())
        self.__init_ds(pidata, about)

    def _build_ds(self, si_values_list: list[NDArray[t.Any]]):
        for rs, si_values in enumerate(si_values_list):
            # print(si_values, rs, self.about_dim, rsbegin)
            rsbegin = rs * self.nvars
            curr_sum = 0.0
            for ix, (svx, by_var) in enumerate(si_values):
                by_idx = self.subset_map[by_var]
                curr_sum += svx
                self.lmr_rank[rsbegin + by_idx] = ix
                self.lmr_sorted[rsbegin + ix] = svx
                self.lmr_pfxsum[rsbegin + ix] = curr_sum

    def _si_values_list(self, pidata: PIDCInterface, about: int):
        vl_dtype = np.dtype([
            ('v', self.lmr_sorted.dtype), ('i', self.lmr_rank.dtype)
        ])
        dvalue = np.float32(0).astype(self.lmr_sorted.dtype)
        svlist = [
            np.fromiter(
                zip(itertools.repeat(dvalue), range(self.nvars)),
                dtype=vl_dtype
            )
            for _ in range(self.dim)
        ]
        for by_var, by_idx in self.subset_map.items():
            lmr_ax = pidata.get_lmr(about, by_var)
            for rs in range(self.dim):
                svlist[rs][by_idx] = (lmr_ax[rs], by_var)
        return svlist
    
    def _sort_si_values(self, si_values_list: list[NDArray[t.Any]]):
        for si_values in si_values_list:
            si_values.sort()
        return si_values_list

    def __init_ds(self, pidata: PIDCInterface, about: int):
        #
        si_values_list = self._si_values_list(pidata, about)
        si_values_list = self._sort_si_values(si_values_list)
        self._build_ds(si_values_list)
   
    def minsum_list(self, src_var: int) -> NDFloatArray:
        src_idx = self.subset_map[src_var]
        rstates = np.arange(self.dim)
        rs_begin = np.multiply(rstates, self.nvars)
        src_ranks = self.lmr_rank[np.add(rs_begin, src_idx)]
        src_locs = np.add(rs_begin, src_ranks)
        lmr_lows = self.lmr_pfxsum[np.subtract(src_locs, 1)]
        lmr_highs = np.multiply(
            self.lmr_sorted[src_locs],
            np.subtract(self.nvars - 1, src_ranks)
        )
        lmr_states = lmr_lows + lmr_highs
        return lmr_states
 
    def lmr_minsum_stvec(self, src_var:int):
        src_idx = self.subset_map[src_var]
        rstates = np.arange(self.dim)
        rs_begin = np.multiply(rstates, self.nvars)
        src_ranks = self.lmr_rank[np.add(rs_begin, src_idx)]
        src_locs = np.add(rs_begin, src_ranks)
        lmr_lows = self.lmr_pfxsum[np.subtract(src_locs, 1)]
        lmr_highs = np.multiply(
            self.lmr_sorted[src_locs],
            np.subtract(self.nvars - 1, src_ranks)
        )
        lmr_states = lmr_lows + lmr_highs
        #return src_ranks, lmr_lows, lmr_highs, lmr_states
        return np.sum(lmr_states)

    def lmr_minsum_stiter(self, src_var:int):
        src_idx = self.subset_map[src_var]
        rdsum = np.float32(0).astype(self.lmr_sorted.dtype)
        lmvalues = np.zeros(self.dim, self.lmr_sorted.dtype)
        lmsums = np.zeros(self.dim, self.lmr_sorted.dtype)
        for rs in range(self.dim):
            rsbegin = rs * self.nvars
            lmrank = self.lmr_rank[rsbegin + src_idx]
            lmlow = self.lmr_pfxsum[rsbegin + lmrank - 1]
            lmhigh = (self.nvars - 1 - lmrank) * self.lmr_sorted[rsbegin + lmrank]
            lmvalues[rs] = lmhigh
            lmsums[rs] = lmlow
            rdsum += lmlow + lmhigh
        # print(lmvalues, lmsums)
        return rdsum

    def lmr_minsum_rxiter(self, src_var:int) -> FloatT:
        lmd = self.pidata.get_lmr(self.about, src_var)
        rdsum = np.float32(0).astype(self.lmr_sorted.dtype)
        for rs in range(self.dim):
            lmv = lmd[rs] 
            rsbegin = rs * self.nvars
            rsend = rsbegin +  self.nvars
            # Find rank of lmv in 
            lmrank = np.searchsorted(
                self.lmr_sorted[range(rsbegin, rsend)],
                lmv
            )
            lmlow = self.lmr_pfxsum[rsbegin + lmrank - 1] if lmrank > 0 else 0.0
            lmhigh = (self.nvars - lmrank) * lmv
            rdsum += lmlow + lmhigh
        return rdsum

    def lmr_minsum_rxvec(self, src_var:int) -> FloatT:
        lm_vec = self.pidata.get_lmr(self.about, src_var)
        #
        rstates = np.arange(self.dim)
        rs_begin = np.multiply(rstates, self.nvars)
        rs_end = np.add(rs_begin, self.nvars)
        #
        rnk_gen: Generator[int] = (
            np.searchsorted(self.lmr_sorted[range(rsb, rse)], lmv)
            for (rsb, rse, lmv) in zip(rs_begin, rs_end, lm_vec)
        )
        src_ranks = np.fromiter(rnk_gen, dtype=np.int32)
        src_locs = np.add(rs_begin, src_ranks)
        lmr_lows = np.zeros(src_locs.size, dtype=self.lmr_sorted.dtype)
        ll_locs = src_locs > 0
        lmr_lows[ll_locs] = self.lmr_pfxsum[np.subtract(src_locs[ll_locs], 1)]
        lmr_highs = np.multiply(lm_vec, np.subtract(self.nvars, src_ranks))
        lmr_states = lmr_lows + lmr_highs
        rdsum = np.sum(lmr_states).astype(self.lmr_sorted.dtype)
        return rdsum

    def lmr_minsum_iter(self, src_var:int):
        if src_var in self.subset_map:
            return self.lmr_minsum_stiter(src_var)
        return self.lmr_minsum_rxiter(src_var)

    def lmr_minsum_vec(self, src_var:int):
        if src_var in self.subset_map:
            return self.lmr_minsum_stvec(src_var)
        return self.lmr_minsum_rxvec(src_var)

    def lmr_minsum(self, src_var:int):
        return self.lmr_minsum_vec(src_var)
