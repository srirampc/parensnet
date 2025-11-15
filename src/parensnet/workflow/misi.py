import logging
import typing as t
import numpy as np

#
from devtools import pformat
from codetiming import Timer

from ..types import NDFloatArray, NDBoolArray 
from ..comm_interface import CommInterface, default_comm
from ..util import triu_pair_to_index
from ..mvim.rv import RVNode, RVNodePair, RVPairData, MRVNodePairs
from ..mvim.misi import MISIData, MISIDataH5
from ..mvim.distributed import MISIDataDistributed
from .util import InputArgs, WorkDistributor

def _log():
    return logging.getLogger(__name__)


class MISIWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface
    cell_select: NDBoolArray | None
    misi_data_file: str
    nobs: int

    def __init__(
        self,
        mxargs: InputArgs,
        wdistr: WorkDistributor,
        cselect: NDBoolArray | None = None,
        data_file: str | None = None
    ) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()
        self.cell_select = cselect
        if data_file: 
            self.misi_data_file = data_file
        else:
            self.misi_data_file = self.mxargs.misi_data_file
        if cselect is None:
            self.nobs = self.mxargs.nobs
        else:
            self.nobs = int(np.count_nonzero(cselect))

    def select_node_data(self, node_data: NDFloatArray, ix: int):
        if self.cell_select is None:
            return node_data[:, ix]
        else:
            return node_data[self.cell_select, ix]

    @Timer(name="MISIWorkflow::construct_nodes_range", logger=None)
    def construct_nodes_range(self, range_itr:range) -> list[RVNode]:
        node_data = self.mxargs.load_range_data(range_itr)
        return [
            RVNode(
                self.select_node_data(node_data, ix),
                self.nobs,
                self.mxargs.disc_method,
                np.float32,
                np.int32
            )
            for ix, _ in enumerate(range_itr)
        ]

    @Timer(name="MISIWorkflow::construct_nodes", logger=None)
    def construct_nodes(self, pid:int) -> list[RVNode]:
        nodes = self.construct_nodes_range(self.wdistr.var_range(pid))
        return nodes

    @Timer(name="MISIWorkflow::construct_all_nodes", logger=None)
    def construct_all_nodes(self) -> list[RVNode]:
        return self.construct_nodes_range(range(self.mxargs.nvars))

    @Timer(name="MISIWorkflow::collect_nodes", logger=None)
    def collect_nodes(self, nodes:list[RVNode]) -> list[RVNode]:
        rnodes = self.comm.collect_merge_lists(nodes)
        return t.cast(list[RVNode], rnodes)

    @Timer(name="MISIWorkflow::save_nodes_to_h5", logger=None)
    def save_nodes_to_h5(self, rnodes:list[RVNode]):
        if self.comm.rank == 0:
            import pickle
            ppld = MRVNodePairs.from_pairs(rnodes, [])
            misd = MISIData.from_pair_list_data(ppld, save_hist=False)
            with open(self.mxargs.nodes_pickle, 'wb') as wfx:
                #
                pickle.dump(ppld, wfx)
            #misd.to_h5(self.misi_data_file)
            del misd
            del ppld
        self.comm.barrier()

    @Timer(name="MISIWorkflow::collect_node_pairs", logger=None)
    def collect_node_pairs(self, node_pairs:list[RVPairData]) -> list[RVPairData]:
        return self.comm.collect_merge_lists_at_root(node_pairs, True)

    @Timer(name="MISIWorkflow::build_misi", logger=None)
    def build_misi(self,
        nodes:list[RVNode],
        node_pairs:list[RVPairData]
    ):
        return MISIData.from_nodes_and_pairs(
            nodes,
            node_pairs,
            (self.nobs, self.mxargs.nvars),
            self.mxargs.disc_method,
            self.mxargs.tbase,
            save_hist=False,
        )

    @Timer(name="MISIWorkflow::gave_misi", logger=None)
    def save_misi(self, misi_data: MISIData):
        misi_data.to_h5(self.misi_data_file)

    @Timer(name="MISIWorkflow::construct_pairs_for_range", logger=None)
    def construct_pairs_for_range(
        self,
        nodes: list[RVNode],
        px_range: tuple[range, range],
        px_data: tuple[NDFloatArray, NDFloatArray],
    ) -> list[RVPairData]:
        row_range, col_range = px_range
        row_data, col_data = px_data
        return [
            RVPairData(
                triu_pair_to_index(self.mxargs.nvars, rx, cx),
                rx,
                cx,
                RVNodePair.from_nodes(
                    (nodes[rx], nodes[cx]),
                    (
                        self.select_node_data(row_data, i),
                        self.select_node_data(col_data, j),
                        #row_data[:, i], col_data[:, j]
                    ),
                    self.mxargs.tbase,
                    np.float32,
                    np.int32,
                    save_hist=False,
                ),
            )
            for j, cx in enumerate(col_range) for i, rx in enumerate(row_range)
            if rx < cx
        ]

    @Timer(name="MISIWorkflow::construct_node_pairs", logger=None)
    def construct_node_pairs(
        self,
        nodes: list[RVNode]
    ) -> list[RVPairData]:
        self.comm.barrier()
        gpairs: list[list[RVPairData] | None] = [None] * self.wdistr.pair_nbatches
        #
        for bidx in range(self.wdistr.pair_nbatches):
            self.comm.log_at_root(_log(), logging.DEBUG, f"Running Batch B{bidx} ")
            row_range = self.wdistr.pair_row_range(bidx, self.comm.rank)
            col_range = self.wdistr.pair_col_range(bidx, self.comm.rank)
            row_data, col_data = self.wdistr.load_pair_block_data(bidx, self.comm.rank)
            # self.comm.log_comm(
            #     _log(), logging.DEBUG,
            #     f"B{bidx} :: R:{row_range} ; C:{col_range}"
            # )
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

    @Timer(name="MISIWorkflow::save_misi_distributed", logger=None)
    def save_misi_distributed(
        self,
        nodes: list[RVNode],
        node_pairs: list[RVPairData]
    ):
        distr_misi =  MISIDataDistributed(
            nodes,
            node_pairs,
            (self.nobs, self.mxargs.nvars),
            self.mxargs.disc_method,
            self.mxargs.tbase
        )
        distr_misi.to_h5(self.misi_data_file)

    @Timer(name="MISIWorkflow::save_misi_at_root", logger=None)
    def save_misi_at_root(
        self,
        nodes: list[RVNode],
        node_pairs: list[RVPairData]
    ):
        import pickle
        rnpairs = self.collect_node_pairs(node_pairs)
        self.comm.log_at_root(_log(), logging.DEBUG, f"rnpairs ::  {len(rnpairs)}" )
        del node_pairs
        rnpairs.sort(key=lambda x: x.pindex)
        with open(self.mxargs.nodes_pickle, 'wb') as wfx:
                pickle.dump(nodes, wfx)
        with open(self.mxargs.nodes_pairs_pickle, 'wb') as wfx:
                pickle.dump(rnpairs, wfx)
        if self.comm.rank == 0:
            mdata = self.build_misi(nodes, rnpairs)
            self.save_misi(mdata)

    @Timer(name="MISIWorkflow::save_misih5", logger=None)
    def save_misih5(self, node_pairs:list[RVPairData]):
        for ix in range(self.comm.size):
            self.comm.barrier()
            if ix == self.comm.rank:
                for pindex, i, j, npair in node_pairs:
                    with MISIDataH5(self.misi_data_file, open_mode='a') as misih5:
                        misih5.set_pair_data(pindex, i, j, npair)
            self.comm.barrier()

    def log_list_sizes(self, lprefix: str, lst: list[t.Any]):
        tlst = self.comm.collect_counts(len(lst))
        if self.comm.rank == 0:
            tarr = np.array(tlst)
            self.comm.log_at_root(
                _log(), logging.DEBUG, "%s    :: %s ",
                lprefix, pformat({"distr": tarr, "count": sum(tlst)})
            )

    @Timer(name="MISIWorkflow::run", logger=None)
    def run(self):
        self.comm.log_at_root(_log(), logging.DEBUG,
                              f"Data : {self.nobs} x {self.mxargs.nvars}")
        #
        nodes = []
        nodes = self.construct_nodes(self.comm.rank)
        self.comm.barrier()
        self.log_list_sizes("Nodes", nodes) 
        nodes = self.collect_nodes(nodes)
        # if self.mxargs.save_nodes and not self.mxargs.save_node_pairs:
        #    self.save_nodes_to_h5(nodes)
        # return
        # nodes = t.cast(list[PIDCNode], nodes)
        #
        # with open(mxargs.nodes_pickle, 'rb') as fx:
        #      nodes =  pickle.load(fx)
        #
        self.comm.log_at_root(_log(), logging.DEBUG, "Total Nodes :: %d", len(nodes)) 
        #
        node_pairs = self.construct_node_pairs(nodes)
        self.comm.barrier()
        self.log_list_sizes("Node Pairs", node_pairs) 
        node_pairs.sort(key=lambda x: x.pindex)
        #
        if self.mxargs.save_node_pairs:
            self.save_misi_at_root(nodes, node_pairs)
            # self.save_misi_distributed(nodes, node_pairs)


