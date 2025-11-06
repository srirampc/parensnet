#
import h5py
import numpy as np
#
from codetiming import Timer

#
from ..types import NDFloatArray, NDIntArray, DataPair, IDDTuple, NPDType
from ..comm_interface import CommInterface, default_comm
from .util import InputArgs, WorkDistributor, iddtuple_to_h5


class PUCUnionWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface
    sub_net_files: list[str]
    puc_file: str
    union_mat: NDFloatArray
    dtype: NPDType
    int_dtype: NPDType
    nindices: NDIntArray

    @Timer(name="PUCUnionWorkflow::__init__", logger=None)
    def __init__(
        self,
        mxargs: InputArgs,
        wdistr: WorkDistributor,
        sub_net_files: list[str] | None = None,
        puc_file: str | None = None,
    ) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()
        if sub_net_files is not None:
            self.sub_net_files = sub_net_files
        elif mxargs.sub_net_files is not None:
            self.sub_net_files = mxargs.sub_net_files 
        else:
            self.sub_net_files = []
        if puc_file is not None:
            self.puc_file = puc_file
        else:
            self.puc_file = mxargs.puc_file
        #
        first_file = self.sub_net_files[0]
        with h5py.File(first_file) as hfptr:
            rindex: NDIntArray = hfptr["/data/index"] # pyright: ignore[reportAssignmentType]
            rpuc: NDFloatArray = hfptr["/data/puc"] # pyright: ignore[reportAssignmentType]
            self.nindices = rindex[:]
            self.dtype = rpuc.dtype
            self.int_dtype = rindex.dtype
        #
        self.union_mat = np.zeros((mxargs.nvars, mxargs.nvars),
                                  self.dtype)
 
    @Timer(name="PUCUnionWorkflow::run_at_root", logger=None)
    def run_at_root(self):
        # if self.sub_net_files is None:
        #     return
        #
        for puc_file in self.sub_net_files:
            with h5py.File(puc_file) as hfptr:
                rindex: NDIntArray = hfptr["/data/index"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
                rpuc: NDFloatArray = hfptr["/data/puc"][:] # pyright: ignore[reportIndexIssue, reportAssignmentType]
                rvalue = np.maximum(
                    self.union_mat[rindex[:, 0], rindex[:, 1]], rpuc
                )
                self.union_mat[rindex[:, 0], rindex[:, 1]] = rvalue
                self.union_mat[rindex[:, 1], rindex[:, 0]] = rvalue
        nfiles = len(self.sub_net_files)
        self.union_mat = self.union_mat/np.float64(nfiles).astype(self.dtype)
        union_puc = self.union_mat[self.nindices[:, 0], self.nindices[:, 1]]   
        iddtuple_to_h5(
            IDDTuple(self.nindices, union_puc),
            self.mxargs.puc_file,
            DataPair[str]("index", "puc"),
            "data"
        )

    @Timer(name="PUCUnionWorkflow::run", logger=None)
    def run(self):
        if self.comm.rank == 0:
            self.run_at_root()
        self.comm.barrier()
