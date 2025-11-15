import argparse
import logging
#
import anndata as an
import numpy as np
#
from devtools import pformat
from codetiming import Timer

#
from ..types import NDIntArray, NDBoolArray, NDObjectArray
from ..comm_interface import CommInterface, default_comm
from ..workflow.util import InputArgs, WorkDistributor, getenv
from ..workflow.misi import MISIWorkflow
from ..workflow.puc import FullPUCWorkflow, SPUCWorkflow
from ..workflow.context import ContextWorkflow
from ..workflow.union import PUCUnionWorkflow

def _log():
    return logging.getLogger(__name__)

class ClusterUnionWorkflow:
    mxargs: InputArgs
    wdistr: WorkDistributor
    comm: CommInterface
    cell_labels: NDObjectArray
    clust_labels: NDObjectArray
    clust_counts: NDIntArray
    # puc_mat: NDFloatArray
    # dtype: NPDType
    # int_dtype: NPDType

    @Timer(name="ClusterUnionWorkflow::__init__", logger=None)
    def __init__(self, mxargs: InputArgs, wdistr: WorkDistributor) -> None:
        self.mxargs = mxargs
        self.wdistr = wdistr
        self.comm = default_comm()
        adata = an.read_h5ad(self.mxargs.h5ad_file, backed='r')
        self.cell_labels = adata.obs['leiden']  # pyright: ignore[reportAttributeAccessIssue]
        luniq = np.unique_counts(self.cell_labels)
        cselected: NDBoolArray = luniq.counts > 100
        self.clust_counts = luniq.counts[cselected]
        self.clust_labels = luniq.values[cselected]

    @Timer(name="ClusterUnionWorkflow::run", logger=None)
    def run(self):
        self.clust_labels = self.clust_labels[10:]
        mfiles = [f"{self.mxargs.wflow_dir}/misi_C{str(clabel)}.h5" for clabel in self.clust_labels]
        pfiles = [f"{self.mxargs.wflow_dir}/puc_C{str(clabel)}.h5" for clabel in self.clust_labels]
        #pdfiles = [f"{self.mxargs.temp_dir}/pidc_C{str(clabel)}.h5" for clabel in self.clust_labels]
        #
        for clabel, misi_file, puc_file in zip(
            self.clust_labels, mfiles, pfiles 
        ):
            self.comm.log_at_root(
                _log(),
                logging.DEBUG,
                "Start  Run for Cluster : %s", str(clabel)
            )
            cxselect: NDBoolArray = self.cell_labels == clabel
            misiwf = MISIWorkflow(self.mxargs, self.wdistr, cxselect, misi_file)
            misiwf.run()
            self.comm.barrier()
            # spucwf = SPUCWorkflow(self.mxargs, self.wdistr, misi_file, puc_file)
            # spucwf.run_with_ranges(True)
            # self.comm.barrier()
            
        # punion = PUCUnionWorkflow(self.mxargs, self.wdistr, pfiles)
        # punion.run()
        # self.comm.barrier()
        # ctxwf = ContextWorkflow(self.mxargs, self.wdistr)
        # ctxwf.run()
        # self.comm.barrier()

 

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
    #return
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
            case 'puc_union':
                PUCUnionWorkflow(mxargs, wdistr).run()
            case 'puc2pidc':
                ContextWorkflow(mxargs, wdistr).run()
            case 'cluster_union':
                ClusterUnionWorkflow(mxargs, wdistr).run()
            case _:
                if comm_ifx.rank == 0:
                    print(f"Mode {rxmode} not implemented" )
        comm_ifx.barrier()
    #
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
   if default_comm().rank == 0:
       print("Run Arguments :: ", run_args)
   with InputArgs.from_yaml(run_args.yaml_file) as mxargs:
       main(mxargs)
