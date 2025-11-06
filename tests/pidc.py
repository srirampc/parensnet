import typing as t, json,  itertools
import numpy as np, anndata as an
from parensnet.types import LogBase, DiscretizerMethod, NDFloatArray, NPDType
from parensnet.pidc import PIDCPairListData, PIDCInterface
from parensnet.misi import MISIData
from parensnet.util import NDIntArray

def compare_with_pidc(
    sub_puc_data: dict[str, t.Any],
    ppidcif: PIDCInterface,
    network: NDFloatArray,
):
    nprob_errs = []
    nprob_eq = []
    nnodes = len(sub_puc_data['nodes'])
    for j  in range(nnodes):
        jnx = sub_puc_data['nodes'][j]
        phist = ppidcif.get_hist(j)
        nprob = phist / ppidcif.ndata
        assert len(nprob) == jnx['number_of_bins']
        nprob_errs.append(np.max(nprob - jnx['probabilities']))
        nprob_eq.append(
            (len(nprob) == jnx['number_of_bins']) and
            np.all(np.isclose(nprob, jnx['probabilities']))
        )
    print("Node Errors :", ",".join(map(str, nprob_errs)))
    print("Node Eq :", ",".join(map(str, nprob_eq)))

    npair_errs = []
    npair_eq = []
    jnpairs = sub_puc_data['node_pairs']
    nvars = ppidcif.nvariables
    for keyx  in itertools.combinations(range(nvars), 2):
        s,t = keyx
        jnp1 = jnpairs[s][t]
        jnp2 = jnpairs[t][s]
        npair_errs.append((
            (jnp1['mi'] - ppidcif.get_mi(s, t)),
            np.max(jnp1['si'] - ppidcif.get_si(about=s, by=t)),
            np.max(jnp2['si'] - ppidcif.get_si(about=t, by=s)),
        ))
        npair_eq.append((
            np.isclose(jnp1['mi'],  ppidcif.get_mi(s, t)) and
            np.all(np.isclose(jnp1['si'], ppidcif.get_si(about=s, by=t))) and
            np.all(np.isclose(jnp2['si'], ppidcif.get_si(about=t, by=s)))
        ))
    print("Node Pair Errors :", ",".join(map(str, npair_errs)))
    print("Node Pair Eq :", ",".join(map(str, npair_eq)))
    pcnet = np.array(sub_puc_data['puc_scores'])
    print("Network Equals :: ", str(np.isclose(network, pcnet)))
    return bool(np.all(np.array(nprob_eq)) and
                np.all(np.array(npair_eq)) and 
                np.all(np.isclose(network, pcnet)))

class LMRNode:
    def __init__(self, rvndata: PIDCInterface, about:int):
        #
        self.about: int = about
        self.nvars: int = rvndata.nvariables
        self.about_dim: int = int(rvndata.get_hist_dim(about))
        lmr_size: int = self.about_dim * self.nvars
        self.lmr_sorted: NDFloatArray = np.zeros(lmr_size, dtype=rvndata.float_dtype())
        self.lmr_pfxsum: NDFloatArray = np.zeros(lmr_size, dtype=rvndata.float_dtype())
        self.lmr_ranks: NDIntArray = np.zeros(lmr_size, dtype=rvndata.int_dtype())
        # 
        for rs in range(self.about_dim):
            si_values = [
                (rvndata.lmr_value(about=about, by=by, rstate=rs), by) 
                if by != about else (0.0, by)
                for by in range(self.nvars)
            ]
            si_values = sorted(si_values)
            rsbegin = rs * self.nvars
            print(si_values, rs, self.about_dim, rsbegin)
            curr_sum = 0.0
            for ix, (svx, by) in enumerate(si_values):
                curr_sum += svx
                self.lmr_ranks[rsbegin + by] = ix
                self.lmr_sorted[rsbegin + ix] = svx
                self.lmr_pfxsum[rsbegin + ix] = curr_sum

    def redundancy_sums(self, by:int):
        rdsum = np.float32(0)
        lmvalues = np.zeros(self.about_dim, np.float32)
        lmsums = np.zeros(self.about_dim, np.float32)
        for rs in range(self.about_dim):
            rsbegin = rs * self.nvars
            lmrank = self.lmr_ranks[rsbegin + by]
            lmlow = self.lmr_pfxsum[rsbegin + lmrank - 1]
            lmhigh = (self.nvars - 1 - lmrank) * self.lmr_sorted[rsbegin + lmrank]
            lmvalues[rs] = lmhigh
            lmsums[rs] = lmlow
            rdsum += lmlow + lmhigh
        print(lmvalues, lmsums)
        return rdsum

class LMRNodeData:
    def __init__(self, rvndata: MISIData | PIDCPairListData):
        self.rvndata: MISIData | PIDCPairListData= rvndata
        self.nodes: list[LMRNode] = [
            LMRNode(rvndata, vx) for vx in range(rvndata.nvariables)
        ]

    def puc_for(self, i:int, j:int):
        mij = self.rvndata.get_mi(i, j)
        mi_factor = mij * (self.rvndata.nvariables - 2)
        return (
            ( (mi_factor - self.nodes[i].redundancy_sums(j)) / mij ) +
            ( (mi_factor - self.nodes[j].redundancy_sums(i)) / mij )
        )
    

class PUCTest:
    def __init__(
        self,
        nroundup: int=4,
        nvars: int=4,
        nobs_list: list[int] = [100, 1000, 10000],
        puc_json: str="./tmp/puc_data4.json",
        h5ad_file: str="./data/pbmc800K/pbmc800K.20K.h5ad",
        out_dir: str="/localscratch/schockalingam6/tmp/"
    ):
        self.disc_method: DiscretizerMethod ="bayesian_blocks"
        self.tbase: LogBase='2'
        self.dtype: NPDType = np.float32
        self.int_dtype: NPDType = np.int32
        self.nvars: int = nvars
        self.puc_data: dict[str, t.Any] = {}
        with open(puc_json) as fx:
            self.puc_data = json.load(fx)
        self.out_dir:str = out_dir
        
        adata = an.read_h5ad(h5ad_file) ; 
        self.ardata : NDFloatArray = np.round(adata.X, nroundup)  # pyright: ignore[reportCallIssue, reportArgumentType]
        self.afdata: dict[int, NDFloatArray] = {}
        self.nobs_list: list[int] = list(nobs_list)
        for nobs in self.nobs_list:
            self.afdata[nobs] = self.ardata[:nobs, :]
        #
        # self.afhund : NDFloatArray = self.ardata[:100, :]
        # self.afthou : NDFloatArray = self.ardata[:1000, :]
        # self.aftenk : NDFloatArray = self.ardata[:10000, :]
        #
        self.ppairsld: dict[int, PIDCPairListData] = {}
        self.misdd: dict[int, MISIData] = {}
        self.reddx: dict[int, tuple[t.Any, t.Any]] = {}
        self.pucmd: dict[int, tuple[NDFloatArray, NDFloatArray, NDFloatArray, NDFloatArray]] = {}
        self.cmpdx: dict[int, tuple[bool, bool, bool, bool]] = {}


    def compare(self, nobs: int):
        stnobs = str(nobs)
        if stnobs in self.puc_data:
            pudx = self.puc_data[stnobs]
            ppairsl = self.ppairsld[nobs]
            pmisdd = self.misdd[nobs]
            netx, net_for, mnetx, mnet_for = self.pucmd[nobs]
            self.cmpdx[nobs] = (
                compare_with_pidc(pudx, ppairsl, netx),
                compare_with_pidc(pudx, ppairsl, net_for),
                compare_with_pidc(pudx, pmisdd, mnetx),
                compare_with_pidc(pudx, pmisdd, mnet_for),
            )

    def build_network(self, nobs: int):
        ppairsl = self.ppairsld[nobs]
        pmisdd = self.misdd[nobs]
        self.reddx[nobs] = (
            ppairsl.compute_redundancies(),
            pmisdd.compute_redundancies(),
        )
        rlist = list(range(self.nvars))
        self.pucmd[nobs] = (
            ppairsl.compute_puc_matrix(self.dtype),
            ppairsl.compute_puc_matrix_for(rlist, self.dtype),
            pmisdd.compute_puc_matrix(self.dtype),
            pmisdd.compute_puc_matrix_for(rlist, self.dtype),
        )

    def init_ds(self, nobs:int):
        sub_data = self.afdata[nobs][:nobs, :self.nvars]
        nobs, nvars = sub_data.shape
        ppairsl = PIDCPairListData.from_data(
            sub_data,
            (nobs, nvars),
            self.disc_method,
            self.tbase,
            self.dtype,
            self.int_dtype
        )
        self.misdd[nobs] = MISIData.from_pair_list_data(ppairsl) 
        self.ppairsld[nobs] = ppairsl

        # nplx_hund = generate_node_pairs(self.afhund, None, self.nvars)
        # nplx_thou = generate_node_pairs(self.afthou, None, self.nvars)
        # nplx_tenk = generate_node_pairs(self.aftenk, None, self.nvars)
        # misd_hund = ppidc.MISIData.from_pair_list_data(nplx_hund)
        # misd_thou = ppidc.MISIData.from_pair_list_data(nplx_thou)
        # misd_tenk = ppidc.MISIData.from_pair_list_data(nplx_tenk)


    def run(self):
        for nobs in self.nobs_list:
            self.init_ds(nobs)

        for nobs in self.nobs_list:
            self.build_network(nobs)

        for nobs in self.nobs_list:
            self.compare(nobs)

        for nobs in self.nobs_list:
            out_file = f"{self.out_dir}/misd_{nobs}.h5"
            self.misdd[nobs].to_h5(out_file)

        # red_thou = nplx_thou.compute_redundancies()
        # net_thou = nplx_thou.compute_puc_matrix(afthou.dtype)
        # net_sub_thou = nplx_thou.compute_puc_matrix_for(list(range(4)), afthou.dtype)
        # compare_with_pidc(puc_data['1000'], nplx_thou, net_thou)
        # compare_with_pidc(puc_data['1000'], nplx_thou, net_sub_thou)
        # 
        # red_tenk = nplx_tenk.compute_redundancies()
        # net_tenk = nplx_tenk.compute_puc_matrix(aftenk.dtype)
        # net_sub_tenk = nplx_tenk.compute_puc_matrix_for(list(range(4)), aftenk.dtype)
        # 
        # misd_red_thou = misd_thou.compute_redundancies()
        # misd_net_thou = misd_thou.compute_puc_matrix(afthou.dtype)
        # misd_net_sub_thou = misd_thou.compute_puc_matrix_for(list(range(4)), afthou.dtype)
        # compare_with_pidc(puc_data['1000'], misd_thou, misd_net_thou)
        # compare_with_pidc(puc_data['1000'], misd_thou, misd_net_sub_thou)
    
        # misd_red_tenk = misd_tenk.compute_redundancies()
        # misd_net_tenk = misd_tenk.compute_puc_matrix(aftenk.dtype)
        # misd_net_sub_tenk = misd_tenk.compute_puc_matrix_for(list(range(4)), aftenk.dtype)
        # compare_with_pidc(puc_data['10000'], misd_tenk, misd_net_tenk)
        # compare_with_pidc(puc_data['10000'], misd_tenk, misd_net_sub_tenk)
    
        # misd_thou.to_h5("/localscratch/schockalingam6/tmp/misd_thou.h5")
        # txmisd_thou = ppidc.MISIData.from_h5("/localscratch/schockalingam6/tmp/misd_thou.h5")

        # txmisd_net_thou = txmisd_thou.compute_puc_matrix(afthou.dtype)
        # txmisd_net_sub_thou = txmisd_thou.compute_puc_matrix_for(list(range(4)), afthou.dtype)
        # compare_with_pidc(puc_data['1000'], txmisd_thou, txmisd_net_thou)
        # compare_with_pidc(puc_data['1000'], txmisd_thou, txmisd_net_sub_thou)
    
        # misd_tenk.to_h5("/localscratch/schockalingam6/tmp/misd_tenk.h5")
        # txmisd_tenk = ppidc.MISIData.from_h5("/localscratch/schockalingam6/tmp/misd_tenk.h5")
        # txmisd_net_tenk = txmisd_tenk.compute_puc_matrix(aftenk.dtype)
        # txmisd_net_sub_tenk = txmisd_tenk.compute_puc_matrix_for(list(range(4)), aftenk.dtype)
        # compare_with_pidc(puc_data['10000'], txmisd_tenk, txmisd_net_tenk)
        # compare_with_pidc(puc_data['10000'], txmisd_tenk, txmisd_net_sub_tenk)


def lmr_data(npx: PIDCPairListData | MISIData, about:int, target: int):
    by_nodes: list[int] = list(
        x for x in range(npx.nvariables) if x != about and x != target
    )
    lmr_abtgt = npx.get_lmr(about=about, by=target)
    bynodes_lmr = [npx.get_lmr(about=about, by=by) for by in by_nodes]
    min_list = [np.minimum(by_lmr, lmr_abtgt) for by_lmr in bynodes_lmr]
    lm_sum = np.sum(np.array([np.sum(rx) for rx in min_list]))
    return lmr_abtgt, bynodes_lmr, min_list, lm_sum

def lmr_puc(npx: PIDCPairListData | MISIData, i: int, j: int):
    rdx = lmr_data(npx, about=i, target=j)
    ldx = lmr_data(npx, about=j, target=i)
    mij = npx.get_mi(i,j)
    return sum(
        x for x in [(mij - np.sum(minx))/mij for minx in rdx[-2]+ldx[-2]]
        if x > 0
    )
