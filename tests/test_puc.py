import typing as t, json,  itertools
import numpy as np, numpy.testing as nptest, anndata as an
import pytest
from parensnet.types import LogBase, DiscretizerMethod, NDFloatArray, NPDType
from parensnet.mvim.rv import MRVInterface, MRVNodePairs
from parensnet.mvim.misi import MISIData
import logging

def _log():
    return logging.getLogger(__name__)

def compare_nodes(
    sub_puc_data: dict[str, t.Any],
    ppidcif: MRVInterface,
) -> tuple[list[np.float32], list[bool]]:
    nprob_errs = []
    nprob_eq = []
    nnodes = len(sub_puc_data['nodes'])
    for j  in range(ppidcif.nvariables):
        jnx = sub_puc_data['nodes'][j]
        phist = ppidcif.get_hist(j)
        nprob = phist / ppidcif.ndata
        assert len(nprob) == jnx['number_of_bins']
        nprob_errs.append(np.max(nprob - jnx['probabilities']))
        nprob_eq.append(
            (len(nprob) == jnx['number_of_bins']) and
            np.all(np.isclose(nprob, jnx['probabilities']))
        )
    _log().debug("Node Err : [%s]", ",".join(map(str, nprob_errs)))
    _log().debug("Node Eq. : [%s]", ",".join(map(str, nprob_eq)))
    return nprob_errs, nprob_eq

def compare_node_pairs(
    sub_puc_data: dict[str, t.Any],
    ppidcif: MRVInterface,
) -> tuple[list[np.float32], list[bool]]:
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
    return npair_errs, npair_eq

def compare_with_pidc(
    sub_puc_data: dict[str, t.Any],
    ppidcif: MRVInterface,
    network: NDFloatArray,
):
    nprob_errs, nprob_eq = compare_nodes(sub_puc_data, ppidcif)
    npair_errs, npair_eq = compare_node_pairs(sub_puc_data, ppidcif)
    pcnet = np.array(sub_puc_data['puc_scores'])
    net_eq = np.isclose(network, pcnet)
    print("Network Equals :: ", str(net_eq))
    return bool(np.all(np.array(nprob_eq)) and
                np.all(np.array(npair_eq)) and
                np.all(net_eq))

class PUCTestData:
    def __init__(
        self,
        nroundup: int=4,
        nvars: int=4,
        nobs_list: tuple[int,...] = (1000, 10000),
        puc_json: str="data/puc_test_data_1Kx4_10Kx4.json",
        h5ad_file: str="data/pbmc_puc_test.h5ad",
        out_dir: str="/localscratch/schockalingam6/tmp/"
    ):
        self.disc_method: DiscretizerMethod ="bayesian_blocks"
        self.tbase: LogBase='2'
        self.dtype: NPDType = np.float32
        self.int_dtype: NPDType = np.int32
        self.nvars: int = nvars
        self.puc_data: dict[int, t.Any] = {}
        json_data = {}
        with open(puc_json) as fx:
            json_data = json.load(fx)
        jkeys = list(json_data.keys())
        for kx in jkeys:
            self.puc_data[int(kx)] =  json_data[kx]
        self.out_dir:str = out_dir

        adata = an.read_h5ad(h5ad_file, 'r')
        self.afdata: dict[int, NDFloatArray] = {}
        self.nobs_list: list[int] = list(nobs_list)
        for nobs in self.nobs_list:
            self.afdata[nobs] = np.round(adata.X[:nobs, :self.nvars], # pyright: ignore[reportCallIssue, reportArgumentType]
                                         nroundup)
        adata.file.close()


class TestClass:

    @pytest.fixture
    def ptdata(self) -> PUCTestData:
        return PUCTestData()

    def test_a_puc(self, ptdata: PUCTestData, subtests: pytest.Subtests):
        logging.basicConfig(level=logging.DEBUG)
        for nobs in ptdata.nobs_list:
            sub_data = ptdata.afdata[nobs][:nobs, :ptdata.nvars]
            nobs, nvars = sub_data.shape
            with subtests.test(i=nobs):
                ppairsl = MRVNodePairs.from_data(
                    sub_data,
                    (nobs, nvars),
                    ptdata.disc_method,
                    ptdata.tbase,
                    ptdata.dtype,
                    ptdata.int_dtype
                )
                sub_ref_data = ptdata.puc_data[nobs]
                _nprob_errs, nprob_eq = compare_nodes(sub_ref_data, ppairsl)
                assert nprob_eq == [True for _ in range(nvars)]
                #nptest.assert_array_equal(nprob_eq, [True for _ in range(nvars)])
                cmp_eq = [True for _ in itertools.combinations(range(nvars), 2)]
                _npair_errs, npair_eq = compare_node_pairs(sub_ref_data, ppairsl)
                assert cmp_eq == npair_eq

    def test_b_puc_network(self, ptdata: PUCTestData, subtests: pytest.Subtests):
        for nobs in ptdata.nobs_list:
            sub_data = ptdata.afdata[nobs][:nobs, :ptdata.nvars]
            nobs, nvars = sub_data.shape
            with subtests.test(i=nobs):
                ppairsl = MRVNodePairs.from_data(
                    sub_data,
                    (nobs, nvars),
                    ptdata.disc_method,
                    ptdata.tbase,
                    ptdata.dtype,
                    ptdata.int_dtype
                )
                #
                ref_net = np.array(ptdata.puc_data[nobs]['puc_scores'])
                full_net = ppairsl.compute_puc_matrix(ptdata.dtype)
                nptest.assert_array_almost_equal(full_net, ref_net)
                rlist = list(range(nvars))
                for_net = ppairsl.compute_puc_matrix_for(rlist, ptdata.dtype)
                nptest.assert_array_almost_equal(for_net, ref_net)

    def test_c_misi(self, ptdata: PUCTestData, subtests: pytest.Subtests):
        for nobs in ptdata.nobs_list:
            sub_data = ptdata.afdata[nobs][:nobs, :ptdata.nvars]
            nobs, nvars = sub_data.shape
            with subtests.test(i=nobs):
                ppairsl = MRVNodePairs.from_data(
                    sub_data,
                    (nobs, nvars),
                    ptdata.disc_method,
                    ptdata.tbase,
                    ptdata.dtype,
                    ptdata.int_dtype
                )
                misidd = MISIData.from_pair_list_data(ppairsl)
                sub_ref_data = ptdata.puc_data[nobs]
                _nprob_errs, nprob_eq = compare_nodes(sub_ref_data, misidd)
                assert [True for _ in range(nvars)] == nprob_eq
                cmp_eq = [True for _ in itertools.combinations(range(nvars), 2)]
                _npair_errs, npair_eq = compare_node_pairs(sub_ref_data, misidd)
                assert cmp_eq == npair_eq

    def test_d_misi_network(self, ptdata: PUCTestData, subtests: pytest.Subtests):
        for nobs in ptdata.nobs_list:
            sub_data = ptdata.afdata[nobs][:nobs, :ptdata.nvars]
            nobs, nvars = sub_data.shape
            with subtests.test(i=nobs):
                ppairslt = MRVNodePairs.from_data(
                    sub_data,
                    (nobs, nvars),
                    ptdata.disc_method,
                    ptdata.tbase,
                    ptdata.dtype,
                    ptdata.int_dtype
                )
                misidd = MISIData.from_pair_list_data(ppairslt)
                #
                ref_net = np.array(ptdata.puc_data[nobs]['puc_scores'])
                full_net = misidd.compute_puc_matrix(ptdata.dtype)
                nptest.assert_array_almost_equal(full_net, ref_net)
                rlist = list(range(nvars))
                for_net = misidd.compute_puc_matrix_for(rlist, ptdata.dtype)
                nptest.assert_array_almost_equal(for_net, ref_net)
