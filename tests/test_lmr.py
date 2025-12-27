import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest


from parensnet.types import DataPair, NDIntArray
from parensnet.mvim.rv import LMRSubsetDataStructure, MRVInterface
from parensnet.mvim.misi import MISIDataH5, MISIPair, MISIRangePair


class LMRTestData:
    def __init__(self) -> None:
        self.data_h5file : str = "data/adata.20k.500.misidata.h5"
        # self.h5ptr : h5py.File = h5py.File(self.data_h5file)
        # self.data_group: h5py.Group = t.cast(h5py.Group, self.h5ptr["data"])
        self.pairs_list : list[tuple[int, int]] = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)
        ]
        self.samples_list : list[NDIntArray] = [
            np.array([
                 11,  38, 149, 162, 169, 172, 183, 209, 222, 255,
                258, 307, 324, 354, 361, 370, 374, 412, 455, 491
            ]),
            np.array([
                  0,   5,  45, 104, 106, 140, 153, 154, 178, 227,
                229, 248, 278, 314, 338, 356, 412, 431, 477, 495
            ]),
        ]
        self.sample_puc : dict[tuple[int, int], list[float]] = dict(zip(
            self.pairs_list,
            [
                [  1.7694760560,  2.72823572],
                [ 10.5683107376,  9.40065860],
                [ 22.5292415618, 19.74103355], 
                [ 6.25307178497,  5.80409240], 
                [ 7.03579425811,  6.36623191],
             ]
        ))
        self.pairs_list2 : list[tuple[int, int]] = [
            (0, 1), (0, 5), (1, 5), (0, 45)
        ]
        self.samples_list2 : list[NDIntArray] = [
            np.array([
                  5, 45,  104, 106, 140, 153, 154, 178, 227,
                229, 248, 278, 314, 338, 356, 412, 431, 477, 495
            ]),
            np.array([
                  0,   5,  45, 104, 106, 140, 153, 154, 178, 227,
                229, 248, 278, 314, 338, 356, 412, 431, 477, 495
            ]),
            np.array([
                  1,   5,  45, 104, 106, 140, 153, 154, 178, 227,
                229, 248, 278, 314, 338, 356, 412, 431, 477, 495
            ]),
            np.array([
                  0, 1, 5, 45,  104, 106, 140, 153, 154, 178, 227,
                229, 248, 278, 314, 338, 356, 412, 431, 477, 495
            ]),
        ]
        self.sample_puc2 : dict[tuple[int,int], list[float]] = dict(zip(
            [(0, 1), (0, 5), (1, 5),(0, 45)], 
            [ [ 2.72823,  2.72823,  2.72823,  2.72823 ],
              [ 6.36623,  6.36623,  7.57903,  7.57903 ],
              [ 7.06674,  7.94825,  7.06674,  7.94825 ],
              [12.87396, 12.87396, 14.51176, 14.51176 ] ]
        ))


@pytest.fixture
def ltdata() -> LMRTestData:
    lmr_data = LMRTestData()
    return lmr_data


def test_a_pair_full_lmr(ltdata: LMRTestData):
    misi_pairs : dict[tuple[(int, int)], MISIPair] = {} 
    with MISIDataH5(ltdata.data_h5file) as misdh:
        for i,j in ltdata.pairs_list:
            misi_pairs[(i,j)] = MISIPair.from_misidata(misdh, i, j)
    #
    print("")
    for (i,j), mpair in misi_pairs.items():
        mij = mpair.get_mi(i, j)
        pucij = mpair.accumulate_redundancies(i, j, None)
        lmrij = mpair.compute_lmr_puc(i, j)
        print("MPair : " , i, j, mij, pucij, lmrij)
        assert_almost_equal(pucij, lmrij, decimal=3)

def test_b_range_pair_full_lmr(ltdata: LMRTestData):
    mrpair = MISIRangePair(
        ltdata.data_h5file, 
        DataPair[range](range(0,6), range(0, 6)),
        None, True
    );
    #
    print("")
    for i,j in ltdata.pairs_list:
        mij = mrpair.get_mi(i, j)
        pucij = mrpair.accumulate_redundancies(i, j, None)
        lmrij = mrpair.compute_lmr_puc(i, j)
        print("MPair : " , mij, pucij, lmrij)
        assert_almost_equal(pucij, lmrij, decimal=3)

def create_lmrss_ds(i: int, j: int, splist: list[int] | NDIntArray, pidata: MRVInterface):
    subset_var = [int(x) for x in splist]
    subset_map = dict(zip(subset_var, range(len(subset_var))))
    lmrssi = LMRSubsetDataStructure(pidata, i, subset_var, subset_map) 
    lmrssj = LMRSubsetDataStructure(pidata, j, subset_var, subset_map) 
    return lmrssi, lmrssj
    
def test_c_pair_subset_lmr_minsum(ltdata: LMRTestData, subtests: pytest.Subtests):
    lpair = ltdata.pairs_list[0]
    i, j = ltdata.pairs_list[0]
    splist = ltdata.samples_list[0]
    with MISIDataH5(ltdata.data_h5file) as misdh:
        mpair = MISIPair.from_misidata(misdh, i, j)
    print("")
    mij = mpair.get_mi(i, j)
    rpfr_vec = 0.0; sfr_vec = 1.0
    rprv_vec = 0; srv_vec = 1.0
    with subtests.test(msg="LMR minsum compare vec & iter for nosrc"):
        #
        lmrssi, lmrssj = create_lmrss_ds(i, j, splist, mpair)
        sfr_vec = lmrssi.minsum_nosrc_vec(j)
        sfr_itr = lmrssi.minsum_nosrc_iter(j)
        srv_vec = lmrssj.minsum_nosrc_vec(i)
        srv_itr = lmrssj.minsum_nosrc_iter(i)
        print(f"PAIR :: {lpair}")
        print(f"LMR SUM : {sfr_vec}, {sfr_itr}, {srv_vec}, {srv_itr} ")
        assert_almost_equal(sfr_vec, sfr_itr, decimal=3)
        assert_almost_equal(srv_vec, srv_itr, decimal=3)
    #assert_almost_equal(srv_vec, srv_itr, decimal=3)
    #
    with subtests.test(msg="LMR minsum compare vec & iter for wsrc"):
        rplist = np.concatenate([np.array([i, j]), splist])
        lmrssi, lmrssj = create_lmrss_ds(i, j, rplist, mpair)
        rpfr_vec = lmrssi.minsum_wsrc_vec(j)
        rpfr_itr = lmrssi.minsum_wsrc_iter(j)
        rprv_vec = lmrssj.minsum_wsrc_vec(i)
        rprv_itr = lmrssj.minsum_wsrc_iter(i)
        print(f"PAIR :: {lpair}")
        print(f"LMR SUM : {rpfr_vec}, {rpfr_itr}, {rprv_vec}, {rprv_itr}")
        assert_almost_equal(rpfr_vec, rpfr_itr, decimal=3)
        assert_almost_equal(rprv_vec, rprv_itr, decimal=3)
    with subtests.test(msg="LMR minsum compare wsrc vec and nosrc vec"):
        assert_almost_equal(rpfr_vec, sfr_vec, decimal=3)
        assert_almost_equal(rprv_vec, srv_vec, decimal=3)

    itr_list = []
    vec_list = []
    for idx, splist in enumerate(ltdata.samples_list2):
        with subtests.test(msg=f"Test Samples list 2 array {idx}"):
            lmrssi, lmrssj = create_lmrss_ds(i, j, splist, mpair)
            fritr = lmrssi.minsum_iter(j)
            rvitr = lmrssj.minsum_iter(i)
            print(f"LMR SUM : {idx} {splist.shape} {mij:.3}, {fritr:.4}, {fritr/mij:.5}")
            print(f"LMR SUM : {idx} {splist.shape} {mij:.3}, {rvitr:.4}, {rvitr/mij:.5}, {(fritr+rvitr)/mij:.5} {(2*lmrssi.nvars - ((fritr+rvitr)/mij)):.5}")
            frvec = lmrssi.minsum_vec(j)
            rvvec = lmrssj.minsum_vec(i)
            itr_list.append(fritr + rvitr)
            vec_list.append(frvec + rvvec)
            print(f"LMR SUM : {idx} {splist.shape} {mij:.3}, {frvec:.4}, {frvec/mij:.5}")
            print(f"LMR SUM : {idx} {splist.shape} {mij:.3}, {rvvec:.4}, {rvvec/mij:.5}, {(frvec+rvvec)/mij:.5} {(2*lmrssi.nvars - ((frvec+rvvec)/mij)):.5}")
            assert_almost_equal(fritr, frvec, decimal=3)
            assert_almost_equal(rvitr, rvvec, decimal=3)
    with subtests.test(msg=f"Test all puc from Samples list 2 are equal"):
        assert_almost_equal(
            vec_list, np.repeat(10.958, len(ltdata.samples_list2)), decimal=3
        )

def num_as_str(tupfx: tuple[int | float, ...]):
    return  str(list(map(
        lambda fx : f"{fx:.3}" if isinstance(fx, float) else str(fx),
        tupfx
    )))

def test_d_pair_subset_lmr(ltdata: LMRTestData, subtests: pytest.Subtests):
    misi_pairs : dict[tuple[(int, int)], MISIPair] = {} 
    with MISIDataH5(ltdata.data_h5file) as misdh:
        for i,j in ltdata.pairs_list:
            misi_pairs[(i,j)] = MISIPair.from_misidata(misdh, i, j)
    #
    print("")
    print("L1")
    for k, splist in enumerate(ltdata.samples_list):
        for (i,j), mpair in misi_pairs.items():
            mpair.init_subset_var(splist)
            mij = mpair.get_mi(i, j)
            rupdate = np.array([
                0.0 if (sx == i) or (sx == j)
                else mpair.redundancy_update_for(i, j, sx) for sx in splist
            ])
            redcies = [mpair.get_redundancies(i,j, sx) for sx in splist]
            redcies = [
                (sx, float(ra), float(rb), float((mij - float(ra))/mij),
                 float((mij - float(rb))/mij), float(rux))
                for sx, (ra,rb, _rc), rux in zip(splist, redcies, rupdate)
            ]
            with subtests.test(msg=f"LMR PUC Test for pair ({i},{j}) and samples_list {k}"):
                pucij = mpair.accumulate_redundancies(i, j, splist)
                lmrij = mpair.compute_lmr_puc(i, j)
                msij = (mpair.get_lmr_minsum(i, j), mpair.get_lmr_minsum(j, i))
                #print(f"MPair : ({i}, {j}), {mij}")
                #print(f"{"\n".join(map(num_str, redcies))}") 
                print(f"R L L2 : ({i}, {j}), {mij} {pucij}, {lmrij}, {msij}")
                #print(f"MSI J:  {msij}" )
                assert_almost_equal(pucij, lmrij, decimal=3)
                assert_almost_equal(pucij, ltdata.sample_puc[(i,j)][k], decimal=3)
                #break
        #break

def test_e_pair_subset_lmr2(ltdata: LMRTestData, subtests: pytest.Subtests):
    print("L2")
    misi_pairs = {} 
    with MISIDataH5(ltdata.data_h5file) as misdh:
        for i,j in ltdata.pairs_list2:
            misi_pairs[(i,j)] = MISIPair.from_misidata(misdh, i, j)
    #
    for k, splist in enumerate(ltdata.samples_list2):
        for (i,j), mpair in misi_pairs.items():
            mpair.init_subset_var(splist)
            with subtests.test(msg=f"LMR PUC Test 2 for pair ({i},{j}) and samples_list2 {k}"):
                pucij = mpair.accumulate_redundancies(i, j, splist)
                lmrij = mpair.compute_lmr_puc(i, j)
                msij = (mpair.get_lmr_minsum(i, j), mpair.get_lmr_minsum(j, i))
                mij = mpair.get_mi(i, j)
                print(f"R L L2 : ({i}, {j}), {mij} {pucij}, {lmrij}, {msij}")
                assert_almost_equal(pucij, lmrij, decimal=3)
                assert_almost_equal(pucij, ltdata.sample_puc2[(i,j)][k], decimal=3)

def test_f_range_pair_subset_lmr(ltdata: LMRTestData, subtests: pytest.Subtests):
    misi_rpair = MISIRangePair(ltdata.data_h5file,
                               DataPair[range](range(0,6), range(0, 6)),
                               None)
    #
    print("")
    print("L1")
    for k, splist in enumerate(ltdata.samples_list):
        for (i,j) in ltdata.pairs_list:
            misi_rpair.init_subset_var(splist, True)
            mij = misi_rpair.get_mi(i, j)
            #rupdate = np.array([
            #    0.0 if (sx == i) or (sx == j)
            #    else misi_rpair.redundancy_update_for(i, j, sx) for sx in splist
            #])
            #redcies = [misi_rpair.get_redundancies(i,j, sx) for sx in splist]
            #redcies = [
            #    (sx, float(ra), float(rb), float((mij - float(ra))/mij),
            #     float((mij - float(rb))/mij), float(rux))
            #    for sx, (ra,rb, _rc), rux in zip(splist, redcies, rupdate)
            #]
            #print(f"{"\n".join(map(num_str, redcies))}") 
            with subtests.test(msg=f"LMR PUC Test for pair ({i},{j}) and samples_list {k}"):
                pucij = misi_rpair.accumulate_redundancies(i, j, splist)
                lmrij = misi_rpair.compute_lmr_puc(i, j)
                msij = (misi_rpair.get_lmr_minsum(i, j), misi_rpair.get_lmr_minsum(j, i))
                #print(f"MPair : ({i}, {j}), {mij}")
                print(f"R L L2 : ({i}, {j}), {mij} {pucij}, {lmrij}, {msij}")
                #print(f"MSI J:  {msij}" )
                assert_almost_equal(pucij, lmrij, decimal=3)
                assert_almost_equal(pucij, ltdata.sample_puc[(i,j)][k], decimal=3)
                #break
        #break

def test_g_range_pair_subset_lmr2(ltdata: LMRTestData, subtests: pytest.Subtests):
    print("L2")
    misi_rpair = MISIRangePair(ltdata.data_h5file,
                               DataPair[range](range(0,6), range(0,46)),
                               None)
    #
    for k, splist in enumerate(ltdata.samples_list2):
        for (i,j) in ltdata.pairs_list2:
            misi_rpair.init_subset_var(splist, True)
            with subtests.test(msg=f"LMR PUC Test 2 for pair ({i},{j}) and samples_list2 {k}"):
                pucij = misi_rpair.accumulate_redundancies(i, j, splist)
                lmrij = misi_rpair.compute_lmr_puc(i, j)
                msij = (misi_rpair.get_lmr_minsum(i, j), misi_rpair.get_lmr_minsum(j, i))
                mij = misi_rpair.get_mi(i, j)
                print(f"R L L2 : ({i}, {j}), {mij} {pucij}, {lmrij}, {msij}")
                assert_almost_equal(pucij, lmrij, decimal=3)
                assert_almost_equal(pucij, ltdata.sample_puc2[(i,j)][k], decimal=3)
