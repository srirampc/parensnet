import typing as t
import logging
import _parensnet_ext as pext
import numpy as np
import itertools

from parensnet.types import NDIntArray, NDFloatArray
from parensnet.comm_interface import default_comm

def _log():
    return logging.getLogger(__name__)

@t.final
class PEData:
    def __init__(self,nvars:int=5, nobs: int=50):
        self.nvars: int = nvars
        self.nobs: int = nobs
        self.index: NDIntArray = np.array(list(
            itertools.combinations(range(nvars), 2)
        ))
        self.npairs:int = self.index.shape[0]
        self.data: NDFloatArray = np.random.random(self.npairs)
    

    def as_dict(self) -> dict[str, t.Any]:
        return {
            "nobs": self.nobs,
            "nvars": self.nvars,
            "npairs": self.npairs,
            "data":  self.data,
            "index": self.index,
        }

def test_save_puc():
    comm_ifx = default_comm()
    pedata = PEData(nvars=9)
    tdata = pedata.as_dict()
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"Before Split::  {pedata.data.shape}, {pedata.index.shape}"
    )
    lc_range = comm_ifx.block_range(pedata.npairs)
    tdata["data"] = pedata.data[lc_range]
    tdata["index"] = pedata.index[lc_range, :]

    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"After Split::  {tdata["data"].shape}, {tdata["index"].shape}"
    )
    pext.save_puc(tdata, "tests/test_out.h5")

def main():
    logging.basicConfig(level=logging.DEBUG)
    test_save_puc()

if __name__ == "__main__":
    main()

