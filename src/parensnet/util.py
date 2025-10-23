import datetime
import logging
import operator
import os
import typing as t
#
import codetiming
import numpy as np
import pandas as pd
import psutil
import yaml
import h5py
import numpy.typing as npt

ProfilesT : t.TypeAlias = list[dict[str, t.Any]]
NDIntArray : t.TypeAlias = npt.NDArray[np.integer[t.Any]]
NDFloatArray : t.TypeAlias = npt.NDArray[np.floating[t.Any]]
NDArray : t.TypeAlias = npt.NDArray[np.integer[t.Any] | np.floating[t.Any]]

def block_low(rank: int, nproc: int, n: int):
    return (rank * n) // nproc

def block_high(rank: int, nproc: int, n: int):
    return (((rank + 1) * n) // nproc) - 1

def block_size(rank: int, nproc: int, n: int):
    return block_low(rank + 1, nproc, n) - block_low(rank, nproc, n)

def block_owner(j: int, nproc: int, n: int):
    return (((nproc) * ((j) + 1) - 1) // (n))

def block_range(rank:int, nproc:int, ndata:int):
    return range(
        block_low(rank, nproc, ndata),
        block_high(rank, nproc, ndata) + 1
    )

def block_2d_range(i:int, j:int, nproc:int, ndata:int):
    return (block_range(i, nproc, ndata), block_range(j, nproc, ndata))

def block_2d_all_ranges(nproc:int, ndata:int):
    return [
        [block_2d_range(i, j, nproc, ndata) for j in range(nproc)]
        for i in range(nproc)
    ]

def block_2d_triu_ranges(nproc:int, ndata:int):
    return [
        [block_2d_range(i, j, nproc, ndata) for j in range(i, nproc)]
        for i in range(nproc)
    ]

def square_diagonal(n:int, offset:int):
    return [(x, x+offset) for x in range(0, n - offset)]

def lr_square_diagonals(n:int, offset:int):
    return (square_diagonal(n, offset), square_diagonal(n, n-offset))

def halve_range(inr: range):
    range_mid = (inr.stop - inr.start) // 2  
    return range(inr.start, range_mid), range(range_mid, inr.stop) 

def half_split_ranges(in_ranges: list[range]):
    hlist=[halve_range(inr) for inr in in_ranges]
    return [ha for ha, _ in hlist],  [hb for _, hb in hlist]

def diag_distribution(nproc:int):
    return [
        # ldiag + rdiag if ldiag != rdiag else ldiag + [() for _ in rdiag]
        ldiag + rdiag
        for ldiag, rdiag in (
            lr_square_diagonals(nproc, offset)
            for offset in range(1 + (nproc//2))
        )
    ] 

def triu_pair_to_index(n: int, i: int, j: int) -> int:
    return i * n + j - (((i + 2)*(i + 1))//2)

def triu_index_to_pair(n: int, k: int) -> tuple[int, int]:
    i = n - 2 - int(np.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - (n*(n-1)//2) + ((n-i)*((n-i)-1)//2)
    # i = self.n - 2 - int(np.sqrt(-8*k + self.ioffset_const)/2.0 - 0.5)
    # j = k + self.joffset[i]
    return i, j


def parse_yaml(yaml_file: str):
    with open(yaml_file) as ymfx:
        return yaml.safe_load(ymfx)


def timestamp_prefix(rank: int):
    return f"{str(datetime.datetime.now())} :: {rank}"

def memory_profile(rank: int = 0) -> dict[str, t.Any]:
    pid = os.getpid()
    pproc = psutil.Process(pid)
    pfmem = pproc.memory_info()
    return {
        "proc": rank,
        "rss": pfmem.rss / (2**30),
        "vms": pfmem.vms / (2**30),
        "pct": pproc.memory_percent(),
    }

def timing_profile(rank: int = 0):
     rtimers = codetiming.Timer.timers
     return [{
         "proc": rank,
         "name": name,
         "ncalls": rtimers.count(name),
         "total_time": ttime,
         "min_time": rtimers.min(name),
         "max_time": rtimers.max(name),
         "mean_time": rtimers.mean(name),
         "median_time": rtimers.median(name),
         "stdev_time": rtimers.stdev(name),
     } for name, ttime in sorted(
         rtimers.items(),
         key=operator.itemgetter(1),
         reverse=True
     )]

def log_mem_usage(
    logger: logging.Logger,
    level: int,
    rank: int = 0,
):
    if not logger.isEnabledFor(level):
        return
    svmem = psutil.virtual_memory()
    used = svmem.used/(2 ** 30)
    total = svmem.total/(2 ** 30)
    avail = svmem.available/(2 ** 30)
    free = svmem.free/(2 ** 30)
    pct = svmem.percent
    logger.log(
        level,
        (
            f"{timestamp_prefix(rank)} :: "
            f"Used/Total : {used}/{total} ({pct} %); "
            f"Free : {free} ; Avail {avail}. "
            f":: CPU : {psutil.cpu_times_percent().user} %." 
         )
    )

def log_with_timestamp(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
    rank: int = 0,
    **kwargs: t.Any
):
    if not logger.isEnabledFor(level):
        return
    tprefix = f"{timestamp_prefix(rank)} :: "
    logger.log(level, tprefix + message, *args, **kwargs)

def log_data_frame(
    logger: logging.Logger,
    level: int,
    df: pd.DataFrame, 
    rank: int = 0,
):
    if not logger.isEnabledFor(level):
        return
    prefix = timestamp_prefix(rank)
    # set display options to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # print the dataframe
    for lx in df.to_string().splitlines():
        logger.log(level, prefix + lx)

def flatten_npalist(
    npa_list: list[NDFloatArray],
    ftype: npt.DTypeLike,
    dim_type: npt.DTypeLike = np.int32,
    st_type: npt.DTypeLike = np.int64,
):
    list_dim = np.array([npa.size for npa in npa_list], dtype=dim_type)
    list_start = np.zeros(list_dim.size, dtype=st_type)
    for dx in range(1, list_dim.size):
        list_start[dx] = list_start[dx - 1] + list_dim[dx - 1]
    flat_data = np.zeros(np.sum(list_dim), dtype=ftype)
    for dx, npa in enumerate(npa_list):
        hsize = list_dim[dx]
        hstart = list_start[dx]
        hend = hstart + hsize
        flat_data[hstart:hend] = npa.reshape(hsize)
    return list_dim, list_start, flat_data

def create_h5ds(
    hgroup: h5py.Group,
    dset_name: str,
    nparr: NDArray,
):
    return hgroup.create_dataset(
            dset_name,
            shape=nparr.shape,
            dtype=nparr.dtype,
            data=nparr
        )


