import numpy as np
import numpy.typing as npt
import scipy
import typing as t

NPFloat : t.TypeAlias = np.floating[t.Any]
NPInteger : t.TypeAlias = np.integer[t.Any]
NDFloatArray : t.TypeAlias = npt.NDArray[np.floating[t.Any]]
NDIntArray : t.TypeAlias = npt.NDArray[np.integer[t.Any]]
FloatT: t.TypeAlias = float | NPFloat 
IntegerT: t.TypeAlias = int | NPInteger 
NPDType: t.TypeAlias = npt.DTypeLike

DataArray : t.TypeAlias = (
    NDFloatArray |
    scipy.sparse.csr_matrix |
    scipy.sparse.csc_matrix
)

LogBase = t.Literal['e', '2', '10', '1p']
DiscretizerMethod = t.Literal["bayesian_blocks", "uniform"]
EstimatorMethod = t.Literal[
    "miller_madow",
    "maximum_likelihood",
    "shrinkage",
    "dirichlet"
]

T = t.TypeVar("T")
U = t.TypeVar("U")

class DataPair(t.Generic[T], t.NamedTuple):
    first: T
    second: T

class DataPair2(t.Generic[T, U], t.NamedTuple):
    first: T
    second: U
