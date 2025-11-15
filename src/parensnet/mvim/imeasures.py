import numpy as np
import typing as t

from collections.abc import Callable
from ..types import NPFloat, NDFloatArray, FloatT, LogBase, EstimatorMethod 
from .bayesian_blocks import block_bins

ShrinkageFn: t.TypeAlias = Callable[[NDFloatArray, NDFloatArray, int], NDFloatArray]

def log(
    p: NDFloatArray | FloatT ,
    tbase: LogBase
):
    match tbase:
        case 'e':
            return np.log(p)
        case '2':
            return np.log2(p)
        case '10':
            return np.log10(p)
        case '1p':
            return np.log1p(p)
    return np.log(p) # pyright: ignore[reportUnreachable]


def log_jvi_ratio(
    xytab: NDFloatArray,
    xtab: NDFloatArray,
    ytab: NDFloatArray,
    lbase: LogBase,
    tweight: FloatT,
):
    jvi_ratio = np.divide(
        xytab,
        (xtab.reshape((xtab.size, 1)) * ytab)
    )
    jvi_ratio = log(jvi_ratio * tweight, lbase)
    jvi_ratio[np.isnan(jvi_ratio) | np.isinf(jvi_ratio)] = 0
    return jvi_ratio


class DiscretePDFBuilder:
    #
    # Parameters:
    #     - frequencies, integer array
    @staticmethod
    def maximum_likelihood(frequencies: NDFloatArray):
        return frequencies / np.sum(frequencies)


    #
    @staticmethod
    def dirichlet(frequencies: NDFloatArray, prior: FloatT):
        prior_frequencies = np.ones(
            frequencies.shape,
            dtype=frequencies.dtype
        ) *  prior
        return (frequencies + prior_frequencies) / (
            sum(frequencies) + sum(prior_frequencies)
        )

    # 
    # TODO: allow pre-calculated target to be passed in instead of uniform distribution
    # Parameters:
    #     - frequencies, integer array
    #     - lambda, void
    @staticmethod
    def shrinkage(
        _frequencies: NDFloatArray,
        _shrinkage_fn: ShrinkageFn | None = None
    ):
        return None
    #TODO:: port to python
    #     target = get_uniform_distribution(frequencies)
    #     n = sum(frequencies)
    #     normalized_frequencies = frequencies / n
    #     calculated_lambda = get_lambda(normalized_frequencies, target, n)
    #     return apply_shrinkage_formula(normalized_frequencies, target, calculated_lambda)
    # end


    @staticmethod
    def probabilities(
        estimator: EstimatorMethod,
        frequencies: NDFloatArray,
        shrinkage_fn: ShrinkageFn | None = None,
        prior: FloatT = 1,
    ):
        match estimator:
            case "maximum_likelihood" | "miller_madow":
                return DiscretePDFBuilder.maximum_likelihood(frequencies)
            case "shrinkage":
                return DiscretePDFBuilder.shrinkage(frequencies, shrinkage_fn)
            case "dirichlet":
                return DiscretePDFBuilder.dirichlet(frequencies, prior)
        return None  # pyright: ignore[reportUnreachable]


class HistogramBuilder:
    #
    #
    @staticmethod
    def joint_histogram(
        xdata: NDFloatArray,
        ydata: NDFloatArray,
        xbins: NDFloatArray,
        ybins: NDFloatArray
    ):
        return (
            np.histogram2d(xdata, ydata, bins=(xbins, ybins))[0],  
            np.histogram(xdata, bins=xbins)[0],
            np.histogram(ydata, bins=ybins)[0],
        )

class Entropy:
    # Parameters:
    #     - probabilities, array of floats
    #     - base, number
    @staticmethod
    def from_p(
        p: NDFloatArray,
        base: t.Literal['e', '2', '10', '1p']
    ):
        pblog = log(p, base)
        pblog = pblog[np.isfinite(pblog)]
        nfp = p[np.isfinite(pblog)]
        return -np.sum(nfp * pblog) + 0.0


class MI:
    # Parameters:
    #     - entropy of first variable, number
    #     - entropy of second variable, number
    #    - joint entropy of both variables, number
    @staticmethod
    def from_entropy(entropy_x: FloatT, entropy_y: FloatT, entropy_xy: FloatT):
        return entropy_x + entropy_y - entropy_xy

    @staticmethod
    def from_tab(
        xytab: NDFloatArray,
        xtab: NDFloatArray,
        ytab: NDFloatArray,
        lbase: LogBase,
        tweight: float | NPFloat | None = None
    ):
        # sum(remove_non_finite.(p_xy .* log.(base, p_xy ./ (p_x .* p_y))))
        tweight = np.sum(xtab) if tweight is None else tweight
        ljvi_ratio = log_jvi_ratio(xytab, xtab, ytab, lbase, tweight)
        mi_prod = xytab * ljvi_ratio
        return np.sum(mi_prod)/tweight
    
    
    @staticmethod
    def from_ljvi(
        ljvi_ratio: NDFloatArray,
        xytab: NDFloatArray,
        tweight: float | NPFloat | None,
    ):
        # sum(remove_non_finite.(p_xy .* log.(base, p_xy ./ (p_x .* p_y))))
        tweight = np.sum(xytab) if tweight is None else tweight
        mi_prod = xytab * ljvi_ratio
        return np.sum(mi_prod)/tweight

    # Parameters:
    #     - joint probabilities, array of floats
    #     - probabilities (first variable), array of floats
    #     - probabilities (second variable), array of floats
    #     - base, number
    @staticmethod
    def from_p(
        p_xy: NDFloatArray,
        p_x: NDFloatArray,
        p_y: NDFloatArray,
        lbase: LogBase,
    ):
        # sum(remove_non_finite.(p_xy .* log.(base, p_xy ./ (p_x .* p_y))))
        return MI.from_tab(p_xy, p_x, p_y, lbase, 1.0)

    @staticmethod
    def from_data_with_bayesian_blocks(
        xdata: NDFloatArray,
        ydata: NDFloatArray,
        tbase: LogBase, 
    ):
        xbins = block_bins(xdata, xdata.dtype)
        ybins = block_bins(ydata, ydata.dtype)
        xyhist, xhist, yhist = HistogramBuilder.joint_histogram(
            xdata, ydata,
            xbins, ybins
        )
        return MI.from_tab(xyhist, xhist, yhist, tbase)


class SI:
    @staticmethod
    def from_ljvi(
        ljvi_ratio: NDFloatArray,
        xytab: NDFloatArray,
        xtab: NDFloatArray,
        ytab: NDFloatArray,
    ):
        #
        # vec(sum(
        #     remove_non_finite.(
        #         (p_xz ./ p_z) .* log.(base, p_xz ./ (p_x .* p_z))
        #     ),
        #     dims = dim_sum))
        #
        assert (xtab.size, ytab.size) == xytab.shape
        xsize, ysize = xytab.shape
        #
        x_ratio = xytab / xtab.reshape((xsize, 1))
        x_ratio[np.isnan(x_ratio) | np.isinf(x_ratio)] = 0
        #
        y_ratio = xytab / ytab.reshape((1, ysize))
        y_ratio[np.isnan(y_ratio) | np.isinf(y_ratio)] = 0
        #
        return (
            np.sum(x_ratio * ljvi_ratio, axis=1),
            np.sum(y_ratio * ljvi_ratio, axis=0)
        )

    # Parameters:
    #     - joint probabilities, array of floats
    #     - probabilities (source), array of floats
    #     - probabilities (target), array of floats
    #     - dimension along which to sum, integer
    #     - base, number
    @staticmethod
    def from_histogram(
        xytab: NDFloatArray,
        xtab: NDFloatArray,
        ytab: NDFloatArray,
        tbase: LogBase,
        tweight: FloatT | None = None
    ):
        #
        # vec(
        #  sum(
        #     remove_non_finite.(
        #         (p_xz ./ p_z) .* log.(base, p_xz ./ (p_x .* p_z))
        #     ),
        #     dims = dim_sum)
        # )
        #
        assert xytab.shape == (xtab.size, ytab.size)
        #
        tweight = np.sum(xtab) if tweight is None else tweight
        #
        jvi_ratio = log_jvi_ratio(xytab, xtab, ytab, tbase, tweight)
        return  SI.from_ljvi(jvi_ratio, xytab, xtab, ytab)


class LMR: 
    @staticmethod
    def about_x_from_ljvi(
        ljvi_ratio: NDFloatArray,
        xytab: NDFloatArray,
        tweight: float | NPFloat | None
    ):
        #
        # vec(sum(
        #     remove_non_finite.(
        #         p_xz .* log.(base, p_xz ./ (p_x .* p_z))
        #     ),
        #     dims = dim_sum))
        #
        assert ljvi_ratio.shape == xytab.shape
        tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
        #
        elp_tab = xytab * ljvi_ratio * tfactor
        return np.sum(elp_tab, axis=1)
    
    @staticmethod
    def about_y_from_ljvi(
        ljvi_ratio: NDFloatArray,
        xytab: NDFloatArray,
        tweight: float | NPFloat | None
    ):
        #
        # vec(sum(
        #     remove_non_finite.(
        #         p_xz .* log.(base, p_xz ./ (p_x .* p_z))
        #     ),
        #     dims = dim_sum))
        #
        assert ljvi_ratio.shape == xytab.shape
        tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
        #
        elp_tab = xytab * ljvi_ratio * tfactor
        return np.sum(elp_tab, axis=0)

    @staticmethod
    def from_ljvi(
        ljvi_ratio: NDFloatArray,
        xytab: NDFloatArray,
        tweight: float | NPFloat | None
    ):
        #
        # vec(sum(
        #     remove_non_finite.(
        #         p_xz .* log.(base, p_xz ./ (p_x .* p_z))
        #     ),
        #     dims = dim_sum))
        #
        assert ljvi_ratio.shape == xytab.shape
        tfactor = 1.0/(np.sum(xytab) if tweight is None else tweight)
        #
        elp_tab = xytab * ljvi_ratio * tfactor
        return (
            np.sum(elp_tab, axis=1),
            np.sum(elp_tab, axis=0)
        )
    
    
    @staticmethod
    def from_histogram(
        xytab: NDFloatArray,
        xtab: NDFloatArray,
        ytab: NDFloatArray,
        tbase: LogBase,
        tweight: float | NPFloat | None = None
    ):
        #
        xsize, ysize = xytab.shape
        assert (xsize, ysize) == (xtab.size, ytab.size)
        #
        tweight = np.sum(xtab) if tweight is None else tweight
        ljvi_ratio = log_jvi_ratio(xytab, xtab, ytab, tbase, tweight)
        return  LMR.from_ljvi(ljvi_ratio, xytab, tweight)



# Parameters:
# 	- probabilities (target), array of floats
# 	- specific information of source 1 and target, array of floats
# 	- specific information of source 2 and target, array of floats
# 	- base, number
def redundancy(
    ztab: NDFloatArray,
    x_si: NDFloatArray,
    y_si: NDFloatArray,
    tweight: FloatT | None=None
) -> FloatT:
    # 	minimum_specific_information = min.(specific_information_1, specific_information_2)
    # 	return sum(vec(p_z) .* vec(minimum_specific_information))
    if tweight:
        tfactor = 1.0/tweight
    else:
        tfactor = np.float64(1).astype(ztab.dtype)
    return np.sum(ztab * np.minimum(x_si, y_si) * tfactor)
