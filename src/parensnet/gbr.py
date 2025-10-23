import itertools
import json
import typing as t
#
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tqdm
import sklearn.ensemble as skm
import arboreto.algo
import arboreto.core
#
from timeit import default_timer as timer
from pydantic import BaseModel
#
from .types import NDFloatArray, DataArray
from .data import ExpDataProcessor


EARLY_STOP_WINDOW_LENGTH = 25

class NoneArgs(BaseModel):
    regressor: t.Literal['none'] = 'none' 


class XGBArgs(BaseModel):
    regressor: t.Literal['xgb'] = 'xgb' 
    learning_rate : float = 0.01
    n_estimators  : int = 500


class STXGBArgs(BaseModel):
    regressor: t.Literal['stxgb'] = 'stxgb' 
    learning_rate: float = 0.01
    n_estimators: int = 500  # can be arbitrarily large
    # max_features: float = 0.1
    colsample_bytree: float = 0.2 
    colsample_bynode: float = 0.5 
    subsample: float  = 0.9


class SGBMArgs(BaseModel): 
    regressor: t.Literal['sgbr'] = 'sgbr' 
    learning_rate : float = 0.01
    n_estimators  : float = 500
    max_features  : float = 0.1
    early_stop: bool = False


class STSGBMArgs(BaseModel): 
    regressor: t.Literal['stsgbr'] = 'stsgbr' 
    learning_rate : float = 0.01
    n_estimators  : float = 5000
    max_features  : float = 0.1
    subsample: float  = 0.9
    early_stop: bool = True


class LGBArgs(BaseModel):
    regressor: t.Literal['lgb'] = 'lgb' 
    n_estimators: int = 500
    learning_rate: float = 0.01
    n_jobs: int = 0
    verbosity: int = 0 
    force_col_wise: bool = False
    force_row_wise: bool = False
    objective: str = 'regression'
    importance_type: str = 'gain'


class STLGBArgs(BaseModel):
    regressor: t.Literal['stlgb'] = 'stlgb' 
    learning_rate: float = 0.01
    n_estimators: int = 500
    n_jobs: int = 0 
    verbosity: int = 0 
    subsample: float = 0.9
    force_col_wise: bool = False
    force_row_wise: bool = False
    colsample_bytree: float = 0.2
    colsample_bynode: float = 0.5 
    objective: str = 'regression'
    importance_type: str = 'gain'


class ARBArgs(BaseModel):
    regressor: t.Literal['arb:default', 'arb:gbm'] = 'arb:default' 


GBRArgs : t.TypeAlias = (
    XGBArgs |
    STXGBArgs |
    SGBMArgs |
    STSGBMArgs |
    LGBArgs |
    STLGBArgs |
    ARBArgs |
    NoneArgs
)


Regressor : t.TypeAlias = (
    skm.GradientBoostingRegressor |
    xgb.XGBModel                  |
    lgb.LGBMRegressor
)


GBMethod = t.Literal[
    'xgb', 'stxgb',
    'sgbr', 'stsgbr',
    'lgb', 'stlgb',
    'arb:default', 'arb:gbm',
    'none',
]


class EarlyStopMonitor:

    def __init__(self, window_length: int=EARLY_STOP_WINDOW_LENGTH):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length : int = window_length
        self.boost_rounds : int = 0

    def window_boundaries(self, current_round: int):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(
        self,
        current_round: int,
        regressor: skm.GradientBoostingRegressor,
        _
    ):
        """
        Implementation of the GradientBoostingRegressor monitor function API.

        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """
        self.boost_rounds = current_round
        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False

def gbrunner_args(method: GBMethod) -> GBRArgs:
    match method:
        case 'sgbr':
            return SGBMArgs()
        case 'stsgbr':
            return STSGBMArgs()
        case 'stxgb':
            return STXGBArgs()
        case 'lgb':
            return LGBArgs()
        case 'stlgb':
            return STLGBArgs()
        case 'xgb':
            return XGBArgs()
        case 'arb:default' | 'arb:gbm':
            return ARBArgs(regressor=method)
        case 'none':
            return NoneArgs()


class GMStats(BaseModel):
    run_time: float = 0.0
    start_fit: float = 0.0
    end_fit: float = 0.0
    fit_time: float = 0.0
    n_rounds: int = 0
    n_features: float = 0.0

class RunGMStats(BaseModel):
    total_run_time: float = 0.0
    gmodel_data: list[GMStats | None] = []

    @classmethod
    def init(cls, ngenes: int):
        return cls(gmodel_data=[None for _ in range(ngenes)])


def sklearn_gboost_regressor(
    data_mat: DataArray,
    tgt_values: DataArray | None,
    gb_args: SGBMArgs | STSGBMArgs,
) -> tuple[skm.GradientBoostingRegressor, GMStats] | None:
    if tgt_values is None:
        return None
    early_stop = EarlyStopMonitor() if gb_args.early_stop else None
    run_args = gb_args.model_dump(exclude=set(["regressor", "early_stop"]))
    gstats = GMStats(start_fit=timer())
    skl = skm.GradientBoostingRegressor(**run_args).fit(
        data_mat, tgt_values, monitor=early_stop
    )
    gstats.end_fit = timer()
    gstats.fit_time = gstats.end_fit - gstats.start_fit
    gstats.n_rounds = early_stop.boost_rounds if early_stop else 0
    return skl, gstats 


def xgboost_regressor(
    data_mat: DataArray,
    tgt_values: DataArray | None,
    gb_args: XGBArgs | STXGBArgs,
    device: str | None = None 
) -> tuple[xgb.XGBModel, GMStats] | None:
    if tgt_values is None:
        return None
    run_args = gb_args.model_dump(exclude=set(["regressor"]))
    if device:
        run_args["device"] = "cuda"
    gstats = GMStats(start_fit=timer())
    xsr = xgb.XGBRegressor(**run_args).fit(
        data_mat,
        tgt_values
    )
    gstats.end_fit = timer()
    gstats.fit_time = gstats.end_fit - gstats.start_fit
    return xsr, gstats


def lightgbm_regressor(
    data_mat: DataArray,
    tgt_values: DataArray | None,
    gb_args: LGBArgs | STLGBArgs,
) -> tuple[lgb.LGBMRegressor, GMStats] | None:
    if tgt_values is None:
        return None
    run_args = gb_args.model_dump(exclude=set(["regressor"]))
    gstats = GMStats(start_fit=timer())
    lsr = lgb.LGBMRegressor(**run_args).fit(data_mat, tgt_values)
    gstats.end_fit = timer()
    gstats.fit_time = gstats.end_fit - gstats.start_fit
    return lsr, gstats


def run_gradient_boosting(
    data_mat: DataArray,
    tgt_values: DataArray | None,
    gb_args: GBRArgs, 
) -> tuple[Regressor, GMStats] | None:
    match gb_args.regressor:
        case 'xgb':
            return xgboost_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case 'stxgb':
            return xgboost_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case 'sgbr':
            return sklearn_gboost_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case 'stsgbr':
            return sklearn_gboost_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case 'lgb':
            return lightgbm_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case 'stlgb':
            return lightgbm_regressor(
                data_mat,
                tgt_values,
                gb_args
            )
        case _:
            return None


def fit_grnboost_model(
    expr_data: ExpDataProcessor,
    target_gene: str,
    gb_args: GBRArgs, 
) -> tuple[Regressor, GMStats] | None:
    exp_mat, tg_exp = expr_data.target_gene_matrix(target_gene)
    return run_gradient_boosting(exp_mat, tg_exp, gb_args)


def grnboost_model_weights_at(
    tg_index: int,
    ntfs: int,
    gmodel: Regressor,
    gstat: GMStats,
) -> tuple[NDFloatArray, GMStats]:
    #
    gfeat: NDFloatArray = t.cast(NDFloatArray, gmodel.feature_importances_)
    gstat.n_features = int(np.sum(gfeat != 0.0))
    #
    if tg_index < 0:
        return gfeat, gstat
    #
    mfeat = np.zeros(ntfs, gfeat.dtype)
    if tg_index > 0:
        mfeat[:tg_index] = gfeat[:tg_index]
    if tg_index+1 < ntfs:
        mfeat[tg_index+1:] = gfeat[tg_index:]
    return mfeat, gstat


def grnboost_model_weights(
    tf_map: dict[str, int],
    target_gene: str,
    gmodel: Regressor,
    gstat: GMStats,
):
    return grnboost_model_weights_at(
        tf_map[target_gene] if target_gene in tf_map else -1,
        len(tf_map),
        gmodel,
        gstat,
    )
 

@t.final
class GBRunner:
    def __init__(
        self,
        expr_data: ExpDataProcessor,
        device: t.Literal["cpu", "gpu"]
    ) -> None:
        self.sd_ = expr_data
        # Output data
        self.rstats_ = RunGMStats.init(self.sd_.ngenes)
        self.importance_ : NDFloatArray | None = None
        self.device_ = "cuda" if device == "gpu" else None
        # self.run_time_ = 0.0
        self.run_desc_ = "GB Runner"

    def update(self, tgene: str, gmodel: Regressor, gstat: GMStats):
        mfeat, rgstat = grnboost_model_weights(self.sd_.tf_map,
                                               tgene, gmodel, gstat)
        gidx = self.sd_.gene_map[tgene]
        self.rstats_.gmodel_data[gidx] = rgstat
        self.importance_[gidx, :] = np.reshape(
            mfeat,
            shape=(1, self.sd_.ntfs)
        )

    def gene_model(self, tgene: str, gb_args: GBRArgs):
        start_time = timer() 
        fmdx  = fit_grnboost_model(self.sd_, tgene, gb_args)
        if fmdx:
            mdx, gstat = fmdx
            gstat.run_time = timer() - start_time
            self.update(tgene, mdx, gstat)

    def init_importance(self, take_n: int | None=None):
        self.importance_  = np.zeros(
            shape=(take_n if take_n else self.sd_.ngenes, self.sd_.ntfs),
            dtype=np.float32
        )

    def genes_itr(self, take_n: int | None, use_tqdm: bool):
        giter = itertools.islice(self.sd_.gene_map.keys(),
                                 take_n if take_n else None)
        if use_tqdm:
            ttotal = take_n if take_n else len(self.sd_.gene_map)
            miniters = ttotal * 0.25
            return tqdm.tqdm(
                giter,
                desc=self.run_desc_,
                miniters=miniters,
                total=ttotal
            )
        else:
            return giter

    def build(
        self,
        run_args: GBRArgs,
        take_n: int | None=None,
        use_tqdm: bool=True,
    ):
        start_time = timer() 
        self.init_importance(take_n)
        for target_gene in self.genes_itr(take_n, use_tqdm):
            self.gene_model(target_gene, run_args)
        self.rstats_.total_run_time = timer() - start_time
        print(f"Model Generation : {self.rstats_.total_run_time} seconds")
    
    def dump(
        self,
        rstats_out_file: str | None=None,
        out_file: str | None=None
    ):
        if rstats_out_file:
            with open(rstats_out_file, 'w') as ofx:
                ofx.write(self.rstats_.model_dump_json(indent=4))
        if (out_file is not None ) and (self.importance_ is not None):
            im_shape = self.importance_.shape
            tf_tgt_itr = itertools.product(
                self.sd_.gene_list[:im_shape[0]],
                self.sd_.tf_list
            )
            rdf = pd.DataFrame(
                tf_tgt_itr,
                columns=pd.Series(["target", "TF"])
            )
            rdf["importance"] = self.importance_.flatten()
            rdf.sort_values(by="importance", ascending=False, inplace=True)
            rdf.to_csv(out_file)


class ARBRunner:
    def __init__(self, exp_data: ExpDataProcessor) -> None:
        self.sd_ : ExpDataProcessor = exp_data
        self.run_time_ : float = 0.0
        self.network_ : pd.DataFrame | None = None

    def run_regression(
        self,
        method: GBMethod = 'arb:default',
        **_run_args: t.Any,
    ) -> pd.DataFrame:
        mat_shape = str(self.sd_.exp_matrix.shape)
        gene_l = str(len(self.sd_.gene_list)) 
        match method:
            case 'arb:gbm':
                print(f"Running GBM with {arboreto.core.GBM_KWARGS}")
                return arboreto.algo.diy(
                    expression_data=self.sd_.exp_matrix,
                    regressor_type='GBM',
                    regressor_kwargs=arboreto.core.GBM_KWARGS,
                    gene_names=self.sd_.gene_list,
                    tf_names=self.sd_.tf_list, #pyright:ignore[reportArgumentType]
                )
            case "arb:default" | _ :
                print(
                    f"Running GBM with default args {mat_shape} {gene_l}"
                )
                return arboreto.algo.grnboost2(
                    expression_data=self.sd_.exp_matrix,
                    gene_names=self.sd_.gene_list,
                    tf_names=self.sd_.tf_list, #pyright:ignore[reportArgumentType]
                )

    def build(
        self,
        method: GBMethod = 'arb:default',
        **run_args: t.Any
    ):
        start_time = timer()
        self.network_ = self.run_regression(method, **run_args)
        self.run_time_ = timer() - start_time
        print(f"Model Generation : {self.run_time_} seconds")

    def dump(
        self,
        rstats_out_file: str | None=None,
        out_file: str | None=None
    ):
        if rstats_out_file:
            with open(rstats_out_file, 'w') as ofx:
                json.dump({
                    "total_run_time": self.run_time_,
                }, ofx, indent=4)
        if (out_file is not None ) and (self.network_ is not None):
            self.network_.to_csv(out_file)
