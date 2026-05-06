from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, RunDir, SweepDir, JobConf


# --------- group: task ----------
@dataclass
class TaskCfg:
    _target_: str = MISSING
    _name: str = MISSING
    _group: str = "task"


@dataclass
class DecomposeTaskCfg(TaskCfg):
    _target_: str = "src.tasks.DecomposeTask"
    _name: str = "decompose"
    log_level: int = 0


@dataclass
class RandomMissingCompletionTaskCfg(TaskCfg):
    _target_: str = "src.tasks.RandomMissingCompletionTask"
    _name: str = "random-missing-completion"
    log_level: int = 0
    missing_rate: float = 0.5
    seed: int = 0


@dataclass
class StructuredMissingCompletionTaskCfg(TaskCfg):
    _target_: str = "src.tasks.StructuredMissingCompletionTask"
    _name: str = "structured-missing-completion"
    log_level: int = 0
    missing_rate: float = 0.5
    seed: int = 0
    pattern: str = "block"
    block_shape: Optional[Tuple[int, int]] = None
    stripe_axis: int = 1
    stripe_width: Optional[int] = None
    band_axis: int = -1


# --------- group: data ----------
@dataclass
class DataCfg:
    _target_: str = MISSING
    _name: str = MISSING
    _group: str = "data"


@dataclass
class RandDataCfg(DataCfg):
    _target_: str = "src.data.RandData"
    _name: str = "rand"
    shape: Tuple[int, ...] = (10, 10, 10)
    type: str = "normal"  # or "uniform"
    seed: int = 0


@dataclass
class TuckerDataCfg(DataCfg):
    _target_: str = "src.data.TuckerData"
    _name: str = "tucker"
    shape: Tuple[int, ...] = (10, 10, 10)
    rank: Tuple[int, ...] = (4, 4, 4)
    seed: int = 0


@dataclass
class CPDataCfg(DataCfg):
    _target_: str = "src.data.CPData"
    _name: str = "cp"
    shape: Tuple[int, ...] = (10, 10, 10)
    rank: Tuple[int, ...] = (4, 4, 4)
    cp_rank: Optional[int] = None
    seed: int = 0


@dataclass
class KodakDataCfg(DataCfg):
    _target_: str = "src.data.KodakData"
    _name: str = "kodak"
    path: str = "data/kodak"
    id: int = 0
    target_shape: Tuple[int, int] = (128, 128)


@dataclass
class CBSDDataCfg(DataCfg):
    _target_: str = "src.data.CBSDData"
    _name: str = "cbsd"
    path: str = "data/cbsd"
    id: int = 0
    target_shape: Tuple[int, int] = (128, 128)


@dataclass
class CAVEHSIDataCfg(DataCfg):
    _target_: str = "src.data.CAVEHSIData"
    _name: str = "cave_hsi"
    path: str = "data/CAVE"
    id: int = 1
    target_shape: Tuple[int, int] = (128, 128)
    crop_shape: Optional[Tuple[int, int]] = None


@dataclass
class IndianPinesHSIDataCfg(DataCfg):
    _target_: str = "src.data.IndianPinesHSIData"
    _name: str = "indian_pines_hsi"
    path: str = "data/hsi-benchmark/Indian_pines_corrected.mat"
    target_shape: Tuple[int, int] = (145, 145)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


@dataclass
class SalinasHSIDataCfg(DataCfg):
    _target_: str = "src.data.SalinasHSIData"
    _name: str = "salinas_hsi"
    path: str = "data/hsi-benchmark/Salinas_corrected.mat"
    target_shape: Tuple[int, int] = (512, 217)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


@dataclass
class BotswanaHSIDataCfg(DataCfg):
    _target_: str = "src.data.BotswanaHSIData"
    _name: str = "botswana_hsi"
    path: str = "data/hsi-benchmark/Botswana.mat"
    target_shape: Tuple[int, int] = (128, 128)
    crop_shape: Optional[Tuple[int, int]] = (100, 100)
    normalize: str = "max"


@dataclass
class PaviaHSIDataCfg(DataCfg):
    _target_: str = "src.data.PaviaHSIData"
    _name: str = "pavia_hsi"
    path: str = "data/PaviaU/PaviaU.mat"
    target_shape: Tuple[int, int] = (128, 128)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


@dataclass
class JasperRidgeHSIDataCfg(DataCfg):
    _target_: str = "src.data.JasperRidgeHSIData"
    _name: str = "jasper_ridge_hsi"
    path: str = "data/hsi/jasperRidge2_R198.mat"
    target_shape: Tuple[int, int] = (100, 100)
    crop_shape: Optional[Tuple[int, int]] = (100, 100)
    normalize: str = "max"


@dataclass
class SamsonHSIDataCfg(DataCfg):
    _target_: str = "src.data.SamsonHSIData"
    _name: str = "samson_hsi"
    path: str = "data/hsi-similar/samson_1.img"
    target_shape: Tuple[int, int] = (95, 95)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


@dataclass
class UrbanHSIDataCfg(DataCfg):
    _target_: str = "src.data.UrbanHSIData"
    _name: str = "urban_hsi"
    path: str = "data/hsi-similar/Urban_R162.mat"
    target_shape: Tuple[int, int] = (307, 307)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


@dataclass
class CupriteHSIDataCfg(DataCfg):
    _target_: str = "src.data.CupriteHSIData"
    _name: str = "cuprite_hsi"
    path: str = "data/hsi-similar/Cuprite_S1_R188.img"
    target_shape: Tuple[int, int] = (250, 190)
    crop_shape: Optional[Tuple[int, int]] = None
    normalize: str = "max"


# --------- group: filter ----------
@dataclass
class FilterCfg:
    _target_: str = MISSING
    _name: str = MISSING
    _group: str = "filter"
    seed: int = 0
    normalize_method: Optional[str] = "energy"  # "energy" | "max"
    snr_db: Optional[float] = None


@dataclass
class BiasFilterCfg(FilterCfg):
    _target_: str = "src.filters.BiasFilter"
    _name: str = "bias-filter"
    bias: Optional[float] = None


@dataclass
class NonlinearFilterCfg(FilterCfg):
    _target_: str = "src.filters.NonlinearFilter"
    _name: str = "nonlinear-filter"
    nonlinear: str = "none"  # "none" | "sin" | "tanh" | "poly2" | "poly3" | "poly34" | "exp"
    alpha: float = 0.0

    
    
# --------- group: method ----------
@dataclass
class MethodCfg:
    _target_: str = MISSING
    _name: str = MISSING
    _group: str = "method"
    rank: Tuple[int, ...] = (4, 4, 4)
    n_iter_max: int = 1000


@dataclass
class NTDPLDecompositionCfg(MethodCfg):
    _target_: str = "src.methods.NTDPLDecomposition"
    _name: str = "ntdpl"
    init_n_iter_max: int = 50
    init: str = "tucker"
    solver_variant: str = "optimized"  # "optimized" | "base"
    stable_beta_update: bool = True
    beta_update_stage: str = "before_grad"  # "before_grad" | "after_grad"
    random_state: int = 0
    p_max: int = 2
    allow_constant_term: bool = True
    use_continuation: bool = True
    factor_normalize: bool = True
    lr_core: float = 1e-4
    lr_factors: float = 3e-4
    lambda_core: float = 1e-6
    lambda_factors: float = 1e-6
    lambda_beta: float = 1e-6
    beta_update_method: str = "moments_normal_eq"  # "moments_normal_eq" | "ridge_lstsq"
    beta_update_interval: int = 5


@dataclass
class TuckerDecompositionCfg(MethodCfg):
    _target_: str = "src.methods.TuckerDecomposition"
    _name: str = "tucker"
    init: str = "svd"
    tol: float = 1e-4


@dataclass
class CPDecompositionCfg(MethodCfg):
    _target_: str = "src.methods.CPDecomposition"
    _name: str = "cp"
    cp_rank: Optional[int] = None
    init_method: str = "random"  # or "svd"
    tol: float = 1e-8
    random_state: int = 0
    normalize_factors: bool = False


@dataclass
class TRDecompositionCfg(MethodCfg):
    _target_: str = "src.methods.TRDecomposition"
    _name: str = "tr"
    tr_rank: Optional[Tuple[int, ...]] = None
    svd: str = "truncated_svd"  # or "full_svd"


@dataclass
class TTDecompositionCfg(MethodCfg):
    _target_: str = "src.methods.TTDecomposition"
    _name: str = "tt"
    tt_rank: Optional[Tuple[int, ...]] = None
    svd: str = "truncated_svd"  # or "full_svd"


@dataclass
class SoftImputeCfg(MethodCfg):
    _target_: str = "src.methods.softimpute.SoftImputeCompletion"
    _name: str = "softimpute"
    matrix_rank: Optional[int] = None
    shrinkage_value: Optional[float] = None
    outer_n_iter_max: int = 20
    matrix_tol: float = 1e-5
    tol: float = 1e-5
    init_fill: str = "mean"   # "mean" | "zero"
    unfold_modes: Optional[Tuple[int, ...]] = None


# --------- top-level config ----------
@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: [
        {"task": "decompose"},
        {"data": "rand"},
        {"method": "cp"},
        {"filter": "bias-filter"},
        "_self_"
    ])

    exp: str = MISSING
    exp_mode: str = "run"  # 'benchmark' or 'run' for experiment mode
    task: TaskCfg = MISSING
    data: DataCfg = MISSING
    method: MethodCfg = MISSING
    filter: FilterCfg = MISSING

    hydra: HydraConf = field(default_factory=lambda: HydraConf(
        run=RunDir(dir="outputs/${exp}/${exp_mode}"),
        sweep=SweepDir(
            dir="multirun/${exp}/${exp_mode}/${now:%Y-%m-%d_%H-%M-%S}",
            subdir="${hydra.job.num}"
        ),
        output_subdir=".hydra",
        job=JobConf(chdir=False),
    ))


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    Cfg_list = [
        DecomposeTaskCfg,
        RandomMissingCompletionTaskCfg,
        StructuredMissingCompletionTaskCfg,
        RandDataCfg,
        KodakDataCfg,
        CBSDDataCfg,
        TuckerDataCfg,
        CPDataCfg,
        CAVEHSIDataCfg,
        IndianPinesHSIDataCfg,
        SalinasHSIDataCfg,
        BotswanaHSIDataCfg,
        PaviaHSIDataCfg,
        JasperRidgeHSIDataCfg,
        SamsonHSIDataCfg,
        UrbanHSIDataCfg,
        CupriteHSIDataCfg,
        BiasFilterCfg,
        NonlinearFilterCfg,
        NTDPLDecompositionCfg,
        TuckerDecompositionCfg,
        CPDecompositionCfg,
        TRDecompositionCfg,
        TTDecompositionCfg,
        SoftImputeCfg,
    ]
    for cfg in Cfg_list:
        cs.store(group=cfg._group, name=cfg._name, node=cfg)
