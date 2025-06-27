import os
from typing import Tuple, List, Literal
from stat_predict.static.utils import COLUMNS, major_attrs

# Data params
TEST_YEAR = 24

# Label params
LABEL_COLUMN = COLUMNS.PLAYER_RATING


# Features params
class FeaturesParameters:
    add_hist_features = True
    hist_features_columns = major_attrs + ['international_reputation', COLUMNS.PLAYER_VALUE]
    metadata_prior_columns = [
        COLUMNS.NATIONALITY,
        COLUMNS.CLUB,
        COLUMNS.AGE,
        COLUMNS.LEAGUE,
        COLUMNS.PLAYER_POSITION
    ]


# Training params
class DefaultParameters:
    feature_selection_steps: int = 3
    classes_weights = (0.5, 1.)
    batch_size: int = 64
    learning_rate: float = 1e-5
    num_epochs: int = 200
    early_stop: int = 10
    num_features_to_select: int = 125
    selection_correlation_method: Literal['pearson', 'kendall', 'spearman'] = 'pearson'
    filter_feature_families: List[str] = []
    correlation_threshold: float = 0.99
    weight_decay: float = 1e-2
    eps: float = 1e-8
    betas: Tuple[float] = (0.9, 0.999)


# Defining bins for various attributes: YoY change, age, rating, etc.
YOY_DIFF_TO_CAT = tuple([0])
YOY_DIFFS_TO_CATS = [-0.02, -0.01, 0.0, 0.03]
YOY_DIFFS_TO_CATS_NAMES = ['sever-decay', 'mild-decay', 'stagnation', 'mild-growth', 'acc-growth']
YOY_CATS = ('decay', 'growth')
AGE_TO_CATS = [18, 21, 24, 27, 30, 35]
AGE_BINS_NAMES = ['<18', '19-21', '22-24', '25-27', '28-30', '30-35', '35+']
RATING_BINS = [60, 70, 75, 80, 85, 90, 100]
RATING_BINS_NAMES = ['<60', '60-70', '70-75', '76-80', '81-85', '85-90', '90+']
AGE_NORM_VALUE = 36


class Font:
    FAMILY = 'Lucida Grande'
    LABEL_SIZE = 16
    TITLE_SIZE = 24


class ArtifactsPaths:
    FEATURE_IMPORTANCE: str = 'feature_importance.csv'
    PROC_DB_PATH = 'processed_data.pickle'
    DATA_DIR = 'data'
    ARTIFACTS_DIR = 'artifacts'
    ALL_YERAS_DATA = 'players_15-24.csv'
    ALL_YEARS_MERGED_DATA = 'StatPrediction_data.csv'
    LAST_YEAR_LABELS = 'players_25.csv'  # data is not null, but we have the labels
    ALL_YEARS_PROCESSED_DF = 'processed_df_path.pickle'
    STAT_DATASET = 'stat_predict_dataset_{NUM_YEARS_BACK}yback.pickle'
    CAREER_PHASE_DIR = os.path.join('artifacts', 'career_phase')
