import itertools
import os.path
import pickle
import warnings
from collections import defaultdict
from typing import Tuple, List, Optional, Literal, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
import plotly.express as px

from stat_predict.dataset.sp_dataset import StatsDataset
from stat_predict.dataset.utils import FeatureFamilies
from stat_predict.eval.evaluation import model_evaluation, build_final_report
from stat_predict.static.config import AGE_NORM_VALUE, LABEL_COLUMN
from stat_predict.static.utils import COLUMNS, TS_COLS, ts_cols_suffix


def get_base_attribute_ts_columns(df: pd.DataFrame, attribute: str) -> List[str]:
    if attribute in TS_COLS.__dict__:
        # Input is a ts attribute, return all columns refer to base attribute
        return sorted([c for c in df.columns if c.startswith(attribute[:-1])])
    else:
        # Input is a basic attribute - not relate to a specific year
        return sorted([c for c in df.columns if c.startswith(attribute) and c != attribute])


def gen_slice_name(attr: str, bin_edges: Tuple[Tuple]) -> str:
    return f"{attr.split('_row_')[0]}::{bin_edges[0]}->{bin_edges[1]}"


def break_bin_name_to_parts(bin_name: str) -> (str, float, float):
    """
    Inverse of gen_slice_name function:
    Breaks a bin name into its parts: attr, bin_bottom, bin_top
    - E.g., 'age:16->23' --> attr = age, bin_bottom=16, bin_top=23
    """
    attr = bin_name.split('::')[0]
    bin_bottom = float(bin_name.split('::')[1].split('->')[0])
    bin_top = float(bin_name.split('::')[1].split('->')[1])
    return attr, bin_bottom, bin_top


class CareerPhaseModelArgs(object):
    def __init__(self, **kwargs):
        self.clustering_metric: str = kwargs.get('clustering_metric', 'dtw')
        self.n_clusters: int = kwargs.get('n_clusters', 7)
        self.clustering_attr: List[str] = kwargs.get('clustering_attr', [LABEL_COLUMN])
        self.num_years_back: int = kwargs.get('num_years_back', 4)
        self.min_years_back: int = kwargs.get('min_years_back', self.num_years_back - 1)
        self.series_na_value: float = kwargs.get('series_na_value', -1.)
        self.series_manipulation: Optional[Literal[None, 'diff', 'pct_change']] = (
            kwargs.get('series_manipulation_function', None))
        self.knn_series_manipulation: Optional[Literal[None, 'diff', 'pct_change']] = kwargs.get(
            'knn_series_manipulation', self.series_manipulation)
        self.model_dir = kwargs.get('model_dir', 'artifacts/career_phase')
        self.grid_params = {'n_neighbors': [5, 7, 9],
                            'metric': ['euclidean', 'cosine']}
        self.knn_attrs = kwargs.get('knn_attrs', [LABEL_COLUMN])
        self.yoy_dist_features: List[str] = kwargs.get('yoy_dist_features',
                                                       ['mean', 'std', 'min', 'p0.1', 'p0.25', 'p0.5', 'p0.75',
                                                        'p0.85', 'p0.95', 'max'])


class CareerPhaseModel:
    def __init__(self, **kwargs):
        self.cls_name: str = kwargs.get('name', 'CareerPhaseModel')

        # Clustering params
        self.model_args: CareerPhaseModelArgs = CareerPhaseModelArgs(**kwargs)
        self.model = None
        self.clusters_stats = {}
        self.classifier = None

    def get_attr_clustering_columns(self, attr: str) -> List[str]:
        return sorted([f"{attr}_row_{i}" for i in range(self.model_args.num_years_back)])

    @property
    def clustering_columns(self) -> List[str]:
        ret = []
        for _attr in self.model_args.clustering_attr:
            ret.extend(self.get_attr_clustering_columns(_attr))
        return ret

    @property
    def knn_columns(self) -> List[str]:
        ret = []
        for _attr in self.model_args.knn_attrs:
            ret.extend(self.get_attr_clustering_columns(_attr))
        return ret

    def columns_to_keep(self) -> List[str]:
        return sorted(list(set(self.clustering_columns) | set(self.knn_columns)))

    @property
    def name(self) -> str:
        return self.cls_name

    @property
    def model_path(self) -> str:
        if not os.path.exists(self.model_args.model_dir):
            os.mkdir(self.model_args.model_dir)
        return os.path.join(self.model_args.model_dir, f"{self.name}.pickle")

    @property
    def artifacts_path(self) -> str:
        if not os.path.exists(os.path.join(self.model_args.model_dir, self.name)):
            os.mkdir(os.path.join(self.model_args.model_dir, self.name))
        return os.path.join(self.model_args.model_dir, self.name)

    def data_split(self,
                   dataset: StatsDataset
                   ) -> (Dict[str, pd.DataFrame], Dict[str, np.array], Dict[str, np.array], Dict[str, np.array]):
        """ Splits the data by time
        :return: x_train, y_train, val, y_val, test, y_test
        """
        print('- Splitting dataset to train, validation, test...')

        data = {}
        labels = {}
        raw_labels = {}
        ts_data = {}
        for _set in ['train', 'validation', 'test']:
            x = dataset.data[_set]
            x.set_index(pd.Series([int(v[COLUMNS.PLAYER_ID]) for v in dataset.labels[_set]]), inplace=True)
            data[_set], valid_indices, ts_data[_set] = self.preprocess_data(
                x, handle_missing_values='filter' if _set != 'test' else 'impute'
            )
            labels[_set] = np.array([int(v['label']) for v in dataset.labels[_set]])[valid_indices]
            raw_labels[_set] = np.array([v['raw_label'] for v in dataset.labels[_set]])[valid_indices]
            assert len(labels[_set]) == len(data[_set])
        return data, ts_data, labels, raw_labels

    def impute_missing_years(self, x):
        # Fill missing years per player
        for attr in self.model_args.clustering_attr:
            # Interpolate ts
            x[f'raw_{attr}'] = x[attr].apply(
                lambda vec: list(pd.Series(vec).interpolate(method='linear', limit_direction='both').round(3)))
            x[attr] = x[attr].apply(
                lambda vec: list(pd.Series(vec).interpolate(method='linear', limit_direction='both').round(3)))

            # Iterate all columns relate to this attribute and interpolates them as well for KNN
            curr_cols = self.get_attr_clustering_columns(attr)
            x.loc[:, curr_cols] = x.loc[:, curr_cols].replace(0, np.nan) \
                .astype(float) \
                .interpolate(method='linear', limit_direction='both', axis=1)

        return x

    def calc_ts_coverage(self, x: pd.DataFrame) -> np.array:
        ts_coverage = ((x[self.clustering_columns] == self.model_args.series_na_value).astype(float) +
                       x[self.clustering_columns].isna().astype(float)) \
            .mean(axis=1)
        return ts_coverage.to_numpy()

    def filter_rows_with_missing_data(self, x: pd.DataFrame) -> (pd.DataFrame, List):
        """
        Filter rows with missing values from dataframe, return:
         - pd.DataFrame: filtered dataframe
         - List[bool]: a boolean mask of remained rows
        """
        valid_indices = [True] * len(x)
        # Iterate over all time series and mark rows that do not miss data across all ts series
        for attr in self.model_args.clustering_attr:
            # Filter players with at least k rows
            series_na_count = x[attr].apply(lambda vec: sum([pd.isna(v) for v in vec]))
            _valid_indices = pd.Series(series_na_count == 0).tolist()
            # Apply AND - valid in for ALL ts series
            valid_indices = [valid_indices[i] and _valid_indices[i] for i in range(len(valid_indices))]
        x = x[valid_indices]
        return x, valid_indices

    def apply_ts_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.model_args.clustering_attr:
            if self.model_args.series_manipulation == 'diff':
                df[col] = df[col].apply(lambda vec: pd.Series(vec).diff()[1:].round(3).tolist())
            elif self.model_args.series_manipulation == 'pct_change':
                df[col] = df[col].apply(lambda vec: pd.Series(vec)[1:].pct_change().round(3).tolist())
            elif self.model_args.series_manipulation is not None:
                raise ValueError(f"unfamiliar value for CareerPhaseModel.series_manipulation: "
                                 f"{self.model_args.series_manipulation}")

        for attr in self.model_args.knn_attrs:
            if attr == COLUMNS.AGE:
                # No need to apply transformation on age
                continue

            attr_columns = self.get_attr_clustering_columns(attr)
            if attr == COLUMNS.LEAGUE_LEVEL:
                df[attr_columns] = df[attr_columns].fillna(0)

            if self.model_args.knn_series_manipulation == 'diff':
                df[attr_columns] = df[attr_columns].apply(lambda vec: vec.diff().round(3), axis=1)
                # Last attr will be NA -> fill with zero to not affect knn
                df[attr_columns[0]] = df[attr_columns[0]].fillna(0)
            elif self.model_args.knn_series_manipulation == 'pct_change':
                df[attr_columns] = df[attr_columns].apply(lambda vec: vec.pct_change().round(3), axis=1)
                # Last attr will be NA -> fill with zero to not affect knn
                df[attr_columns[0]] = df[attr_columns[0]].fillna(0)
            elif self.model_args.knn_series_manipulation is not None:
                raise ValueError(f"unfamiliar value for CareerPhaseModel.knn_series_manipulation: "
                                 f"{self.model_args.knn_series_manipulation}")
        return df

    def transform_into_ts(self, x: pd.DataFrame) -> np.array:
        """
        Takes all time series columns (can be multiple vectors) and return a float ts vector for each row.
        In case of multiple vectors, they are concat horizontally
        """
        x.loc[:, 'clustering_attr'] = x.loc[:, self.model_args.clustering_attr].apply(lambda row: sum(row, []), axis=1)
        return np.vstack(np.array(x['clustering_attr'].tolist()))

    def preprocess_data(self,
                        x: pd.DataFrame,
                        handle_missing_values: Literal['impute', 'filter'] = 'impute') \
            -> (pd.DataFrame, List, np.array):
        # Get all timeseries columns given
        cluster_attrs_columns = {_attr: self.get_attr_clustering_columns(_attr)
                                 for _attr in self.model_args.clustering_attr}
        columns_to_keep = self.columns_to_keep()
        x = x[columns_to_keep]

        # Fill Time-Series
        for clstr_attr in self.model_args.clustering_attr:
            # Put NAs instead of zeroes, round values
            x.loc[:, cluster_attrs_columns[clstr_attr]] = x[cluster_attrs_columns[clstr_attr]] \
                .replace(self.model_args.series_na_value, np.nan) \
                .astype(float) \
                .round(3)
            # Vectorize
            x.loc[:, clstr_attr] = x[cluster_attrs_columns[clstr_attr]].apply(lambda vec: list(vec)[::-1], axis=1)

        if handle_missing_values == 'impute':
            x_imputed = self.impute_missing_years(x)
            valid_indices = np.array([True] * len(x_imputed))
        elif handle_missing_values == 'filter':
            x_imputed, valid_indices = self.filter_rows_with_missing_data(x)
        else:
            raise ValueError(f"Unfamiliar value for '': {handle_missing_values}, "
                             f"possible values are 'impute' (prediction) or 'filter' (train)")

        # Merge into a single vec
        x_imputed = self.apply_ts_transformation(x_imputed)
        return x_imputed, valid_indices, self.transform_into_ts(x_imputed)

    def plot_centroids(self):
        centroids = self.model.cluster_centers_[:, :, 0]
        df = pd.DataFrame(centroids, columns=list(range(centroids.shape[1])))
        df['Cluster'] = list(range(centroids.shape[0]))

        # Convert wide format to long format
        df_long = df.melt(id_vars=['Cluster'], var_name='year', value_name='value').round(3)

        # Plot with Plotly
        fig = px.line(df_long,
                      x='year',
                      y='value',
                      color='Cluster',
                      markers=True,
                      title=f"Phase Model Clusters: Slice - "
                            f"{self.name.split('slice:')[1].replace('::', ' = ').replace('>', '').replace('&', ' & ').title()}")
        fig.update_layout(template="plotly_white",
                          yaxis_title='Player Rating',
                          xaxis_title='Years')
        fig.show()

    def fit_classifier(self, x_train: pd.DataFrame, y_train: np.array):
        grid_params = self.model_args.grid_params.copy()
        # Check if given population is too small to be searched (relevant for SplitCareerPhaseModel)
        if 'n_neighbors' in grid_params and max(grid_params['n_neighbors']) < len(x_train):
            grid_params['n_neighbors'] = [len(x_train)]
            knn_grid = GridSearchCV(KNeighborsClassifier(), grid_params)
            knn_grid.fit(x_train, y_train)
            best_params = knn_grid.best_params_
        else:
            best_params = {'n_neighbors': min(len(x_train), min(grid_params['n_neighbors'])), 'metric': 'euclidean'}
        self.classifier = KNeighborsClassifier(**best_params)
        self.classifier.fit(x_train, y_train)

    def fit(self,
            data: Dict[str, pd.DataFrame],
            ts_data: Dict[str, np.array],
            labels: Dict[str, np.array],
            raw_labels=Dict[str, np.array],
            use_validation: bool = True,
            export_model: bool = True
            ):
        # x, valid_train_indices, ts_train = self.preprocess_data(data['train'], handle_missing_values='filter')
        x = data['train']
        ts_train = ts_data['train']
        y = labels['train']
        y_raw = raw_labels['train']
        if use_validation and 'validation' in data and 'validation' in labels:
            x = pd.concat([x, data['validation']], axis=0)
            ts_train = np.vstack([ts_train, ts_data['validation']])
            y = np.hstack([y, labels['validation']])
            y_raw = np.hstack([y_raw, raw_labels['validation']])
        self.model = TimeSeriesKMeans(n_clusters=self.model_args.n_clusters, metric=self.model_args.clustering_metric)

        print('\nFitting KNN..')
        print(' [info] KNN columns:', self.knn_columns)
        self.fit_classifier(x[self.knn_columns], y)

        print('\nFitting clustering...')
        print(' [info] clustering columns:', self.clustering_columns)
        labels = self.model.fit_predict(ts_train)

        # diff between label rating to latest rating
        yoy_diffs = pd.Series(y_raw - x[TS_COLS.PLAYER_RATING].to_numpy()).astype(float).round(3)
        yoy_diffs_dist = pd.DataFrame({'cluster': labels, 'yoy_diff': yoy_diffs}) \
                             .groupby('cluster') \
                             .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.85, 0.95]) \
                             .reset_index().iloc[:, 2:]
        yoy_diffs_dist.columns = self.model_args.yoy_dist_features[:]
        self.clusters_stats = {
            'player2cluster': {x.index[i]: labels[i] for i in range(len(x))},
            'cls': pd.DataFrame({'cluster': labels, 'label': y}).groupby('cluster').mean()['label'].to_dict(),
            'yoy_diff': yoy_diffs_dist
        }

        if export_model:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'clusters_stats': self.clusters_stats,
                    'classifier': self.classifier,
                    'args': self.model_args
                }, f)

    def train(self, dataset: StatsDataset, use_existing: bool = True):
        data, ts_data, labels, raw_labels = self.data_split(dataset)
        if use_existing and os.path.exists(self.model_path):
            self.load_model()
        else:
            self.fit(data, ts_data, labels, raw_labels)
        return self.evaluate(data['test'], ts_data['test'], labels['test'], raw_labels['test'], dataset)

    def predict(self, x: pd.DataFrame, ts_data: Optional[np.array] = None) -> Dict:
        """
        Input: dataframe of players stats over years.
            - Mandatory columns: COLUMNS.YEAR, COLUMNS.PLAYER_ID, self.clustering_attr
        Return: Dict
            'clusters': np.array of cluster associated with each prediction item
            'cls_preds': np.array of KNN based prediction associated with each prediction item
            'cls_probs': np.array of KNN based probability associated with each prediction item
            'cluster_prob': np.array of cluster members label prior, associated with each prediction item
            'cluster_yoy_dist': distribution of the cluster member yoy diff, associated with each prediction item
            'ts_confidence': % of years the player had / num years back model is using
        """
        ts_coverage = self.calc_ts_coverage(x)
        if ts_data is None:
            x, valid_indices, ts_data = self.preprocess_data(x.copy(), handle_missing_values='impute')

        # Get cluster
        pred_clusters = self.model.predict(ts_data)

        # Get all cluster's members
        cluster_next_year_pred = [self.clusters_stats['cls'][cluster] for cluster in pred_clusters]
        cluster_yoy_dist = [self.clusters_stats['yoy_diff'].iloc[cluster] for cluster in pred_clusters]
        knn_preds = self.classifier.predict(x[self.knn_columns])
        knn_probs = self.classifier.predict_proba(x[self.knn_columns])
        knn_probs = knn_probs[:, knn_probs.shape[1] - 1]

        return {
            'clusters': pred_clusters,
            'cls_preds': knn_preds,
            'cls_probs': knn_probs,
            'cluster_prob': cluster_next_year_pred,
            'cluster_yoy_dist': cluster_yoy_dist,
            'ts_confidence': ts_coverage
        }

    def add_career_phase_attributes(self,
                                    df: pd.DataFrame,
                                    prefix: str = FeatureFamilies.phase_model,
                                    ) -> (pd.DataFrame, Dict[str, List[str]]):
        """
        Add predictions and other attributes to a given dataframe, and a list of the attributes added
        """
        orig_columns = list(df.columns)
        predictions_dict = self.predict(df.copy())
        categorical_features = [f'{prefix}:clusters', f'{prefix}:slices']
        for key, val in predictions_dict.items():
            key_name = f'{prefix}:{key}'
            if isinstance(val, pd.DataFrame):
                cat_df = predictions_dict[key]
                cat_df.columns = [f'{prefix}:{c}' for c in cat_df.columns]
                cat_df.index = df.index
                df = pd.concat([df, cat_df.fillna(self.model_args.series_na_value)], axis=1)
            else:
                df[key_name] = list(val)
                df[key_name] = df[key_name].fillna(self.model_args.series_na_value)

            if key_name in categorical_features:
                if key == 'clusters':
                    df[key_name] = df[key_name].astype(int)
                df[key_name] = df[key_name].astype('category')
                if int(self.model_args.series_na_value) not in df[key_name].cat.categories:
                    df[key_name] = df[key_name].cat.add_categories(int(self.model_args.series_na_value))
        added_features = {'numeric': [c for c in df.columns if c not in orig_columns and c not in categorical_features],
                          'categorical': categorical_features[:]}
        return df, added_features

    def evaluate(self,
                 test: pd.DataFrame,
                 ts_test: np.array,
                 y_test: np.array, raw_labels: np.array,
                 dataset: StatsDataset) -> (Dict, Dict):
        print(f'\nevaluating {self.name}...')

        # Collect attributes for analysis
        test_players_ratings = np.array([v['raw_label'] for v in dataset.labels['test']])
        test_ratings_y1_back = dataset.data['test'][TS_COLS.PLAYER_RATING].to_numpy()
        test_ratings_yoy_diffs = test_players_ratings - test_ratings_y1_back  # diff between label rating to latest rating
        test_players_names = dataset.data['test'][TS_COLS.PLAYER_FULL_NAME].tolist()
        test_players_ages = dataset.data['test'][TS_COLS.AGE].tolist()
        num_years_data = dataset.data['test'][[c for c in dataset.data['test'].columns if c.startswith('pad_')]].sum(
            axis=1).apply(lambda x: self.model_args.num_years_back - x).tolist()

        # Evaluate
        predictions_dict = self.predict(test, ts_data=ts_test)
        predictions = [int(v) if v is not None else int(0) for v in predictions_dict['cls_preds']]
        probs = [v if v is not None else 0 for v in predictions_dict['cls_probs']]
        cluster_prob = [v if v is not None else 0 for v in predictions_dict['cluster_prob']]
        clustering_cls_predictions = [int(cluster_prob[i] > 0.5) if cluster_prob[i] is not None else int(0)
                                      for i in range(len(cluster_prob))]
        print('Evaluating KNN model')
        knn_report = model_evaluation(
            y_test,
            predictions,
            probs,
            rating_labels=list(test_players_ratings),
            ratings_yoy_diffs=test_ratings_yoy_diffs,
            players_ratings=test_ratings_y1_back,
            players_names=test_players_names,
            players_age=test_players_ages,
            output_dir=self.artifacts_path,
            model_name=self.name + "-KNN",
            num_years_data=num_years_data,
            prints=False
        )

        print('Evaluating Clustering model')
        clustering_report = model_evaluation(
            y_test,
            clustering_cls_predictions,
            cluster_prob,
            rating_labels=list(test_players_ratings),
            ratings_yoy_diffs=test_ratings_yoy_diffs,
            players_ratings=test_ratings_y1_back,
            players_names=test_players_names,
            players_age=test_players_ages,
            output_dir=self.artifacts_path,
            model_name=self.name + "-Clustering",
            num_years_data=num_years_data,
            prints=False
        )
        return knn_report, clustering_report

    def load_model(self, from_path: Optional[str] = None):
        _path = from_path if from_path is not None else self.model_path
        print(f'Loading CareerPhaseModel from {_path}...')
        with open(_path, 'rb') as f:
            pkl = pickle.load(f)
        self.model = pkl['model']
        self.classifier = pkl['classifier']
        self.clusters_stats = pkl['clusters_stats']
        self.model_args = pkl['args']


class SplitCareerPhaseModel(CareerPhaseModel):
    def __init__(self, splits_config: Dict[str, Tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self._name = kwargs.get('model_name', None)
        self.splits_config: Dict[str, Tuple] = splits_config
        self.slices = None
        self.models: Dict[Any, CareerPhaseModel] = {}
        if self.splits_config is not None:
            self.init()

        self.inv_scaler = {COLUMNS.AGE: AGE_NORM_VALUE}

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        slices_attrs = list(self.splits_config.keys())
        series_manip_str, knn_series_manip_str = '', ''
        if self.model_args.knn_series_manipulation is not None:
            knn_series_manip_str = self.model_args.knn_series_manipulation
        if self.model_args.series_manipulation not in [None, 'raw']:
            series_manip_str = f"_series-{self.model_args.series_manipulation}_"

        def short_attr(s: str):
            if 'position' in s:
                return 'Pos'
            if '_row_' in s:
                s = s.split('_row_')[0]
            if '_' in s:
                return ''.join([_s[:1].capitalize() for _s in s.split('_')])
            else:
                return s[:3].capitalize()

        def summarize_attrs(attr_list) -> str:
            return '-'.join([f"{short_attr(att)}{len(self.splits_config[att])}" for att in attr_list])

        knn_attrs_str = '-'.join([f"{att[:3]}" for att in self.model_args.knn_attrs])
        return (f"Split-by-{summarize_attrs(slices_attrs)}{series_manip_str}"
                f"_{self.model_args.num_years_back}y_nc{self.model_args.n_clusters}_knn"
                f"-{knn_attrs_str}{knn_series_manip_str}")

    def columns_to_keep(self) -> List[str]:
        columns = super().columns_to_keep()
        return sorted(list(set(columns + list(self.splits_config.keys()))))

    def generate_slices(self) -> List[str]:
        """
        Generate all possible slices of the DataFrame based on the given attributes and their bins.
        bins_dict (dict): Dictionary of {attribute: list of bin edges}.

        Returns:
        dict: A dictionary where keys are slice names and values are DataFrame slices.
        """
        slices = []
        for attr, bin_edges in self.splits_config.items():
            slices.append([gen_slice_name(attr, bin_edges[i]) for i in range(len(bin_edges))])

        if len(slices) > 1:
            return ['&'.join(list(p)) for p in itertools.product(*slices)]
        else:
            return slices[0]

    def deduce_slice_name(self, row: pd.Series):
        """
        Deduce the appropriate slice name for a given row based on the bin definitions.

        Args:
        row (pd.Series): A single row from a DataFrame.
        attributes (list): List of attributes to slice by.
        bins_dict (dict): Dictionary of {attribute: list of bin edges}.

        Returns:
        str: The name of the slice the row belongs to, or None if it doesn't fit into any bin.
        """
        slice_names = []

        for attr, bin_edges in self.splits_config.items():
            for i in range(len(bin_edges)):
                if bin_edges[i][0] <= row[attr] < bin_edges[i][1]:
                    slice_names.append(gen_slice_name(attr, bin_edges[i]))
                    break

        return "_".join(slice_names) if slice_names else None

    def slice_dataframe(self, df: pd.DataFrame) -> (Dict[str, pd.DataFrame], Dict[str, List]):
        """
        Slice the entire DataFrame based on attribute bins.

        Args:
        df (pd.DataFrame): The DataFrame to slice.
        attributes (list): List of attributes to slice by.
        bins_dict (dict): Dictionary of {attribute: list of bin edges}. - bins edges CANNOT overlap

        Returns:
        dict: A dictionary where keys are slice names and values are DataFrame slices.
        """
        df = df.copy()
        df['index'] = list(range(len(df)))
        slices = {}
        slices_indexes = {}

        # Convert lists to DataFrames
        for key in self.slices:
            # Extract slice components from its name (each attribute is separated by '&')
            query = []
            slice_bin_names = key.split('&')
            for bin_name in slice_bin_names:
                attr, bin_bottom, bin_top = break_bin_name_to_parts(bin_name)
                ts_attr = f"{attr}{ts_cols_suffix}"
                if attr in self.inv_scaler:
                    bin_bottom, bin_top = bin_bottom / self.inv_scaler[attr], bin_top / self.inv_scaler[attr]

                if ':' in ts_attr:
                    ts_attr = ts_attr.replace(':', '_')
                    df[ts_attr] = df[f"{attr}{ts_cols_suffix}"]
                query.append(f"{bin_bottom} <= {ts_attr} and {ts_attr} <= {bin_top}")

            slices[key] = df.query(' and '.join(query))
            slices_indexes[key] = slices[key]['index'].tolist()

        # Verify all input players are covered in one of the slices - both dfs and indexes
        covered_indices = set([ii for i in slices_indexes.values() for ii in i])
        uncovered_indices = [i for i in df['index'] if i not in covered_indices]
        uncovered_df = df.iloc[uncovered_indices]
        if len(uncovered_df) > 0:
            warnings.warn(f"{uncovered_df} players were uncovered by the phase model")
        return slices, slices_indexes

    def assert_ranges_are_contiguous_and_exclusive(self):
        for attr, bins_ranges in self.splits_config.items():
            # Sort ranges by their start values
            sorted_ranges = sorted(bins_ranges, key=lambda x: x[0])

            for i in range(len(sorted_ranges) - 1):
                current_end = sorted_ranges[i][1]
                next_start = sorted_ranges[i + 1][0]

                # Check for gap or overlap
                if current_end != next_start:
                    raise AssertionError(
                        f"{attr} ranges are not contiguous or exclusive "
                        f"between {sorted_ranges[i]} and {sorted_ranges[i + 1]}"
                    )

    def init(self):
        self.models = {}
        self.slices = self.generate_slices()
        for slice in self.slices:
            self.models[slice] = CareerPhaseModel(
                name=f"{self.name}/slice:{slice}",
                **self.model_args.__dict__
            )

    def get_split_range(self, attr_name, attr_val: float | int):
        for i, split_range in enumerate(self.splits_config[attr_name]):
            if self.splits_config[attr_name][i][0] <= attr_val <= self.splits_config[attr_name][i][1]:
                return i
        return len(self.splits_config[attr_name])

    def slice_stats_dataset(self,
                            data: Dict[str, pd.DataFrame],
                            ts_data: Dict[str, np.array],
                            labels: Dict[str, np.array],
                            raw_labels: Dict[str, np.array]) \
            -> (Dict[str, pd.DataFrame], Dict[str, np.array], Dict[str, pd.array], Dict[str, pd.array]):
        """
        Split data, ts data, labels and raw_labels according to the slicing config
        :return: Tuple
            - sliced data: Dict[str, pd.DataFrame] : for each slice (str) -> relevant sub-dataframe from data
            - sliced ts data: Dict[str, np.array] : for each slice (str) -> relevant sub-array from ts_data
            - sliced labels: Dict[str, np.array] : for each slice (str) -> relevant sub-array from labels
            - sliced raw_labels: Dict[str, np.array] : for each slice (str) -> relevant sub-array from raw_labels
        """
        sliced_data, sliced_ts_data, sliced_labels, sliced_raw_labels = defaultdict(dict), defaultdict(
            dict), defaultdict(dict), defaultdict(dict)

        for _set in data.keys():
            # Slice the dataset data, and get the indexes masks between each slice to its samples
            set_sliced_data, sliced_indexes = self.slice_dataframe(data[_set])
            # Populate new data & labels dicts using the mask
            for _slice in self.slices:
                sliced_data[_slice][_set] = set_sliced_data[_slice]
                sliced_ts_data[_slice][_set] = ts_data[_set][sliced_indexes[_slice]]
                sliced_labels[_slice][_set] = labels[_set][sliced_indexes[_slice]]
                sliced_raw_labels[_slice][_set] = raw_labels[_set][sliced_indexes[_slice]]
        return sliced_data, sliced_ts_data, sliced_labels, sliced_raw_labels

    def load_model(self, from_path: Optional[str] = None):
        _path = from_path if from_path else self.model_path
        print(f'Loading SplitCareerPhaseModel from {_path}...')
        with open(_path, 'rb') as f:
            pkl = pickle.load(f)
        self.models = pkl['models']
        self.slices = pkl['slices']
        self.model_args = pkl['args']
        self.splits_config = pkl['split_config']

    def train(self, dataset: StatsDataset, use_existing: bool = True):
        print(f'\nTraining {self.name}')
        data, ts_data, labels, raw_labels = self.data_split(dataset)
        if use_existing and os.path.exists(self.model_path):
            self.load_model()
        else:
            self.init()
            sliced_data, sliced_ts_data, sliced_labels, sliced_raw_labels = self.slice_stats_dataset(
                data, ts_data, labels, raw_labels
            )
            print(f'\nFitting across {len(self.slices)} population slices:', self.slices)
            print()
            for s in tqdm(self.slices):
                if len(sliced_data[s]['train']) == 0:
                    print(f"slice {s} has zero members > skip")
                    continue
                self.models[s].fit(sliced_data[s], sliced_ts_data[s], sliced_labels[s], sliced_raw_labels[s],
                                   export_model=False)
            with open(self.model_path, 'wb') as f:
                pickle.dump({'models': self.models,
                             'slices': self.slices,
                             'args': self.model_args,
                             'split_config': self.splits_config
                             }, f)
        return self.evaluate(data['test'], ts_data['test'], labels['test'], raw_labels['test'], dataset)

    def reorder_predictions(self, df, slices, predictions_dict):
        """
        Reorders the predictions to match the original order of the DataFrame.

        Args:
        df (pd.DataFrame): The original DataFrame.
        slices (dict): Dictionary of sliced DataFrames, keys are slice names.
        predictions_dict (dict): Dictionary where each key is a slice name and the value is another dict
                                 {prediction_name: numpy array of predictions for that slice}.

        Returns:
        dict: Final predictions where each numpy array is ordered according to the original DataFrame.
        """
        pass

    def predict(self, x: pd.DataFrame, ts_data: Optional[np.array] = None) -> Dict:
        """
        Input: dataframe of players stats over years.
            - Mandatory columns: COLUMNS.YEAR, COLUMNS.PLAYER_ID, self.clustering_attr
        Return: Dict
            'clusters': np.array of cluster associated with each prediction item
            'cls_preds': np.array of KNN based prediction associated with each prediction item
            'cls_probs': np.array of KNN based probability associated with each prediction item
            'cluster_prob': np.array of cluster members label prior, associated with each prediction item
            'cluster_yoy_dist': distribution of the cluster member yoy diff, associated with each prediction item
            'ts_confidence': % of years the player had / num years back model is using
        """
        ts_coverage = self.calc_ts_coverage(x)
        assert len(ts_coverage) == len(x)
        if ts_data is None:
            x, valid_indices, ts_data = self.preprocess_data(x.copy(), handle_missing_values='impute')

        # Initialize final predictions dictionary with empty lists
        predictions = {
            'slices': [None] * len(x),
            'clusters': [None] * len(x),
            'cls_preds': [None] * len(x),
            'cls_probs': [None] * len(x),
            'cluster_prob': [None] * len(x),
            'cluster_yoy_dist': [None] * len(x),
            'ts_confidence': ts_coverage
        }

        x['orig_index'] = x.index.copy()
        x_sliced, sliced_indexes = self.slice_dataframe(x)

        # Create a mapping from original index to the proper (slice name, ordered position in slice)
        # e.g., original index 9876 -> ('age::15->20&overall::0.0->0.64', 2),
        #   meaning index 9876 (loc) is number 3 (starting at 0) in the slice (iloc)
        # Assumption - each player ts belongs to only 1 slice
        index2slice: Dict[int, Tuple] = {}
        for slice_name, slice_df in x_sliced.items():
            for i, orig_idx in enumerate(slice_df['orig_index']):
                index2slice[orig_idx] = (slice_name, i)

        # Look for uncovered indices (not in all slices)
        uncovered_indices = set(x['orig_index'].tolist()) - set(list(index2slice.keys()))
        for ind in uncovered_indices:
            index2slice[ind] = (None, None)

        # Run predictions by slice
        sliced_predictions = {}
        for _slice, _x in x_sliced.items():
            if len(_x) == 0:
                continue
            elif self.models[_slice].model is None:
                # Empty in training
                sliced_predictions[_slice] = self.na_predictions(x_sliced[_slice])
            else:
                sliced_predictions[_slice] = self.models[_slice].predict(x_sliced[_slice],
                                                                         ts_data[sliced_indexes[_slice]])

        # Iterate over original indices and fill predictions in correct order
        # Get predictions using original index, write results using enumerator i (ordered)
        for i, orig_idx in enumerate(x['orig_index']):
            sample_slice, sample_idx_in_slice = index2slice[orig_idx]
            predictions['slices'][i] = sample_slice
            # If this sample is covered
            if not pd.isna(sample_slice):
                for pred_key, pred_array in sliced_predictions[sample_slice].items():
                    # Skip confidence - calculated once at the first step
                    if pred_key == 'ts_confidence':
                        continue
                    predictions[pred_key][i] = pred_array[sample_idx_in_slice]
            else:
                for key in predictions:
                    if key == 'cluster_yoy_dist':
                        predictions[key][i] = pd.Series({k: None for k in self.model_args.yoy_dist_features})
                    else:
                        predictions[key][i] = None

        # Convert lists to numpy arrays
        for key in predictions:
            # Skip confidence - calculated once at the first step
            if key == 'ts_confidence':
                continue
            if isinstance(next(iter(predictions[key])), pd.Series):
                predictions[key] = pd.concat(predictions[key], axis=1).T.reset_index(drop=True) \
                    .fillna(self.model_args.series_na_value)
            else:
                if key == 'slices':
                    predictions[key] = pd.Series(predictions[key]).fillna(self.model_args.series_na_value) \
                        .replace(self.model_args.series_na_value, int(self.model_args.series_na_value))
                elif key == 'clusters':
                    predictions[key] = pd.Series(predictions[key]).fillna(self.model_args.series_na_value) \
                        .replace(self.model_args.series_na_value, int(self.model_args.series_na_value))
                else:
                    predictions[key] = pd.Series(predictions[key]).fillna(self.model_args.series_na_value).to_numpy()
                predictions[key] = np.array(predictions[key])
                if pd.Series(predictions[key]).isna().sum() > 0 and key == 'clusters':
                    warnings.warn(f"SplitCareerPhaseModel > Found {pd.Series(predictions[key]).isna().sum()} NA values")
        return predictions

    def na_predictions(self, x: pd.DataFrame) -> Dict:
        """
         Return: Dict
            'clusters': np.array of cluster associated with each prediction item
            'cls_preds': np.array of KNN based prediction associated with each prediction item
            'cls_probs': np.array of KNN based probability associated with each prediction item
            'cluster_prob': np.array of cluster members label prior, associated with each prediction item
            'cluster_yoy_dist': distribution of the cluster member yoy diff, associated with each prediction item
            'ts_confidence': % of years the player had / num years back model is using
        """
        return {
            'clusters': [None] * len(x),
            'cls_preds': [None] * len(x),
            'cls_probs': [None] * len(x),
            'cluster_prob': [None] * len(x),
            'cluster_yoy_dist': [pd.Series({key: None for key in self.model_args.yoy_dist_features})] * len(x),
            'ts_confidence': [None] * len(x)
        }
