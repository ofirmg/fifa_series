import os
import pickle
from collections import defaultdict
from typing import List, Dict, Literal, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, EFstrType
from plotly import express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

from stat_predict.dataset.utils import get_feature_family, RAW_FAMILIES
from stat_predict.static.utils import SPECIFIC_POS_COLS
from stat_predict.dataset.sp_dataset import StatsDataset, DatasetArgs
from stat_predict.eval.evaluation import model_evaluation
from stat_predict.eval.utils import get_model_short_name
from stat_predict.models.career_phase import CareerPhaseModel, SplitCareerPhaseModel
from stat_predict.static.config import TEST_YEAR, ArtifactsPaths, DefaultParameters
from stat_predict.static.utils import COLUMNS


class ModelArgs:
    def __init__(self, **kwargs):
        self.random_seed = 2603
        self.model_class_name: str = kwargs.get('model_class_name', '')

        self.grid_params = kwargs.get('grid_params', {})
        self.args = kwargs.get('args', {})
        self.training_args = kwargs.get('training_args', {})
        self.output_dir = kwargs.get('output_dir', ArtifactsPaths.ARTIFACTS_DIR)
        self.na_filler: float = kwargs.get('na_filler', DatasetArgs.na_filler)
        self.apply_feature_selection = kwargs.get('apply_feature_selection', True)
        self.num_features_to_select = kwargs.get('num_features_to_select', DefaultParameters.num_features_to_select)
        self.selection_correlation_method: Literal['pearson', 'kendall', 'spearman'] = kwargs.get(
            'selection_correlation_method', DefaultParameters.selection_correlation_method)
        self.correlation_threshold: float = kwargs.get('correlation_threshold', DefaultParameters.correlation_threshold)
        self.feature_selection_steps: int = kwargs.get('feature_selection_steps',
                                                       DefaultParameters.feature_selection_steps)
        self.filter_feature_families: Optional[List[str]] = kwargs.get('filter_feature_families', None)
        self.numeric_features_only: bool = kwargs.get('numeric_features_only', False)
        self.career_phase_model_path: Optional[str] = kwargs.get('career_phase_model_path')
        self.show_plots: bool = kwargs.get('show_plots', True)
        self.export_plots: bool = kwargs.get('export_plots', True)


class MLStatPredict(object):
    def __init__(self, model_args: ModelArgs, num_years_back: int = None, **kwargs):
        self.compute_features_importance = kwargs.get('compute_features_importance', True)
        self.experiment_suffix = kwargs.get('', None)
        self.model_args = model_args
        self.num_years_back = num_years_back
        self.min_years_back = kwargs.get('min_years_back', num_years_back - 1)
        self.test_year: int = kwargs.get('test_year', TEST_YEAR)
        self.verbose = kwargs.get('verbose', True)
        self.model = None
        self.model_features: Optional[List[str]] = None
        self.career_phase_model: Optional[CareerPhaseModel] = None
        self.load_career_model()
        self.use_existing: bool = kwargs.get('use_existing', False)
        self.name = self.get_model_name(kwargs)

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return f'MLStatPredict_{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}'

    @property
    def model_suffix(self) -> str:
        suffix = ''
        if self.model_args.filter_feature_families is not None:
            for fam in self.model_args.filter_feature_families:
                suffix += f'-{fam[:3]}'
        suffix += f"_nfeatures={self.model_args.num_features_to_select}"
        suffix += f"_corr={self.model_args.selection_correlation_method}"
        suffix += f"-{round(self.model_args.correlation_threshold, 2)}"
        if self.career_phase_model is not None:
            phase_str = self.model_args.career_phase_model_path.split('-by-')[1].split('.pickle')[0] \
                .replace('_series-raw', '')
            suffix += f"_SplitPhaseBy-{phase_str}"
        else:
            suffix += '_wo_phase_model'
        return suffix

    @property
    def model_path(self) -> str:
        return str(os.path.join(self.model_args.output_dir, 'models', f'{self.name}.pickle'))

    @property
    def reports_output_dir(self) -> str:
        reports_output_dir = os.path.join(self.model_args.output_dir, 'reports', self.name)
        if not os.path.exists(reports_output_dir):
            os.mkdir(reports_output_dir)
        return reports_output_dir

    @property
    def features_path(self) -> str:
        return str(os.path.join(self.model_args.output_dir, 'features', f'selected_features_{self.name}.pickle'))

    @staticmethod
    def merge_sets(x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame, y_val: np.array):
        """ Util static function to combine train and validation when needed """
        x = pd.concat([x_train, x_val], axis=0)
        y = np.hstack([y_train, y_val])
        return x, y

    def load_career_model(self, from_path: Optional[str] = None):
        if self.model_args.career_phase_model_path is None:
            return
        if from_path is None:
            from_path = os.path.join(ArtifactsPaths.CAREER_PHASE_DIR, self.model_args.career_phase_model_path)
        # Understand phase model class
        if self.model_args.career_phase_model_path.lower().startswith('split'):
            self.career_phase_model = SplitCareerPhaseModel()
        else:
            self.career_phase_model = CareerPhaseModel()
        self.career_phase_model.load_model(from_path=from_path)


    def remove_correlated_features(self,
                                   _df: pd.DataFrame,
                                   y_true: np.array,
                                   max_features_remove: int | float = 0.6,
                                   step_size: int = 0.005,
                                   seed: int = 26031991
                                   ) -> List:
        """
        Iteratively remove correlated features from a DataFrame.

        Parameters:
        df (pd.DataFrame): The input dataframe with features.
        y_true (np.array): The target array for mutual information calculation.
        minimal_threshold (float): The desired minimal correlation threshold.
        correlation_method (str): Method for correlation calculation ('pearson', 'spearman', 'kendall').
        max_features_remove (int\float, optional): Maximum number/frac of features to remove. Defaults to None.

        Returns: list of removed features
        """
        np.random.seed(seed)

        # Validate input
        if self.model_args.selection_correlation_method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("Invalid correlation method. Choose 'pearson', 'spearman', or 'kendall'.")

        df = _df.copy()
        if max_features_remove <= 1:
            max_features_remove = df.shape[1] * max_features_remove
        max_features_remove = int(min(df.shape[1], max_features_remove))

        # Compute initial correlation matrix and mutual information scores
        corr_matrix = df.corr(method=self.model_args.selection_correlation_method)
        if y_true is not None:
            mi_scores = mutual_info_classif(df, y_true, discrete_features=False)
            mi_scores = pd.Series(mi_scores, index=df.columns)
        else:
            mi_scores = [1] * len(y_true)

        # Keep track of removed features
        removed_features = set()
        removes_logs = []
        # Iterate until correlation threshold is met or max features removed
        print(f'\nremove_correlated_features...')
        starting_point = len(df.columns)
        print(f'- starting point = {starting_point} features (max to remove = {max_features_remove})')
        num_steps = max(1, int((1 - self.model_args.correlation_threshold) / step_size))
        for iter_i in tqdm(range(num_steps)):
            features = list(df.columns)

            # Find highly correlated feature pairs
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = {(row, col): upper_triangle.iloc[row, col]
                               for row, col in
                               zip(*np.where(upper_triangle >= self.model_args.correlation_threshold))}
            corr_pair_iter = [k for k in high_corr_pairs.items()]
            corr_pair_iter = sorted(corr_pair_iter, key=lambda x: x[1], reverse=True)
            if len(corr_pair_iter) == 0:
                print(f'\n - END remove corr features: no candidates left '
                      f'(threshold={round(self.model_args.correlation_threshold, 3)})')
                break
            threshold = max([item[1] for item in corr_pair_iter])
            if len(removed_features) >= max_features_remove:
                print(f'\n - END remove corr features: '
                      f'{len(removed_features)} features removed LIMIT (=threshold={round(threshold, 3)})')
                break
            elif threshold < self.model_args.correlation_threshold:
                print(f'\n - END remove corr features: reached prob threshold')
                break

            # Select features to remove
            features_to_remove = set()
            for (i, j), corr_value in corr_pair_iter:
                col1, col2 = features[i], features[j]
                # Check if features were already removed
                if (col1 not in df.columns or col2 not in df.columns) or (
                        col1 in features_to_remove or col2 in features_to_remove):
                    continue

                # Decide which feature to drop based on MI scores or at random
                col1_family, col2_family = get_feature_family(col1), get_feature_family(col2)
                if col1_family in RAW_FAMILIES and col2_family not in RAW_FAMILIES:
                    to_drop = col2
                elif col1_family not in RAW_FAMILIES and col2_family in RAW_FAMILIES:
                    to_drop = col1
                elif corr_value == 1.0:
                    to_drop = np.random.choice([col1, col2])
                else:
                    to_drop = col1 if mi_scores[col1] < mi_scores[col2] else col2

                features_to_remove.add(to_drop)
                removes_logs.append({'col1': col1, 'col2': col2, 'corr': corr_value})

                # Break if max features removed
                if len(removed_features) >= max_features_remove:
                    print(f'\n - END remove corr features: REACHED max_features_remove LIMIT (={max_features_remove})')
                    break

            # Remove selected features, excluding phase_model features
            features_to_remove = [c for c in features_to_remove if 'phase_model:' not in c]
            df.drop(columns=list(features_to_remove), inplace=True)
            removed_features = removed_features.union(features_to_remove)

            # Update correlation matrix
            corr_matrix = df.corr(method=self.model_args.selection_correlation_method)
            threshold -= step_size

        removed_features = list(removed_features)
        pd.DataFrame(removes_logs).to_csv(os.path.join(self.reports_output_dir,
                                                       f'remove_corr_features_{self.num_years_back}yback.csv'))
        print(f'finished remove_correlated_features > removing ({len(removed_features)}) features '
              f'({df.shape[1]} remained)')
        return removed_features

    def apply_args_filtering(self, x: pd.DataFrame, columns_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Function gets a dataframe and its mapping between columns and types.
        The function applies feature filtering based on the feature family, type, etc.
        The function updates the model_features self attribute and returns updated version of columns_mapping
        :return: columns_mapping: Dict[str, List[str]] - mapping x's columns to types
        """
        self.model_features = [c for c in self.model_features if c.split('_row_')[0] not in SPECIFIC_POS_COLS]

        # Apply features filtering by feature family
        if self.model_args.filter_feature_families:
            print(f'Filtering features families: {self.model_args.filter_feature_families}')
            print(' - num before:', len(self.model_features))
            self.model_features = [c for c in self.model_features
                                   if get_feature_family(c) not in self.model_args.filter_feature_families]
            print(' - num left:', len(self.model_features))

        # Apply features filtering to numeric type if required
        if self.model_args.numeric_features_only:
            columns_mapping['categorical'] = []

            print(f'Filtering features to numeric features only')
            numeric_cols = []
            for col in x.columns:
                try:
                    # Try to convert to numeric
                    converted = pd.to_numeric(x[col], errors='coerce')
                    # If most values were successfully converted, consider it numeric
                    if converted.notna().sum() / len(x[col]) > 0.9:
                        numeric_cols.append(col)
                except:
                    continue

            self.model_features = [c for c in self.model_features
                                   if (c in x.select_dtypes(np.number).columns or c in numeric_cols)
                                   and '_id' not in c
                                   ]
            print(' - num left:', len(self.model_features))

        # Update columns_mapping
        if not self.model_args.numeric_features_only:
            columns_mapping['categorical'] = [c for c in columns_mapping['categorical'] if c in self.model_features]
        columns_mapping['numeric'] = [c for c in columns_mapping['numeric'] if c in self.model_features]
        return columns_mapping

    def feature_selection(self,
                          train: pd.DataFrame,
                          val: pd.DataFrame,
                          y_train: np.array,
                          y_val: np.array,
                          columns_mapping: Dict[str, List[str]],
                          use_existing: bool = True
                          ):
        """
        Perform feature selection, populates self.selected_features
        """
        if use_existing and os.path.exists(self.features_path):
            self.load_features()
            return

        print(f'\nInitial features distribution by family:\n'
              f'{pd.Series([get_feature_family(feat) for feat in train.columns]).value_counts()}\n')

        # Apply selection based on model args
        self.model_features = list(train.columns)
        columns_mapping = self.apply_args_filtering(train, columns_mapping)
        train = train[self.model_features]

        if not self.model_args.apply_feature_selection:
            print(f'\nSaving features to {self.features_path}...')
            with open(self.features_path, 'wb') as f:
                pickle.dump(self.model_features, f)
            print(f'\nFinal features distribution by family:\n'
                  f'{pd.Series([get_feature_family(feat) for feat in self.model_features]).value_counts()}')
            return

        print('- performing feature selection...')
        categorical_features = columns_mapping['categorical'][:] if not self.model_args.numeric_features_only else []
        numerical_features = columns_mapping['numeric'][:]
        self.model_features = list(set(categorical_features) | set(numerical_features))

        # REMOVE FEATURE WITH ZERO VARIANCE (ZERO DIFF FEATURES, ETC)
        zero_variance = list(np.array(numerical_features)[train[numerical_features].std(axis=0) == 0])
        print(f'  - Zero variance features ({len(zero_variance)}):', zero_variance)
        train = train.drop(columns=zero_variance)
        self.model_features = [c for c in self.model_features if c not in zero_variance]
        fixed_categorical = [c for c in categorical_features if len(train[c].unique()) == 1]
        print(f'  - Fixed categorical features ({len(fixed_categorical)}):', fixed_categorical)
        train = train.drop(columns=fixed_categorical)
        self.model_features = [c for c in self.model_features if c not in fixed_categorical]

        print(f'\nUpdated features distribution by family:\n'
              f'{pd.Series([get_feature_family(feat) for feat in train.columns]).value_counts()}\n')

        # remove_correlated_features
        numerical_features = [c for c in numerical_features if c in self.model_features]
        removed_features = self.remove_correlated_features(train[numerical_features], y_train)
        self.model_features = [c for c in self.model_features if c not in removed_features]
        print('\nremove_correlated_features removed features:')
        print(removed_features)

        print(f'\nUpdated features distribution by family:\n'
              f'{pd.Series([get_feature_family(feat) for feat in train.columns]).value_counts()}\n')

        # Feature selection
        feature_names_ixes = list(range(len(self.model_features)))
        train = train[self.model_features]
        val = val[self.model_features]

        # Use train and validation to select features
        train_pool = Pool(train, y_train, cat_features=categorical_features)
        test_pool = Pool(val, y_val, cat_features=categorical_features)

        print(' - calculating feature importance before selection...')
        model = CatBoostClassifier(iterations=1000, early_stopping_rounds=DefaultParameters.early_stop)
        model.fit(train_pool, verbose=False)
        feature_importance = model.get_feature_importance(
            type=EFstrType.FeatureImportance,
            thread_count=-1,
            verbose=False,
        )
        importance_df = pd.Series(feature_importance, index=self.model_features)
        importance_df.to_csv(os.path.join(self.reports_output_dir,
                                          f'feature_importance_before_selection_{self.num_years_back}yback.csv'))

        if self.model_args.num_features_to_select < len(feature_names_ixes):
            print(f' - CatBoost feature selection (n_select = {self.model_args.num_features_to_select}, '
                  f'out of {len(self.model_features)} features)...')
            selected_features_results = model.select_features(
                train_pool,
                eval_set=test_pool,
                features_for_select=feature_names_ixes,
                num_features_to_select=self.model_args.num_features_to_select,
                steps=self.model_args.feature_selection_steps,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model=False,
                logging_level='Silent',
                plot=self.model_args.show_plots
            )
            # Selected features
            selected_features_names = selected_features_results['selected_features_names']
        else:
            print(f' - CatBoost NOT NEEDED feature selection (num to select={self.model_args.num_features_to_select},'
                  f' num features={len(feature_names_ixes)}...')
            selected_features_names = self.model_features[:]

        print(' - calculating feature importance after selection...')
        model = CatBoostClassifier(iterations=1000, early_stopping_rounds=DefaultParameters.early_stop)
        train_pool = Pool(
            train[selected_features_names],
            y_train,
            cat_features=[c for c in categorical_features if c in selected_features_names]
        )
        model.fit(train_pool, verbose=False)
        feature_importance = model.get_feature_importance(
            type=EFstrType.FeatureImportance,
            thread_count=-1,
            verbose=False,
        )
        importance_df = pd.Series(feature_importance, index=selected_features_names)
        importance_df.to_csv(os.path.join(self.reports_output_dir,
                                          f'feature_importance_after_selection_{self.num_years_back}yback.csv'))
        selected_features = [f for f in selected_features_names]
        print(f'\nFinal features distribution by family:\n'
              f'{pd.Series([get_feature_family(feat) for feat in selected_features]).value_counts()}')
        print(f'\nFinal features distribution by years back:\n'
              f'{pd.Series([int(feat.split('_row_')[1]) for feat in selected_features if '_row_' in feat]).value_counts()}')
        self.model_features = selected_features[:]

        print(f'\nSaving features to {self.features_path}...')
        with open(self.features_path, 'wb') as f:
            pickle.dump(self.model_features[:], f)

    def data_split(self,
                   dataset: StatsDataset
                   ) -> (pd.DataFrame, List, pd.DataFrame, List, pd.DataFrame, List):
        """
        Splits the data StatsDataset dataset by time
        :return: x_train, y_train, val, y_val, test, y_test
        """
        print('- Splitting dataset to train, validation, test...')
        categorical_features = dataset.columns_mapping['categorical'][:]
        numeric_features = dataset.columns_mapping['numeric'][:]

        def get_set_processed_x_and_y(split_name: str) -> Tuple[pd.DataFrame, np.array]:
            _df = dataset.data[split_name].reset_index(drop=True)
            _df['label'] = np.array([int(v['label']) for v in dataset.labels[split_name]])

            if self.min_years_back is not None:
                # Remove samples with too few rows in history
                print(f'Filtering samples with num years < {self.min_years_back}')
                print(f' - samples before: {len(_df)}')
                mask = (_df[[c for c in _df.columns if c.startswith('pad_row')]].sum(axis=1) <=
                        self.num_years_back - self.min_years_back).to_numpy()
                _df = _df[mask].reset_index(drop=True)
                dataset.data[split_name] = dataset.data[split_name][mask].reset_index(drop=True)
                dataset.labels[split_name] = [dataset.labels[split_name][i] for i in range(len(mask)) if mask[i]]
                print(f' - samples after: {len(_df)}')

            y = _df['label'].to_numpy()

            # Convert types, keep only numeric and categorical features
            _df[numeric_features] = _df[numeric_features].astype(float).fillna(self.model_args.na_filler)
            _df[categorical_features] = _df[categorical_features].replace(self.model_args.na_filler, '').fillna(
                '').astype('category')
            _df = _df[numeric_features + categorical_features]

            return _df, y

        x_train, y_train = get_set_processed_x_and_y('train')
        x_val, y_val = get_set_processed_x_and_y('validation')
        x_test, y_test = get_set_processed_x_and_y('test')

        print('\nDatasets split:')
        for set_name, set_obj in zip(['train', 'validation', 'test'], [x_train, x_val, x_test]):
            print(f" - {set_name} N={len(set_obj)} samples, "
                  f"{len(dataset.data[set_name]['long_name_row_0'].unique())} players")

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _grid_search(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame, y_val: np.array) -> Dict:
        return {}

    def _fit(self,
             x_train: pd.DataFrame,
             y_train: np.array,
             x_val: pd.DataFrame,
             y_val: np.array,
             params: Dict = None):
        x, y = self.merge_sets(x_train, y_train, x_val, y_val)
        x = x.select_dtypes(np.number)
        self.init_model(params=params)
        self.model.fit(x, y)

    def init_model(self, params: Dict = None, **kwargs):
        pass

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        print(f'\nLoading {self.name} from {self.model_path}...')
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.features_path, 'rb') as f:
            self.model_features = pickle.load(f)

    def load_features(self):
        print(f'\nLoading features from {self.features_path}...')
        with open(self.features_path, 'rb') as f:
            self.model_features = pickle.load(f)

    def predict(self, x: pd.DataFrame) -> Iterable[int]:
        """
        Note - this is 'predict' in the context of the experiment (StatPredict dataset).
        It cannot be used on raw data.
        """
        return self.model.predict(x[self.model_features])

    def predict_proba(self, x: pd.DataFrame) -> Iterable[float]:
        """
            Note - this is 'predict_proba' in the context of the experiment (StatPredict dataset).
            It cannot be used on raw data.
        """
        return self.model.predict_proba(x[self.model_features])[:, 1]

    def fit(self, dataset: StatsDataset, x_train, y_train, x_val, y_val, x_test, y_test) -> (Dict, pd.DataFrame):
        # Grid search for fit parameters
        best_params = self._grid_search(x_train, y_train, x_val, y_val)

        # Use all data
        print(f'\nFit final {self.name} (train + validation)...')
        _params = self.model_args.args
        _params.update(best_params)
        self._fit(x_train, y_train, x_val, y_val, params=best_params)

        # Export final model
        print(f'\nExporting {self.name} to {self.model_path}...')
        self.save_model()
        return self.evaluate(x_test, y_test, dataset)

    def train(self, dataset: StatsDataset, use_existing: bool = True) -> (Dict, pd.DataFrame):
        print('-' * 250, f'\n{self.name}', f'\n{'-' * 250}')
        # Split data into train-val-test
        x_train, y_train, x_val, y_val, x_test, y_test = self.data_split(dataset)
        x_train, x_val, x_test = self.add_phase_model_features(dataset, x_train, x_val, x_test)

        if use_existing and os.path.exists(self.model_path):
            self.load_model()
            print(self.model)
            return self.evaluate(x_test[self.model_features], y_test, dataset)

        # Feature selection - populates self.model_features
        self.feature_selection(x_train, x_val, y_train, y_val, dataset.columns_mapping, use_existing=use_existing)
        if hasattr(self, 'scaler'):
            x_train = pd.DataFrame(self.scaler.fit_transform(x_train[self.model_features]), columns=self.model_features)
            x_val = pd.DataFrame(self.scaler.transform(x_val[self.model_features]), columns=self.model_features)
            x_test = pd.DataFrame(self.scaler.transform(x_test[self.model_features]), columns=self.model_features)
        return self.fit(dataset,
                        x_train[self.model_features], y_train,
                        x_val[self.model_features], y_val,
                        x_test[self.model_features], y_test
                        )

    def evaluate(self, test: pd.DataFrame, y_test: np.array, dataset: StatsDataset) -> (Dict, pd.DataFrame):
        print(f'\nevaluating {self.name}...')

        if self.model_args.show_plots:
            # Show centroids of slices
            for i in range(len(self.career_phase_model.models)):
                list(self.career_phase_model.models.values())[i].plot_centroids()

        # Collect attributes for analysis
        test_players_ratings = np.array([v['raw_label'] for v in dataset.labels['test']])
        test_ratings_y1_back = dataset.data['test']['overall_row_0'].to_numpy()
        test_ratings_yoy_diffs = test_players_ratings - test_ratings_y1_back  # = label rating - latest rating
        test_players_names = dataset.data['test'][f'{COLUMNS.PLAYER_FULL_NAME}_row_0'].tolist()
        test_players_ages = dataset.data['test'][f'{COLUMNS.AGE}_row_0'].tolist()
        num_years_data = dataset.data['test'][[c for c in dataset.data['test'].columns if c.startswith('pad_')]].sum(
            axis=1).apply(lambda x: self.num_years_back - x).tolist()

        # Evaluate
        predictions = self.predict(test)
        probs = self.predict_proba(test)
        report = model_evaluation(
            y_test,
            list(predictions),
            list(probs),
            rating_labels=list(test_players_ratings),
            ratings_yoy_diffs=test_ratings_yoy_diffs,
            players_ratings=test_ratings_y1_back,
            players_names=test_players_names,
            players_age=test_players_ages,
            output_dir=self.reports_output_dir,
            model_name=self.name,
            num_years_data=num_years_data,
            show_plots=self.model_args.show_plots,
            export_plots=self.model_args.export_plots,
        )

        # feature importance
        if self.compute_features_importance:
            self.report_features_importance(list(test.columns))

        test_pred_df = pd.DataFrame(dict(
            predictions=list(predictions),
            labels=y_test,
            probs=list(probs),
            rating_labels=list(test_players_ratings),
            players_ids=np.array([v[COLUMNS.PLAYER_ID] for v in dataset.labels['test']]),
            ratings_yoy_diffs=test_ratings_yoy_diffs,
            ratings_y1_back=test_ratings_y1_back,
            players_names=test_players_names,
            players_age=test_players_ages)
        )
        return report, test_pred_df

    def _feature_importance(self):
        return self.model.feature_importances_

    def report_features_importance(self, features_names: List[str]):
        """
        The method uses the '_feature_importance' method and analyze model feature importance across years and families.
        """
        print('\nfeature importance')
        importance_df = pd.DataFrame({'feature': features_names, 'importance': self._feature_importance()})
        print(' - top 50')
        print(importance_df.sort_values(by='importance', ascending=False).head(50))

        # By family
        features_families_values = []
        columns_family_mapping = defaultdict(list)

        for f in importance_df['feature']:
            feature_family = get_feature_family(f)
            columns_family_mapping[feature_family].append(f)
            features_families_values.append(feature_family)
        importance_df['feature_family'] = features_families_values
        importance_df['feature_row_family'] = [f.split('_row_')[1] if '_row' in f else '-1' for f in features_names]

        # Plot importance
        importance_df_by_fam = importance_df[['feature_family', 'importance']].groupby('feature_family').agg(
            ['mean', 'sum', 'count']).reset_index()
        importance_df_by_fam.columns = ['feature_family', 'avg_importance_per_feature', 'total_family_importance',
                                        'num_features']
        importance_fig = go.Figure()
        importance_fig.add_trace(go.Bar(
            x=importance_df_by_fam['feature_family'],
            y=importance_df_by_fam['total_family_importance'],
            name='Total importance'
        ))
        importance_fig.add_trace(go.Bar(
            x=importance_df_by_fam['feature_family'],
            y=importance_df_by_fam['avg_importance_per_feature'],
            name='Average importance'
        ))
        importance_fig.add_trace(go.Scatter(x=importance_df_by_fam['feature_family'],
                                            y=importance_df_by_fam['num_features'],
                                            marker=dict(color='grey', size=10),
                                            mode='markers',
                                            name='Number of features'
                                            ))
        short_model_name = get_model_short_name(self.name)
        importance_fig.update_layout(template="plotly_white",
                                     barmode='group',
                                     barcornerradius=10,
                                     xaxis_title='Feature family',
                                     yaxis_title='Total importance',
                                     title=f'Importance by Feature Family: {short_model_name}')

        # Plot importance by row
        importance_df_by_row = importance_df[['feature_row_family', 'importance']].groupby('feature_row_family').agg(
            ['mean', 'sum', 'count']).reset_index()
        importance_df_by_row.columns = ['feature_row_family', 'avg_importance_per_feature',
                                        'total_row_importance', 'num_features']
        n_years = self.num_years_back  # importance_df['feature_row_family'].nunique()
        years_names = {'0': 'pred_year', '-1': 'Phase model'}
        for y in range(1, n_years):
            years_names[str(y)] = f'pred_year-{y}'
        importance_df_by_row['feature_row_family'] = importance_df_by_row.feature_row_family.apply(
            lambda x: years_names[x])
        rows_importance_fig = go.Figure()
        rows_importance_fig.add_trace(go.Bar(
            x=importance_df_by_row['feature_row_family'],
            y=importance_df_by_row['total_row_importance'],
            name='Total importance'
        ))
        rows_importance_fig.add_trace(go.Scatter(x=importance_df_by_row['feature_row_family'],
                                                 y=importance_df_by_row['num_features'],
                                                 marker=dict(color='grey', size=10),
                                                 mode='markers',
                                                 name='Number of features'
                                                 ))
        rows_importance_fig.update_layout(template="plotly_white",
                                          barmode='group',
                                          barcornerradius=15,
                                          bargap=0.15,
                                          bargroupgap=0.1,
                                          title=f'Importance by Year: {short_model_name}',
                                          xaxis_title='Features year',
                                          yaxis_title='Total importance',
                                          xaxis={'categoryorder': 'array',
                                                 'categoryarray': [years_names[str(y)] for y in range(n_years)]})

        # Plot top features
        importance_df = importance_df.sort_values(by="importance", ascending=False).head(20).round(3)
        top_features_fig = px.bar(importance_df.sort_values(by="importance", ascending=True),
                                  x="importance",
                                  y="feature",
                                  title=f"Top Features by Importance: {short_model_name}",
                                  orientation='h')
        top_features_fig.update_layout(barmode="group",
                                       barcornerradius=10,
                                       bargap=0.15,  # gap between bars of adjacent location coordinates.
                                       bargroupgap=0.1,  # gap between bars of the same location coordinate.)
                                       xaxis_title='Feature importance',
                                       yaxis_title='Features'
                                       )

        if self.model_args.show_plots:
            top_features_fig.show()
            importance_fig.show()
            rows_importance_fig.show()

        if self.model_args.export_plots:
            rows_importance_fig.write_html(os.path.join(self.reports_output_dir,
                                                        f'{short_model_name}_rows_importance_fig.html'))
            top_features_fig.write_html(os.path.join(self.reports_output_dir,
                                                        f'{short_model_name}_top_features_by_importance_fig.html'))
            importance_df.to_csv(os.path.join(self.reports_output_dir,
                                              f"{self.name}_{ArtifactsPaths.FEATURE_IMPORTANCE}"))
            importance_fig.write_html(
                os.path.join(self.reports_output_dir, f'{get_model_short_name(self.name)}_family_importance_fig.html'))

    def add_phase_model_features(self,
                                 dataset: StatsDataset,
                                 x_train: pd.DataFrame,
                                 x_val: pd.DataFrame,
                                 x_test: pd.DataFrame) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ This method adds the career phase model prediction to the feature space """
        print('add_phase_model_features...')
        new_attributes = {}
        if self.career_phase_model is not None:
            x_train, new_attributes = self.career_phase_model.add_career_phase_attributes(x_train)
            x_val, _ = self.career_phase_model.add_career_phase_attributes(x_val)
            x_test, _ = self.career_phase_model.add_career_phase_attributes(x_test)
        if len(new_attributes) > 0:
            dataset.columns_mapping['numeric'].extend(new_attributes.get('numeric', []))
            dataset.columns_mapping['categorical'].extend(new_attributes.get('categorical', []))
        return x_train, x_val, x_test
