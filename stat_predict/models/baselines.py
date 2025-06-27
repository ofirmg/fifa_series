from typing import Dict, Iterable
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import scipy
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from stat_predict.models.model import ModelArgs, MLStatPredict
from stat_predict.static.config import DefaultParameters


class LRArgs(ModelArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_params = {'C': [0.01, 1, 10, 20, 50, 100],
                            'solver': ['newton-cholesky', 'liblinear', 'lbfgs', 'sag', 'saga'],
                            'max_iter': [100, 250, 500]
                            }
        self.args = dict(C=kwargs.get('C', 1.),
                         solver=kwargs.get('solver', 'lbfgs'),
                         max_iter=kwargs.get('max_iter', 500),
                         penalty=kwargs.get('penalty', 'l2'),
                         class_weight=kwargs.get('classes_weights', [0.5, 2.])
                         )
        self.numeric_features_only = True


class XGBArgs(ModelArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_params = kwargs.get('grid_params', {
            'min_child_weight': [1, 5, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5, 6, 8],
            'iterations': [100, 500, 1000]
        })
        self.param_dist = {'n_estimators': scipy.stats.randint(150, 1000),
                           'subsample': scipy.stats.uniform(0.5, 0.8),
                           'max_depth': [3, 4, 5, 6, 7, 8, 9, 11],
                           'colsample_bytree': scipy.stats.uniform(0.4, 0.8),
                           'min_child_weight': [1, 5, 10],
                           'gamma': [0.5, 1, 1.5, 2, 5],
                           }

        self.args = dict(eta=kwargs.get('eta', 0.3),
                         reg_lambda=kwargs.get('reg_lambda', 1.),
                         max_depth=kwargs.get('max_depth', 6),
                         objective='binary:logistic',
                         iterations=kwargs.get('iterations', 500),
                         scoring=kwargs.get('scoring', 'roc_auc'),
                         tree_method='hist',
                         )
        self.training_args = dict(
            early_stopping=kwargs.get('early_stopping', DefaultParameters.early_stop),
            class_weights=kwargs.get('classes_weights', list(DefaultParameters.classes_weights)),
            random_search_iter=kwargs.get('random_search_iter', 5)
        )
        self.numeric_features_only = True


class CatboostArgs(ModelArgs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_params = kwargs.get('grid_params', {'depth': [3, 4, 5, 6, 10],
                                                      'iterations': [150, 175, 250],
                                                      'auto_class_weights': ['Balanced'],
                                                      'rsm': [0.3, 0.6, 0.8, 1.0],
                                                      'grow_policy': ['Depthwise'],
                                                      'min_child_samples': [3, 5, 9, 11, 13, 15]
                                                      })
        self.args = dict(learning_rate=kwargs.get('learning_rate', 0.3),
                         reg_lambda=kwargs.get('reg_lambda', 1.),
                         depth=kwargs.get('depth', 6),
                         loss_function=kwargs.get('loss', 'Logloss'),
                         iterations=kwargs.get('iterations', 500),
                         )
        self.training_args = dict(
            early_stopping=kwargs.get('early_stopping', DefaultParameters.early_stop),
            classes_weights=kwargs.get('classes_weights', DefaultParameters.classes_weights)
        )


class CatBoostStatPredict(MLStatPredict):

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return f'CatBoost_{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}'

    def _fit(self,
             x_train: pd.DataFrame,
             y_train: np.array,
             x_val: pd.DataFrame,
             y_val: np.array,
             params: Dict = None):
        if params is None:
            params = self.model_args.args
        # x, y = self.merge_sets(x_train, y_train, x_val, y_val)
        categorical_features = list(x_train.select_dtypes('category').columns)
        train_pool = Pool(x_train, y_train, cat_features=categorical_features)
        self.init_model(params=params)
        self.model.fit(train_pool, eval_set=(x_val, y_val), verbose=False)

    def init_model(self, params: Dict = None, **kwargs):
        if params is None:
            params = self.model_args.args
        self.model = CatBoostClassifier(**params, logging_level='Silent')

    def _grid_search(self, x_train: pd.DataFrame, y_train: np.array, x_val: pd.DataFrame, y_val: np.array):
        self.init_model()
        print('-' * 200)
        if self.model_args.grid_params is not None:
            print(f' - performing {self.name} grid search...')
            # Grid data
            categorical_features = list(x_train.select_dtypes('category').columns)
            x, y = self.merge_sets(x_train, y_train, x_val, y_val)
            train_pool = Pool(x, y, cat_features=categorical_features)

            self.init_model()
            grid_search_result = self.model.grid_search(
                self.model_args.grid_params,
                train_pool,
                cv=5,
                verbose=False,

            )
            grid_search_result = grid_search_result['params']
            print(f'Catboost best params: {grid_search_result}')
        else:
            grid_search_result = self.model_args
        return grid_search_result


class LRStatPredict(MLStatPredict):

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return f'LogisticRegression_{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}'

    def _feature_importance(self):
        return [abs(v) for v in self.model.coef_[0]]

    def init_model(self, params: Dict = None, **kwargs):
        if params is None:
            params = self.model_args.args
        self.model = LogisticRegression(**params)

    def _grid_search(self, x_train: pd.DataFrame, y_train, x_val: pd.DataFrame, y_val: np.array) -> Dict:
        self.init_model()
        print(f' - performing {self.name} grid search')
        lr_grid = GridSearchCV(LogisticRegression(), self.model_args.grid_params)
        x, y = self.merge_sets(x_train, y_train, x_val, y_val)

        lr_grid.fit(x, y)
        best_params = lr_grid.best_params_
        best_score = lr_grid.best_score_
        print(f'\nLR best score: {best_score}')
        print(f'LR best params: {best_params}')
        return best_params


class XGBStatPredict(MLStatPredict):

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return f'XGBoost_{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}'

    def init_model(self, params: Dict = None, **kwargs):
        if params is None:
            params = self.model_args.args
        self.model = XGBClassifier(**params)

    def _fit(self,
             x_train: pd.DataFrame,
             y_train: np.array,
             x_val: pd.DataFrame,
             y_val: np.array,
             params: Dict = None):
        if params is None:
            params = self.model_args.args
        self.init_model(params=params)
        self.model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    def _grid_search(self, x_train: pd.DataFrame, y_train, x_val: pd.DataFrame, y_val: np.array) -> Dict:
        self.init_model()
        print(f' - performing {self.name} grid search...')
        x, y = self.merge_sets(x_train, y_train, x_val, y_val)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
        random_search = RandomizedSearchCV(self.model,
                                           param_distributions=self.model_args.param_dist,
                                           n_iter=self.model_args.training_args['random_search_iter'],
                                           scoring='roc_auc',
                                           n_jobs=-1,
                                           cv=skf.split(x, y),
                                           verbose=False,
                                           random_state=1001
                                           )
        random_search.fit(x, y)
        print(f'\nXGB best score: {random_search.best_score_}')
        print(f'XGB best params: {random_search.best_params_}')
        return random_search.best_params_


class TabNetArgs(ModelArgs):
    def __init__(self, **kwargs):
        filter_feature_families = kwargs.get('filter_feature_families', DefaultParameters.filter_feature_families)
        super().__init__(filter_feature_families=filter_feature_families,
                         correlation_threshold=1.0,
                         selection_correlation_method='pearson',
                         num_features_to_select=150,
                         **kwargs)
        self.grid_params = {}
        self.numeric_features_only = True
        self.training_args = dict(
            eval_metric=[kwargs.get('metric', 'logloss')],  # auc
            max_epochs=DefaultParameters.num_epochs,
            patience=DefaultParameters.early_stop,
            batch_size=128,
            virtual_batch_size=128,
            weights=1
        )


class TabNetStatPredict(MLStatPredict):

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return f'TabNet_{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}'

    def init_model(self, params: Dict = None, **kwargs):
        if params is None:
            params = self.model_args.args
        self.model = TabNetClassifier(**params)

    def _fit(self,
             x_train: pd.DataFrame,
             y_train: np.array,
             x_val: pd.DataFrame,
             y_val: np.array,
             params: Dict = None):
        # Convert to numpy arrays if needed
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        self.init_model()
        print('Fitting TabNet...')
        self.model.fit(
            X_train=x_train,
            y_train=y_train,
            eval_set=[(x_val, y_val)],
            eval_name=['val'],
            drop_last=False,
            **self.model_args.training_args
        )

    def predict_proba(self, x: pd.DataFrame) -> Iterable[float]:
        x = np.array(x[self.model_features])
        probs = self.model.predict_proba(x)
        return probs[:, 1]

    def predict(self, x: pd.DataFrame) -> Iterable[int]:
        """
        Predicts using a trained TabNet model.

        For classification, returns class probabilities and predicted classes.
        For regression, returns predicted values.
        """
        x = np.array(x[self.model_features])
        probs = self.model.predict_proba(x)
        preds = probs.argmax(axis=1)
        return preds
