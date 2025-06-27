import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict
import pandas as pd

from stat_predict.dataset.utils import missing_values_report, features_distribution_report, label_description
from stat_predict.dataset.label import FifaStatLabel
from stat_predict.dataset.sp_dataset import StatsDataset, DatasetArgs, process_all_years_df
from stat_predict.eval.evaluation import build_final_report
from stat_predict.models.model import MLStatPredict
from stat_predict.static.config import ArtifactsPaths, TEST_YEAR
from stat_predict.static.utils import COLUMNS


@dataclass
class ExperimentArgs:
    years_back: int
    db_path: str = ArtifactsPaths.STAT_DATASET
    output_dir: str = ArtifactsPaths.ARTIFACTS_DIR
    scaling: str = 'max'
    use_existing_models: bool = False
    test_year: int = TEST_YEAR
    seed = 26031991
    validation_frac: float = 0.8


@dataclass
class Experiment:
    label_creator: FifaStatLabel
    model: MLStatPredict
    experiment_args: ExperimentArgs
    dataset: StatsDataset = None

    @property
    def dataset_path(self) -> str:
        return str(os.path.join(self.experiment_args.output_dir,
                                self.experiment_args.db_path.replace('{NUM_YEARS_BACK}',
                                                                     str(self.experiment_args.years_back))))

    @property
    def name(self):
        return f'model={self.model.name}_y_back={self.model.min_years_back}-{self.model.num_years_back}'

    @property
    def raw_data_all_years_path(self) -> str:
        return str(os.path.join(ArtifactsPaths.DATA_DIR, ArtifactsPaths.ALL_YEARS_MERGED_DATA))

    @property
    def processed_data_all_years_path(self) -> str:
        return str(os.path.join(ArtifactsPaths.DATA_DIR,
                                ArtifactsPaths.PROC_DB_PATH.replace('.pickle',
                                                                    f'_scale={self.experiment_args.scaling}.pickle')))

    @staticmethod
    def data_describe(df: pd.DataFrame):
        print('#' * 200)
        print('Data describe')
        print(' - Num rows:', len(df))
        print(f' - Years range: {df[COLUMNS.YEAR].min()} - {df[COLUMNS.YEAR].max()}')
        print(' - Unique players:', len(df[COLUMNS.PLAYER_ID].unique()))
        print(' - Unique player - years combinations:', len(df))
        print(' - Player - years distribution (# rows per player in data):')
        print(pd.Series(
            df[[COLUMNS.PLAYER_ID, COLUMNS.YEAR]].groupby(COLUMNS.PLAYER_ID).count()[COLUMNS.YEAR]).value_counts())
        missing_values_report(df)
        features_distribution_report(df)

    def get_processed_dataset(self, dataset_args: DatasetArgs):
        # Prepare raw data & labels
        scaling = self.experiment_args.scaling
        print('-' * 100)
        print('Processing all years players stats...')
        processed_df, columns_mapping, scaling_values = process_all_years_df(
            self.experiment_args.test_year,
            raw_data_all_years_path=self.raw_data_all_years_path,
            scaling=scaling,
            na_filler=dataset_args.na_filler,
            processed_df_path=self.processed_data_all_years_path
        )
        print(' - done')
        processed_df, columns_mapping['label_columns'] = self.label_creator.label_encoding(processed_df)

        self.data_describe(processed_df)
        return processed_df, columns_mapping

    def get_fifa_stats_dataset(self) -> StatsDataset:
        if os.path.exists(self.dataset_path) and self.experiment_args.use_existing_models:
            with open(self.dataset_path, 'rb') as f:
                return pickle.load(f)

        dataset_args = DatasetArgs(self.experiment_args.years_back,
                                   output_dir=self.experiment_args.output_dir,
                                   test_year=self.experiment_args.test_year,
                                   validation_frac=self.experiment_args.validation_frac,
                                   dataset_path=self.experiment_args.db_path
                                   )
        dataset = StatsDataset(dataset_args)
        processed_df, columns_mapping = self.get_processed_dataset(dataset_args)
        t_start = time.time()
        print(f'build_time_series_dataset, years back: {self.experiment_args.years_back}...\n')
        dataset.build(processed_df, columns_mapping)
        t_end = time.time()
        print(f' - dataset.build > finished in {round((t_end - t_start) / 60, 2)} minutes')
        return dataset

    def init(self):
        random.seed(self.experiment_args.seed)
        self.model.num_years_back = self.experiment_args.years_back
        self.model.model_args.output_dir = self.experiment_args.output_dir

    def describe_dataset(self):
        print('\nYears labels distribution')
        all_data = []
        for _set in ['train', 'validation', 'test']:
            labels = pd.DataFrame(self.dataset.labels[_set])
            data_w_labels = pd.concat([self.dataset.data[_set], labels], axis=1)
            all_data.append(data_w_labels.copy())

        all_data = pd.concat(all_data)
        label_description(all_data)

    def run(self) -> (Dict, pd.DataFrame):
        self.dataset = self.get_fifa_stats_dataset()
        self.describe_dataset()
        self.init()
        report, test_pred_df = self.model.train(self.dataset, use_existing=self.experiment_args.use_existing_models)
        return build_final_report(report), test_pred_df
