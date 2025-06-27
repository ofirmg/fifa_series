import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Iterable
import numpy as np
import pandas as pd
from tqdm import tqdm

from stat_predict.dataset.features_utils import adds_categorical_prior_features, \
    add_categorical_yoy_features, history_features, enrich_feature_space, relative_attributes_features, \
    enrich_positions_feature_space, value_rating_features
from stat_predict.dataset.utils import scale_data
from stat_predict.static.config import LABEL_COLUMN, TEST_YEAR, \
    FeaturesParameters, ArtifactsPaths
from stat_predict.static.utils import COLUMNS, SPECIFIC_POS_COLS, REMOVE_COLS, IDENTIFIERS, METADATA_COLS, \
    SKILLS_PREFIXES, major_attrs


@dataclass
class DatasetArgs:
    years_back: int
    label_column: str = LABEL_COLUMN
    dataset_path: str = ArtifactsPaths.STAT_DATASET
    output_dir: str = ArtifactsPaths.ARTIFACTS_DIR
    include_hist_features: bool = FeaturesParameters.add_hist_features
    na_filler: float = -1  # na_filler: value to replace NAs with
    shuffle: bool = True
    validation_frac: float = 0.8
    test_year: int = TEST_YEAR


@dataclass
class StatsDataset:
    args: DatasetArgs
    data: Dict[str, pd.DataFrame] = None
    labels: dict = None
    columns_mapping: Dict[str, List[str]] = None

    @property
    def dataset_path(self):
        return os.path.join(self.args.output_dir,
                            self.args.dataset_path.replace('{NUM_YEARS_BACK}', str(self.args.years_back)))

    def manage_missing_years(self, x: pd.DataFrame) -> (pd.DataFrame, bool):
        """
        Manage possible missing years in player data
        return: updated dataframe - add rows with -1 for all columns, and if to skip the player or not
        """
        # Check if we have enough rows for the player / fill if missing
        if len(x) < self.args.years_back:
            # Pad missing years with zeros
            existing_years = x[COLUMNS.YEAR].tolist()
            max_year, min_year = max(existing_years), min(existing_years)
            missing_years = [y for y in range(max_year - self.args.years_back + 1, max_year)
                             if y not in existing_years]
            x = pd.concat([
                x,
                -pd.DataFrame(np.ones((self.args.years_back - x.shape[0], x.shape[1])), columns=x.columns)],
                axis=0)
            x['pad'] = [0] * len(existing_years) + [1] * len(missing_years)
            assert len(x) == self.args.years_back
            assert len(missing_years) == x['pad'].sum()

            # fix years -> order df accordingly (player may miss one year in between)
            all_years = existing_years + missing_years
            x[COLUMNS.YEAR] = all_years
            x = x.sort_values(COLUMNS.YEAR, ascending=False)
        else:
            # No pad required
            x['pad'] = 0
        return x, False

    def collect_full_feature_names(self,
                                   df: pd.DataFrame,
                                   columns_mapping: Dict[str, List[str]]
                                   ) -> (List[str], Dict[str, List[str]]):
        """ Flat the base features space namings, adding the year index ('_row_{i}') """
        base_columns = list(df.columns)
        numeric_columns = [c for c in df.select_dtypes(np.number)
                           if c not in columns_mapping['identifiers']
                           and not c.endswith('_id')
                           ]
        columns_mapping['numeric'].append('pad')
        assert len(df) == self.args.years_back
        full_feature_names = [f'{c}_row_{i}' for i in range(len(df)) for c in base_columns]
        columns_mapping['numeric'] = [f'{c}_row_{i}' for i in range(len(df)) for c in numeric_columns]
        columns_mapping['categorical'] = [f'{c}_row_{i}' for i in range(len(df)) for c in
                                          columns_mapping['categorical']]
        return full_feature_names, columns_mapping

    def build(
            self,
            df: pd.DataFrame,
            columns_mapping: dict,
            hist_columns: Iterable[str] = FeaturesParameters.hist_features_columns,
            force: bool = False):
        """
        Builds a dataset with the required structure:
        Breaks the players all-years history dataframe into data, labels dicts to collection of time-series format:
            data: {years_back [int]: {year [int]: {player_id [int]: player_df [pd.DataFrame]}, ...}, ...}
            data: {years_back [int]: {year [int]: {player_id [int]: labels [pd.Series]}, ...}, ...}

        Args:
            df: DataFrame with FIFA data.
            hist_columns: columns to use for history features function
            force: whether to force new dataset creation of allow loading existing one
            columns_mapping: util mapping of columns types -> names of columns (e.g., categorical)

        Populate the object with:
            data: {'X': pd.Series of features,
                'identifiers': dict of identifiers values (e.g., player id) - set up in columns_mapping
                'metadata': dict of metadata attrs - set up in columns_mapping
            }
            labels: {player_id: pd.Series of label columns}
        """
        if not force and os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'rb') as f:
                return pickle.load(f)

        identifiers_cols, metadata_cols, numeric_cols, categorical_columns, label_columns = (
            columns_mapping['identifiers'], \
            columns_mapping['metadata'], \
            columns_mapping['numeric'], \
            columns_mapping['categorical'], \
            columns_mapping['label_columns'])

        # Ensure data is sorted
        df = df.sort_values([COLUMNS.PLAYER_ID, COLUMNS.YEAR], ascending=[True, False])
        players = list(df[COLUMNS.PLAYER_ID].unique())

        # Convert categorical columns
        df[categorical_columns] = df[categorical_columns].astype("category")
        if self.args.shuffle:
            random.shuffle(players)

        data, labels = ({'train': [], 'validation': [], 'test': [], 'future': []},
                        {'train': [], 'validation': [], 'test': [], 'future': []})
        players_count = defaultdict(int)
        full_feature_names = None

        potential_percent_col = [c for c in df.columns if COLUMNS.POTENTIAL_PERCENT in c][0]
        train_val_year_players = np.random.choice(players, size=int(len(players) * self.args.validation_frac))
        for player in tqdm(players, desc='Build dataset: iterating players'):
            player_data = df[df[COLUMNS.PLAYER_ID] == player]
            for start_idx in range(len(player_data)):
                x = self.get_player_df_years_back(player_data, start_idx)

                # Validate we have at least 1 row of history and a label for the first tow
                if len(x) < 1 or pd.isna(x['label'].iloc[0]):
                    continue

                # Add features calculated based on num years back
                x = self.add_years_back_features(x, hist_columns, potential_percent_col)
                # Treat missing years in the sequence, if exist
                x, skip = self.manage_missing_years(x)
                if skip:
                    continue

                # Count player
                year = x[COLUMNS.YEAR].iloc[0]
                players_count[year] += 1

                # Remove year, 'label' and 'raw_label' from data - label_columns
                y = x[label_columns].iloc[0].to_dict()
                x.drop(columns=label_columns, inplace=True)

                # FLAT DATA + GIVE COLUMNS NAMES OVER YEAR +
                if full_feature_names is None:
                    full_feature_names, columns_mapping = self.collect_full_feature_names(x, columns_mapping)

                # Append to the relevant dataset
                set_name = self.infer_set_by_year(player, train_val_year_players, year)
                x.select_dtypes(np.number).fillna(self.args.na_filler, inplace=True)
                data[set_name].append(x.to_numpy().flatten())
                labels[set_name].append(y)

        # Concat all instances
        for _set in ['train', 'validation', 'test']:
            data[_set] = pd.DataFrame(np.vstack(data[_set]), columns=full_feature_names)
        self.data = data
        self.labels = labels
        self.columns_mapping = columns_mapping
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self, f)

    def infer_set_by_year(self, player: str, train_val_year_players: List[str], year: int) -> str:
        """ returns the split set name given the year """
        set_name = 'train'
        if year == self.args.test_year + 1:
            set_name = 'future'
        elif year == self.args.test_year:
            set_name = 'test'
        elif year == self.args.test_year - 1:
            if player not in train_val_year_players:
                set_name = 'validation'
        return set_name

    def add_years_back_features(self,
                                x: pd.DataFrame,
                                hist_columns: Iterable[str],
                                potential_percent_col: str) -> pd.DataFrame:
        # Add year-over-year features
        x = add_categorical_yoy_features(x, potential_percent_col=potential_percent_col)

        if self.args.include_hist_features:
            row_hist = history_features(
                x[hist_columns + [COLUMNS.YEAR, 'enrich:in_national_team', potential_percent_col]],
                na_filler=self.args.na_filler
            )
            x = pd.concat([x, row_hist], axis=1)
        return x

    def get_player_df_years_back(self, player_data: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        """ Returns player history dataframe """
        x = player_data.iloc[start_idx:start_idx + self.args.years_back]
        min_year_to_use = x[COLUMNS.YEAR].max() - self.args.years_back + 1
        x = x[x[COLUMNS.YEAR] >= min_year_to_use]
        return x


def get_all_years_data_df(file_path: str, label_col: str) -> pd.DataFrame:
    """ Return a dataframe - concat of all the raw years fifa dataset"""
    if not os.path.exists(file_path):
        # Read all years df
        df = pd.read_csv(os.path.join(ArtifactsPaths.DATA_DIR, ArtifactsPaths.ALL_YERAS_DATA))
        df = df[[c for c in df.columns if 'Unnamed' not in c and c not in REMOVE_COLS]]
        df = align_columns_names(df)
        df[COLUMNS.YEAR] = df[COLUMNS.YEAR].astype(int)

        # Add fifa 2025 data (labels only)
        fifa_25_data = pd.read_csv(os.path.join(ArtifactsPaths.DATA_DIR, ArtifactsPaths.LAST_YEAR_LABELS))
        fifa_25_data[COLUMNS.PLAYER_ID] = fifa_25_data.url.apply(lambda x: x.split('/')[-1]).astype(int)
        fifa_25_labels = fifa_25_data.set_index(COLUMNS.PLAYER_ID)['OVR'].to_dict()

        # Create labels - next year label_col value
        print(' - creating raw labels')
        labels = df[[COLUMNS.PLAYER_ID, COLUMNS.YEAR, label_col]].copy()
        labels[COLUMNS.YEAR] -= 1
        labels.rename(columns={label_col: 'label'}, inplace=True)
        df = df.merge(labels, on=[COLUMNS.PLAYER_ID, COLUMNS.YEAR], how='left')
        df['label'] = df[[COLUMNS.PLAYER_ID, COLUMNS.YEAR, 'label']].apply(
            lambda x: fifa_25_labels.get(x[COLUMNS.PLAYER_ID]) if x[COLUMNS.YEAR] == 24 else x['label'], axis=1)

        df.to_csv(file_path)
        return df
    else:
        df = pd.read_csv(file_path)
        return df[[c for c in df.columns if 'Unnamed' not in c and c not in REMOVE_COLS]]


def align_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align naming across different versions of fifa datasets
    """
    df = df.rename(columns={'club_contract_valid_until_year': COLUMNS.CONTRACT_EXPIRATION_DATE,
                            'fifa_version': COLUMNS.YEAR,
                            }
                   )
    return df


def process_all_years_df(
        test_year: int,
        raw_data_all_years_path: str = os.path.join(ArtifactsPaths.DATA_DIR, ArtifactsPaths.ALL_YEARS_MERGED_DATA),
        force: bool = False,
        processed_df_path: str = ArtifactsPaths.ALL_YEARS_PROCESSED_DF,
        scaling: str = 'max',
        na_filler: float = -1.,
        label_col: str = LABEL_COLUMN,
        metadata_priors_cols: None | List[str] = None,
        year_col: str = COLUMNS.YEAR
) -> (pd.DataFrame, dict, dict):
    """
    Return all years dataframe, processed:
        - scaled - to be between 0,1
        - with new features
    :param test_year: the year used for testing (int)
    :param raw_data_all_years_path: path to existing file / to save after create
    :param metadata_priors_cols: list of metadata features to apply prior-feature extraction on
    :param force: if to load existing data when possible
    :param processed_df_path: path of processed df
    :param scaling: form of scaling to apply
    :param label_col: name of label column
    :param year_col: name of year column
    :return: data df, labels_dict, columns mapping
    """
    if metadata_priors_cols is None:
        metadata_priors_cols = FeaturesParameters.metadata_prior_columns[:]
    if not force:
        if os.path.exists(processed_df_path):
            with open(processed_df_path, 'rb') as f:
                return pickle.load(f)

    # Read
    all_data = get_all_years_data_df(raw_data_all_years_path, label_col=label_col)
    raw_columns = list(all_data.columns)
    numeric_cols = [c for c in all_data.select_dtypes(np.number).columns
                    if c not in IDENTIFIERS and c not in METADATA_COLS and c not in SPECIFIC_POS_COLS]
    print(' - Non numeric columns:', ', '.join([c for c in raw_columns if c not in numeric_cols]))
    all_data[numeric_cols].fillna(na_filler, inplace=True)

    # Search for duplicates
    print(' - Raw df size:', len(all_data))
    all_data = all_data.drop_duplicates(subset=[COLUMNS.PLAYER_ID, COLUMNS.YEAR], keep='first')
    all_data.drop(columns=['fifa_update'], inplace=True)
    print(' - Non duplicates df size:', len(all_data))
    print()

    # Enrich features
    all_data = enrich_feature_space(all_data, family_prefix='enrich:', na_value=na_filler)
    all_data = enrich_positions_feature_space(all_data, family_prefix='position:')
    all_data = relative_attributes_features(all_data, family_prefix='relative_attributes:', na_value=na_filler)
    all_data = value_rating_features(all_data,
                                     test_year=test_year, family_prefix='relative_attributes:', na_value=na_filler)
    # meta_priors - categorical columns that will be represented by their label priors (up to this year, no leakage)
    # e.g., 'age', 'nationality_name', 'club_name', 'league_name', 'club_position'
    all_data = adds_categorical_prior_features(all_data,
                                               metadata_priors_cols=metadata_priors_cols,
                                               label_col=label_col,
                                               year_col=year_col,
                                               feature_prefix='cat_prior:',
                                               na_value=na_filler
                                               )

    # Define scaling rules
    skills_cols = [c for c in all_data.columns if c in major_attrs or
                   any([c.startswith(pre) for pre in SKILLS_PREFIXES])]

    # Scale data
    all_data, scaling_values = scale_data(all_data, skills_cols, scaling=scaling)
    columns_mapping = {
        'numeric': numeric_cols,
        'metadata': METADATA_COLS[:],
        'identifiers': IDENTIFIERS[:],
        'specific_positions': SPECIFIC_POS_COLS[:],
        'skills': skills_cols[:],
        'major_attrs': major_attrs[:],
        'categorical': [c for c in metadata_priors_cols if c != COLUMNS.AGE],
    }

    if processed_df_path is not None:
        print(' - dumping processed_df pickle')
        with open(processed_df_path, 'wb') as f:
            pickle.dump((all_data, columns_mapping, scaling_values), f)
    return all_data, columns_mapping, scaling_values
