from typing import List, Iterable, Optional
import numpy as np
import pandas as pd

from stat_predict.static.utils import positions_emb, body_types, COLUMNS, work_rate_mapper
from stat_predict.dataset.utils import number_to_cat_bins


def percentile(n):
    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = 'percentile_{:02.0f}'.format(n * 100)
    return percentile_


def check_columns_validity(df: pd.DataFrame, columns: Optional[Iterable[str]], allow_singular_values: bool = False):
    if columns is None:
        columns = df.columns
    # Check
    for col in columns:
        assert df[col].isna().sum() < len(df), f"Col {col} has only NAs"
        if not allow_singular_values:
            assert len(df[col].unique()) > 1, f"Col {col} has only one value in it"


def years_in_club(x: pd.Series, norm_factor: float = 10.):
    if pd.isna(x[COLUMNS.JOIN_CLUB_DATE]):
        return np.nan
    date_join_club = x[COLUMNS.JOIN_CLUB_DATE]
    year_join_club = np.nan
    if '/' in str(date_join_club):
        year_join_club = float(str(year_join_club).split('/')[-1])
    elif '-' in str(date_join_club):
        year_join_club = float(date_join_club.split('-')[0])
    else:
        return np.nan
    return (2000 + x[COLUMNS.YEAR] - year_join_club) / norm_factor


def value_rating_features(all_players_df: pd.DataFrame,
                          test_year: float,
                          year_col: str = COLUMNS.YEAR,
                          na_value: float = .0,
                          family_prefix: str = 'relative_attributes:') -> pd.DataFrame:
    """
        Normalizing player rating by his value / wage
    """
    print(' - value_rating_features...')
    start_columns = list(all_players_df.columns)
    all_players_df[f'{family_prefix}potential-overall-%'] = all_players_df.overall / all_players_df.potential
    all_players_df[f'{family_prefix}value-over-overall'] = all_players_df[COLUMNS.PLAYER_VALUE] / all_players_df[
        COLUMNS.PLAYER_RATING]
    all_players_df[f'{family_prefix}value-over-potential'] = all_players_df[
                                                                 COLUMNS.PLAYER_VALUE] / all_players_df.potential
    all_players_df[f'{family_prefix}wage-over-overall'] = all_players_df[COLUMNS.PLAYER_WAGE] / all_players_df[
        COLUMNS.PLAYER_RATING]
    all_players_df[f'{family_prefix}wage-over-potential'] = all_players_df[
                                                                COLUMNS.PLAYER_WAGE] / all_players_df.potential
    # Normalize values to be up to 1
    for col in [f'{family_prefix}value-over-overall', f'{family_prefix}value-over-potential',
                f'{family_prefix}wage-over-overall', f'{family_prefix}wage-over-potential']:
        col_max_val = all_players_df.loc[all_players_df[year_col] < test_year, col].max()
        all_players_df[col] /= col_max_val

    new_features = [c for c in all_players_df if c not in start_columns]
    all_players_df[new_features].fillna(na_value, inplace=True)

    check_columns_validity(all_players_df, columns=new_features)
    return all_players_df


def relative_attributes_features(all_players_df: pd.DataFrame,
                                 na_value: float = -1.,
                                 family_prefix: str = 'relative_attributes:') -> pd.DataFrame:
    """
    Normalizing player rating by benchmarking to other players in team / league/ etc
    """
    print(' - relative_attributes_features...')
    start_columns = list(all_players_df.columns)

    def create_ratio_feature(
            group_cols: List[str],
            target_col: str,
            feature_suffix: str,
            normalize_100: bool = False,
            agg: str = 'max'):
        group_key = group_cols + [target_col]
        grouped = all_players_df[group_key].groupby(group_cols).agg(agg)[target_col].to_dict()
        feature_name = f'{family_prefix}{feature_suffix}'
        all_players_df[feature_name] = all_players_df[group_cols + [target_col]].apply(
            lambda x: x[target_col] / grouped.get(tuple(x[col] for col in group_cols), np.nan)
            if not pd.isna(x[target_col]) else np.nan,
            axis=1
        )
        if normalize_100:
            all_players_df[feature_name] /= 100

    # Define common settings
    VALUE = COLUMNS.PLAYER_VALUE
    WAGE = COLUMNS.PLAYER_WAGE
    CLUB = COLUMNS.CLUB
    LEAGUE = COLUMNS.LEAGUE
    POS = COLUMNS.PLAYER_POSITION
    YEAR = COLUMNS.YEAR

    # Define all normalization configs
    normalization_configs = [
        # Single dimension - VALUE
        ([LEAGUE, YEAR], VALUE, 'player_relative_max_league_value', False, 'max'),
        ([LEAGUE, YEAR], VALUE, 'player_relative_mean_league_value', True, 'mean'),
        ([CLUB, YEAR], VALUE, 'player_relative_max_club_value', False, 'max'),
        ([CLUB, YEAR], VALUE, 'player_relative_mean_club_value', True, 'mean'),
        ([CLUB, YEAR], VALUE, 'player_relative_sum_club_value', False, 'sum'),
        # Single dimension - WAGE
        ([LEAGUE, YEAR], WAGE, 'player_relative_max_league_wage', False, 'max'),
        ([LEAGUE, YEAR], WAGE, 'player_relative_mean_league_wage', True, 'mean'),
        ([CLUB, YEAR], WAGE, 'player_relative_max_club_wage', False, 'max'),
        ([CLUB, YEAR], WAGE, 'player_relative_mean_club_wage', True, 'mean'),
        ([CLUB, YEAR], WAGE, 'player_relative_sum_club_wage', False, 'sum'),
        # Two dimensions (League + Position + Year)
        ([LEAGUE, POS, YEAR], VALUE, 'player_relative_max_league_position_value', False, 'max'),
        ([LEAGUE, POS, YEAR], VALUE, 'player_relative_mean_league_position_value', True, 'mean'),
        ([LEAGUE, POS, YEAR], WAGE, 'player_relative_max_league_position_wage', False, 'max'),
        ([LEAGUE, POS, YEAR], WAGE, 'player_relative_mean_league_position_wage', True, 'mean'),
    ]

    # Apply all normalization configs
    for group_cols, target_col, feature_suffix, normalize_100, agg in normalization_configs:
        create_ratio_feature(
            group_cols=group_cols,
            target_col=target_col,
            feature_suffix=feature_suffix,
            normalize_100=normalize_100,
            agg=agg
        )

    # Fill NaNs in new features
    new_features = [c for c in all_players_df.columns if c not in start_columns]
    all_players_df[new_features] = all_players_df[new_features].fillna(na_value)
    check_columns_validity(all_players_df, columns=new_features)

    return all_players_df


def enrich_feature_space(
        all_players_df: pd.DataFrame,
        year_col: str = COLUMNS.YEAR,
        na_value: float = .0,
        fill_foot: str = 'right',
        family_prefix: str = 'enrich:'
) -> pd.DataFrame:
    """ Add simple features based on current features manipulation """
    print(' - enrich_feature_space...')
    start_columns = list(all_players_df.columns)

    # Work rate
    all_players_df[f'{family_prefix}work_rate_def'] = all_players_df['work_rate'].apply(
        lambda x: work_rate_mapper[x.split('/')[0]] if not pd.isna(x) else x)
    all_players_df[f'{family_prefix}work_rate_attack'] = all_players_df['work_rate'].apply(
        lambda x: work_rate_mapper[x.split('/')[1]] if not pd.isna(x) else x)
    # More features such as: in_national_team, play_in_based_nation, years_left_in_contract, etc
    all_players_df[f'{family_prefix}in_national_team'] = all_players_df['nation_position'].notna().astype(int)
    all_players_df[f'{family_prefix}age_trans'] = all_players_df.age.apply(lambda x: abs(x - 25))

    # Foot based features
    all_players_df[f'{family_prefix}preferred_foot_right'] = all_players_df['preferred_foot'].fillna(fill_foot) \
        .apply(lambda x: float(x.lower() == 'right' or x.lower() == 'both'))
    all_players_df[f'{family_prefix}preferred_foot_left'] = all_players_df['preferred_foot'].fillna(fill_foot) \
        .apply(lambda x: float(x.lower() == 'left' or x.lower() == 'both'))

    # Time based features
    all_players_df[f'{family_prefix}on_loan'] = all_players_df['club_loaned_from'].apply(
        lambda x: float(not pd.isna(x)))
    all_players_df[f'{family_prefix}years_left_in_contract'] = (
            all_players_df[COLUMNS.CONTRACT_EXPIRATION_DATE].astype(float) -
            all_players_df[year_col] - 2000).apply(lambda x: max(0, x))
    all_players_df[f'{family_prefix}_years_in_club'] = all_players_df[[COLUMNS.JOIN_CLUB_DATE, COLUMNS.YEAR]].apply(
        lambda x: years_in_club(x), axis=1)

    # League to nation
    league2nation = all_players_df[[COLUMNS.LEAGUE, COLUMNS.NATIONALITY]] \
        .groupby(COLUMNS.LEAGUE).agg(mod=(COLUMNS.NATIONALITY, lambda x: x.value_counts().index[0]))['mod'] \
        .to_dict()
    all_players_df[f'{family_prefix}play_in_based_nation'] = (
            all_players_df[COLUMNS.NATIONALITY] ==
            all_players_df[COLUMNS.LEAGUE].map(league2nation)).astype(int)

    body_type_df = pd.DataFrame([[float(b == p)
                                  for p in all_players_df['body_type']] for b in body_types]).T.reset_index(drop=True)
    body_type_df.columns = [f'{family_prefix}body_{b}' for b in body_types]
    all_players_df = pd.concat([all_players_df.reset_index(drop=True), body_type_df], axis=1)

    new_features = [c for c in all_players_df if c not in start_columns]
    all_players_df[new_features].fillna(na_value, inplace=True)
    check_columns_validity(all_players_df, columns=new_features)

    return all_players_df.drop(columns=[COLUMNS.CONTRACT_EXPIRATION_DATE,
                                        COLUMNS.WORK_RATE,
                                        COLUMNS.JOIN_CLUB_DATE,
                                        'nation_position',
                                        ])


def adds_categorical_prior_features(df: pd.DataFrame,
                                    metadata_priors_cols: List[str],
                                    label_col: str,
                                    year_col: str = COLUMNS.YEAR,
                                    feature_prefix: str = 'cat_prior:',
                                    allow_singular_values: bool = False,
                                    na_value: float = -1.
                                    ) -> pd.DataFrame:
    '''
    Encode categorical features using their label prior
    :param df: player / all data dataframe
    :param metadata_priors_cols: columns to operate on
        - 'nationality_name', 'club_name', 'league_name', 'club_position', 'age'
    :param label_col: label column name to add prior by
    :param year_col: year column name to add prior by without leakage
    :param feature_prefix: prefix to add to the feature name (str)
    :param allow_singular_values: whether to allow a single value to be output (zero variance)
    '''
    print(' - adds_categorical_prior_features...')
    meta_priors = {col: {} for col in metadata_priors_cols}
    max_meta_priors = {col: {} for col in metadata_priors_cols}
    percentile_75_meta_priors = {col: {} for col in metadata_priors_cols}
    percentile_25_meta_priors = {col: {} for col in metadata_priors_cols}
    normalization_factor = float(df[label_col].max())
    for agg_col in metadata_priors_cols:
        # Prior - mean
        mean_df = df[[agg_col, year_col, label_col]].groupby([agg_col, year_col]).mean().to_dict(orient='index')
        meta_priors[agg_col] = {k: v[label_col] for k, v in mean_df.items()}

        # Prior by max value
        max_df = df[[agg_col, year_col, label_col]].groupby([agg_col, year_col]).max().to_dict(orient='index')
        max_meta_priors[agg_col] = {k: v[label_col] for k, v in max_df.items()}

        # Prior by q1, q3
        percentile_75_df = df[[agg_col, year_col, label_col]].groupby([agg_col, year_col]).agg(
            percentile(0.75)).to_dict(orient='index')
        percentile_75_meta_priors[agg_col] = {k: v[label_col] for k, v in percentile_75_df.items()}
        percentile_25_df = df[[agg_col, year_col, label_col]].groupby([agg_col, year_col]).agg(
            percentile(0.25)).to_dict(orient='index')
        percentile_25_meta_priors[agg_col] = {k: v[label_col] for k, v in percentile_25_df.items()}

    # Enrich loan status
    df[f'{feature_prefix}club_loaned_from_label_prior'] = df[['club_loaned_from', COLUMNS.YEAR]].apply(
        lambda x: meta_priors[COLUMNS.CLUB][(x[0], x[1])] / 100.0 if (x[0], x[1]) in meta_priors[COLUMNS.CLUB] \
            else na_value, axis = 1)
    df['club_loaned_from'] = df['club_loaned_from'].replace(np.nan, '').astype('category')

    df = get_prior_features(df, meta_priors, scale_factor=normalization_factor, prefix=feature_prefix)
    df = get_prior_features(df, max_meta_priors,
                            prefix=f'{feature_prefix}prior_max_',
                            scale_factor=normalization_factor,
                            fill_prior=na_value)
    df = get_prior_features(df, percentile_25_meta_priors,
                            prefix=f'{feature_prefix}percentile_25_',
                            scale_factor=normalization_factor,
                            fill_prior=na_value)
    df = get_prior_features(df, percentile_75_meta_priors,
                            prefix=f'{feature_prefix}percentile_75_',
                            scale_factor=normalization_factor,
                            fill_prior=na_value)

    check_columns_validity(df, columns=meta_priors.keys(), allow_singular_values=allow_singular_values)
    check_columns_validity(df, columns=percentile_75_meta_priors.keys(), allow_singular_values=allow_singular_values)
    check_columns_validity(df, columns=percentile_25_meta_priors.keys(), allow_singular_values=allow_singular_values)
    check_columns_validity(df, columns=max_meta_priors.keys(), allow_singular_values=allow_singular_values)
    return df


def enrich_positions_feature_space(df: pd.DataFrame,
                                   fill_position: tuple = (-2, -2),
                                   pos_prefix: str = 'pitch_pos_',
                                   na_value: float = -1.,
                                   family_prefix: str = 'position:'
                                   ) -> pd.DataFrame:
    ''' Adds features to categorical features '''
    print(' - enrich_categorical_feature_space...')

    # 'player_positions' - specified all pitch positions the player can play at.
    df[f'{family_prefix}{pos_prefix}num_player_positions'] = df['player_positions'].apply(lambda x: x.count(',') + 1)

    # COLUMNS.PLAYER_POSITION - specify the position of the player in its club. Has also 'RES' and 'SUB'.
    # Add indication to the player position in the squad
    df[f'{family_prefix}{pos_prefix}is_RES'] = df[COLUMNS.PLAYER_POSITION].apply(lambda x: x == 'RES').astype(float)
    df[f'{family_prefix}{pos_prefix}is_SUB'] = df[COLUMNS.PLAYER_POSITION].apply(lambda x: x == 'SUB').astype(float)

    # Create 'player_playing_position' that related only to pitch position
    df['player_pitch_position'] = df[[COLUMNS.PLAYER_POSITION, 'player_positions']].apply(
        lambda x: x[0] if (x[0] not in ['RES', 'SUB'] and not pd.isna(x[0])) else x[1].split(',')[0].rstrip(), axis=1
    )
    # tup: vertical position (gk - defence - midfield - offense), horizontal position (left - center - right)
    df['position_tup'] = df['player_pitch_position'].apply(
        lambda x: positions_emb.get(x.upper(), fill_position) if isinstance(x, str) else fill_position)

    # y: vertical position (gk - defence - midfield - offense)
    df[f'{family_prefix}{pos_prefix}y'] = df['position_tup'].apply(lambda x: x[0])
    # Another INT version with 1-step for career phase model
    pos_y_int_mapper = {v: i + 1 for i, v in enumerate(sorted(df[f'{family_prefix}{pos_prefix}y'].unique().tolist()))}
    df[f'{family_prefix}{pos_prefix}position_y'] = df[f'{family_prefix}{pos_prefix}y'].map(pos_y_int_mapper)

    # x: horizontal position (left - center - right)
    df[f'{family_prefix}{pos_prefix}x'] = df['position_tup'].apply(lambda x: x[1])
    df = df.drop(columns=['position_tup', 'player_pitch_position'])
    return df


def add_categorical_yoy_features(player_df: pd.DataFrame,
                                 potential_percent_col: str,
                                 changed_prefix: str = 'changed_',
                                 ) -> pd.DataFrame:
    # Changed metadata - looking for changes in yoy values over categories
    for col, suffix in zip([COLUMNS.CLUB, COLUMNS.LEAGUE, COLUMNS.PLAYER_POSITION, 'player_positions'],
                           ['club', 'league', 'position', 'player_positions']
                           ):
        player_df[f'cat_yoy:{changed_prefix}{suffix}'] = player_df[col].ne(player_df[col].shift(-1))
        # Override last row - we do not have history for it
        player_df[f'cat_yoy:{changed_prefix}{suffix}'].iloc[-1] = np.nan

    # Changing squad status position (SUB / RES)
    status_col = f'cat_yoy:{changed_prefix}squad_status'

    def get_status(row):
        if row['position:pitch_pos_is_RES'] == 1:
            return 0
        elif row['position:pitch_pos_is_SUB'] == 1:
            return 1
        else:
            return 2

    player_df[status_col] = player_df.apply(get_status, axis=1)
    player_df['prev_status'] = player_df[status_col].shift(-1)

    def compute_change(row):
        if pd.isna(row['prev_status']):
            return None  # or 0, or leave as NaN
        diff = row[status_col] - row['prev_status']
        if diff == 0:
            return 0
        elif diff == 1:
            return 0.5  # upgrade: RES -> SUB or SUB -> FT
        elif diff == -1:
            return -0.5  # downgrade: FT -> SUB or SUB -> RES
        elif diff == 2:
            return 1  # upgrade: RES -> FT
        elif diff == -2:
            return -1  # downgrade: FT -> RES
        else:
            return None  # shouldn't happen

    player_df[status_col] = player_df.apply(compute_change, axis=1)
    del player_df['prev_status']

    # Normalize prices by player max value
    max_value = player_df[COLUMNS.PLAYER_VALUE].max()
    max_rating = player_df[COLUMNS.PLAYER_RATING].max()
    max_potential = player_df[COLUMNS.PLAYER_RATING].max()
    max_wage = player_df[COLUMNS.PLAYER_WAGE].max()

    max_potential_ratio = player_df[potential_percent_col].max()
    for col, norm_value in zip(
            [COLUMNS.PLAYER_VALUE, COLUMNS.PLAYER_RATING, COLUMNS.POTENTIAL, potential_percent_col,
             COLUMNS.PLAYER_WAGE],
            [max_value, max_rating, max_potential, max_potential_ratio, max_wage]):
        player_df[f'hist_norm:{col}'] = player_df.apply(lambda x: round(x[col] / norm_value, 2), axis=1)

    return player_df


def get_prior_features(player_df: pd.DataFrame,
                       meta_priors: {str: {int: float}},
                       fill_prior: float = 0,
                       scale_factor: float = 1.,
                       year_col: str = 'year',
                       prefix: str = 'prior_'):
    """ Convert a categorical column with its prior probability (pre-calculated) pulled from given dict """
    for pr in meta_priors:
        player_df[f'{prefix}{pr}'] = player_df[[pr, year_col]].apply(lambda x: meta_priors[pr].get((x[0], x[1])),
                                                                     axis=1) \
                                         .fillna(fill_prior) / scale_factor
    return player_df


def history_features(player_hist_df: pd.DataFrame,
                     na_filler: float = 0,
                     prefix: str = 'history:',
                     return_year_descending: bool = True) -> pd.DataFrame:
    """
    Function extract aggregated features on top of history of target
        - e.g., average of label history, cumsum, etc
    :param player_hist_df: DataFrame ORDERED BY YEAR, DESC
        -   with ONLY columns to be aggregated
    :param na_filler: value to fill na values
    :param prefix: string to add before the input feature name
    :param return_year_descending: whether return the df DESC by year (false for ASC)
    :return: numpy array of features
    """
    features = []
    # Reorder df to be older to newest year
    player_hist_df = player_hist_df.sort_values(by=COLUMNS.YEAR, ascending=True)

    # Sum, max, avg min of YoY diff - overall and potential, overall / potential
    for col in [c for c in player_hist_df.columns if c not in ['index', COLUMNS.YEAR]]:
        history_diffs = [0] + list(np.diff(player_hist_df[col]))
        # Apply agg function
        for agg in [np.cumsum, np.cumprod]:
            player_hist_df[f"{prefix}{col}_{agg.__name__}"] = agg(player_hist_df[col])
            player_hist_df[f"{prefix}{col}_{agg.__name__}_max"] = np.max(
                player_hist_df[f"{prefix}{col}_{agg.__name__}"])
            player_hist_df[f"{prefix}{col}_{agg.__name__}_min"] = np.min(
                player_hist_df[f"{prefix}{col}_{agg.__name__}"])
            player_hist_df[f"{prefix}{col}_diff_{agg.__name__}"] = agg(history_diffs)
            player_hist_df[f"{prefix}{col}_diff_{agg.__name__}_max"] = np.max(
                player_hist_df[f"{prefix}{col}_diff_{agg.__name__}"])
            player_hist_df[f"{prefix}{col}_diff_{agg.__name__}_min"] = np.min(
                player_hist_df[f"{prefix}{col}_diff_{agg.__name__}"])
            features.extend([f"{prefix}{col}_{agg.__name__}",
                             f"{prefix}{col}_{agg.__name__}_max",
                             f"{prefix}{col}_{agg.__name__}_min",
                             f"{prefix}{col}_diff_{agg.__name__}",
                             f"{prefix}{col}_diff_{agg.__name__}_max",
                             f"{prefix}{col}_diff_{agg.__name__}_min",
                             ])

        # Momentum - cumsum where each sample is weighted be its recency
        player_hist_df[f"{prefix}{col}_momentum"] = pd.Series(player_hist_df[col] *
                                                              np.array([i / len(player_hist_df) for i in
                                                                        range(len(player_hist_df))])).cumsum()
        features.append(f"{prefix}{col}_momentum")

        # YoY one hot encoded history (num categories as label)
        column_hist_categories = [number_to_cat_bins(d, cats_thresholds=[-1, 0, 1]) for d in history_diffs]
        player_hist_df[f"{prefix}{col}_growth"] = [v == 1 for v in column_hist_categories]
        features.append(f"{prefix}{col}_growth")
        player_hist_df[f"{prefix}{col}_growth_cumsum"] = np.cumsum(
            [v == 1 for v in column_hist_categories]) / len(column_hist_categories)
        features.append(f"{prefix}{col}_growth_cumsum")

    if return_year_descending:
        player_hist_df = player_hist_df.sort_values(by=COLUMNS.YEAR, ascending=False)
        assert player_hist_df.year.iloc[0] > player_hist_df.year.iloc[-1] or len(player_hist_df) == 1
    return player_hist_df[features].fillna(na_filler)
