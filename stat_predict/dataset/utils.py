from dataclasses import dataclass
from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stat_predict.static.config import YOY_DIFF_TO_CAT, TEST_YEAR, AGE_NORM_VALUE
from stat_predict.static.utils import SKILLS_PREFIXES, COLUMNS


def number_to_cat_bins(num: float, cats_thresholds: List[float] | Tuple[float] = YOY_DIFF_TO_CAT):
    """ Convert a number into a category based on list of categories thresholds """
    for i, val in enumerate(cats_thresholds):
        if num <= val:
            return i
    return len(cats_thresholds)


@dataclass
class FeatureFamilies:
    major_attrs: str = 'major_attrs'
    raw_skill: str = 'raw_skill'
    raw: str = 'raw'
    enrich: str = 'enrich'
    pitch_position: str = 'position'
    cat_prior: str = 'cat_prior'
    relative_attributes: str = 'relative_attributes'
    cat_yoy: str = 'cat_yoy'
    hist_norm: str = 'hist_norm'
    history: str = 'history'
    phase_model: str = 'phase_model'


RAW_FAMILIES = [FeatureFamilies.raw, FeatureFamilies.raw_skill, FeatureFamilies.major_attrs]


def get_feature_family(feature: str):
    if ':' in feature:
        return feature.split(':')[0]
    else:
        if FeatureFamilies.major_attrs in feature:
            return FeatureFamilies.major_attrs
        if any([feature.startswith(pre) for pre in SKILLS_PREFIXES]):
            return FeatureFamilies.raw_skill
        if feature.startswith('pitch_pos_'):
            return FeatureFamilies.pitch_position
        return 'raw'


def scale_data(all_data: pd.DataFrame, skills_cols: Iterable[str], scaling: str = 'max') -> (pd.DataFrame, dict):
    print(' - scaling data')
    unscaled_columns = ['weight_kg', 'height_cm', 'league_level', 'skill_moves', 'weak_foot',
                        'international_reputation']
    for raw_col in ['work_rate_def', 'work_rate_attack', 'years_left_in_contract', 'age_trans', 'potential-overall-%']:
        for c in all_data.columns:
            if raw_col in c:
                unscaled_columns.append(c)
    if scaling == 'max':
        scaling_max_values = {skill: 100 for skill in skills_cols}
        scaling_max_values.update({'age': AGE_NORM_VALUE, 'release_clause_eur': 1e7, 'value_eur': 1e7, 'wage_eur': 5e4})
        # Take max ignoring test year
        scaling_max_values.update(all_data.loc[all_data.year < TEST_YEAR, unscaled_columns].max(axis=0).to_dict())
        # Transform (do not limit to 1)
        for col in scaling_max_values:
            all_data[col] = all_data[col] / scaling_max_values[col]
    else:
        scaler = StandardScaler()
        col_to_scale = [c for c in skills_cols] + unscaled_columns
        scaler.fit(all_data[all_data.year < TEST_YEAR, col_to_scale])
        all_data[col_to_scale] = scaler.fit_transform(all_data[col_to_scale])
        scaling_max_values = scaler

    return all_data, scaling_max_values


def missing_values_report(df: pd.DataFrame):
    print()
    print('#' * 200)
    print('Missing values')
    na_cols = df.isna().mean(axis=0).to_dict()
    na_cols = {k: round(v, 3) for k, v in na_cols.items() if v > 0}
    sig_na_cols = {k: round(v, 3) for k, v in na_cols.items() if v >= 0.15 and 'goalkeeping' not in k}
    print(f'{df.shape[1]} total columns')
    print(f'{len(na_cols)} columns with empty values (11% are GKs)')
    print(f'{len(sig_na_cols)} columns with > 15% empty values (without GK columns)')

    print(' - Significant NA columns:\n', sig_na_cols)

    print('\nFeature families distribution')
    features_families = pd.Series([get_feature_family(c) for c in df.columns])
    print(features_families.value_counts())


def features_distribution_report(df: pd.DataFrame):
    # Distribution of columns
    print('#' * 200)
    print('\nColumns scale - max')
    cols_max = df.select_dtypes(np.number).max(axis=0).to_dict()
    cols_max = {k: round(v, 3) for k, v in cols_max.items() if v > 1}
    print(f'{len(cols_max)} columns have max value > 1')
    sig_non_normed_max = {k: round(v, 3) for k, v in cols_max.items() if v >= 2}
    print(f'{len(sig_non_normed_max)} columns have max value >= 2')
    max_vals_iter = sorted(list(sig_non_normed_max.items()), key=lambda x: x[1], reverse=True)
    if len(max_vals_iter) > 0:
        print(f'{len(max_vals_iter)} Columns with max value >= 2:')
        print(max_vals_iter[:10])

    print('\nColumns scale - min')
    cols_min = df.select_dtypes(np.number).min(axis=0).to_dict()
    cols_min = {k: round(v, 3) for k, v in cols_min.items() if v < -1}
    print(f'{len(cols_min)} columns have min value < -1')
    sig_non_normed_min = {k: round(v, 3) for k, v in cols_min.items() if v <= -2}
    print(f'{len(sig_non_normed_min)} columns have min value <= -2')
    min_vals_iter = sorted(list(sig_non_normed_min.items()), key=lambda x: x[1], reverse=False)
    if len(max_vals_iter) > 0:
        print(f'{len(max_vals_iter)} Columns with min value <= -2:')
        print(min_vals_iter[:10])

    print('\nColumns scale - std')
    cols_std = df.select_dtypes(np.number).std(axis=0).to_dict()
    cols_std = {k: round(v, 3) for k, v in cols_std.items() if v < 1}
    print(f'{len(cols_std)} columns have std value < 1')
    sig_low_std = {k: round(v, 3) for k, v in cols_std.items() if v <= 0.1}
    print(f'{len(sig_low_std)} columns have min value <= 0.1')
    min_vals_std = sorted(list(sig_low_std.items()), key=lambda x: x[1], reverse=False)
    print('20 Columns with lowest std value:')
    print(min_vals_std[:20])


def plot_label_yoy_distribution(yoy_dist_df: pd.DataFrame):
    yoy_5th = yoy_dist_df.set_index('year')['5%'].loc[16:24].astype(float).values
    yoy_10th = yoy_dist_df.set_index('year')['10%'].loc[16:24].astype(float).values
    yoy_50th = yoy_dist_df.set_index('year')['50%'].loc[16:24].astype(float).values
    yoy_90th = yoy_dist_df.set_index('year')['90%'].loc[16:24].astype(float).values
    yoy_95th = yoy_dist_df.set_index('year')['95%'].loc[16:24].astype(float).values
    label_years = list(range(2016, 2025))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=label_years,
        y=yoy_dist_df['proportion'],
        name='% label = 1',
        marker=dict(color='rgba(166, 166, 166, 0.5)', opacity=0.85),
    ), secondary_y=False)
    fig.update_layout(barcornerradius=10)
    fig.add_trace(go.Scatter(
        x=label_years, y=yoy_5th,
        mode='lines',
        name='5th Percentile (YoY)',
        line=dict(color='firebrick', dash='dash'),
        marker=dict(size=6)
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=label_years, y=yoy_10th,
        mode='lines',
        name='10th Percentile (YoY)',
        line=dict(color='rgba(240, 105, 105, 0.9)', dash='dash'),
        marker=dict(size=6)
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=label_years, y=yoy_50th,
        mode='lines',
        name='Median YoY',
        line=dict(color='grey'),
        marker=dict(size=6)
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=label_years, y=yoy_90th,
        mode='lines',
        name='90th Percentile (YoY)',
        line=dict(color='rgba(3, 166, 8, 0.5)', dash='dash'),
        marker=dict(size=6)
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=label_years, y=yoy_95th,
        mode='lines',
        name='95th Percentile (YoY)',
        line=dict(color='forestgreen', dash='dash'),
        marker=dict(size=6)
    ), secondary_y=True)
    fig.update_yaxes(title_text="YoY Rating - Percentiles", secondary_y=True)
    fig.update_yaxes(title_text="Proportion (%)", secondary_y=False)
    fig.update_xaxes(title_text="Year")
    fig.update_layout(
        template="plotly_white",
        title_text="Label Distribution",
        height=500,
        width=1000,
        legend=dict(x=0.02, y=1.1, orientation="h")
    )
    fig.show()


def label_description(all_data):
    print('Player-year distribution')
    x = all_data[all_data[COLUMNS.YEAR] >= 18]
    player_year_dist = pd.Series(x[[COLUMNS.PLAYER_ID, COLUMNS.YEAR]].groupby(COLUMNS.PLAYER_ID) \
                                 .count()[COLUMNS.YEAR]).value_counts()
    print(player_year_dist)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=player_year_dist.index.astype(str),
        y=player_year_dist.values,
        marker=dict(color='gray', opacity=0.85),
    ))
    fig.update_yaxes(title_text="Number of Players")
    fig.update_xaxes(title_text="Years in Dataset")
    fig.update_layout(
        title_text="Player-Year History Distribution",
        height=500,
        width=1000,
        legend=dict(x=0.02, y=1.1, orientation="h"),
        template="plotly_white",
    )
    fig.show()

    print(f'\nYears label distribution')
    label_year_distribution = x[[COLUMNS.YEAR, 'label']].groupby(COLUMNS.YEAR) \
        .value_counts(normalize=True).reset_index()
    print()
    print(f'\nYoY rating distribution')
    x['yoy_diff'] = (x['raw_label'] - x[f'{COLUMNS.PLAYER_RATING}_row_0']).astype(float)
    print(x['yoy_diff'].round(2)
          .describe(percentiles=[0.05, 0.1, 0.2, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]))
    print('\nYoY diff distribution by year:')
    yoy_dist_df = x.groupby(COLUMNS.YEAR)['yoy_diff'].describe(
        percentiles=[0.05, 0.1, 0.2, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    yoy_dist_df = yoy_dist_df.merge(label_year_distribution[label_year_distribution['label'] == 1], on=COLUMNS.YEAR)
    yoy_dist_df.to_csv('yoy_dist_by_year.csv')
    plot_label_yoy_distribution(yoy_dist_df)
