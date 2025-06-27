import os
from collections import defaultdict
from typing import List, Dict, Optional, Literal, Tuple
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from stat_predict.eval.utils import PLAYERS_OF_INTEREST, SELECTED_PLAYERS_OF_INTEREST, SELECTED_PLAYERS_U21, \
    FIFA24_FUTURE_STARS, get_model_short_name, plot_calibration_curve, cls_metrics, PLAYERS_NAMES_CONVERTER
from stat_predict.static.config import Font, ArtifactsPaths, YOY_DIFFS_TO_CATS, AGE_TO_CATS, AGE_BINS_NAMES, \
    RATING_BINS, RATING_BINS_NAMES, YOY_DIFFS_TO_CATS_NAMES, AGE_NORM_VALUE
from stat_predict.dataset.utils import number_to_cat_bins


def build_eval_df(labels: List[int],
                  preds: List[int],
                  probs: List[float],
                  ratings_y1_back: List[float],
                  rating_labels: List[float],
                  ratings_yoy_diffs: List[float],
                  players_names: List[str],
                  players_age: List[str],
                  num_years_data: List[float],
                  yoy_diff2cats_thresholds=YOY_DIFFS_TO_CATS
                  ):
    eval_df = pd.DataFrame({'prob': np.round(probs, 2),
                            'label': labels,
                            'pred': preds,
                            'yoy_diff': ratings_yoy_diffs,
                            'last_year_rating': ratings_y1_back,
                            'label_rating': rating_labels,
                            'player': players_names,
                            'player_age': players_age,
                            'num_years_data': num_years_data
                            })
    eval_df['prob_decile'] = eval_df['prob'].round(1)
    eval_df['error'] = (eval_df['pred'] != eval_df['label']).astype(float)
    eval_df['correct'] = 1 - eval_df['error']
    eval_df['correct_1'] = eval_df['correct'] * (eval_df['pred'] == 1)
    eval_df['yoy_rating_diff'] = eval_df['label_rating'] - eval_df['last_year_rating']
    eval_df['last_year_rating_bin'] = eval_df['last_year_rating'].apply(lambda x: (x // 0.05) * 0.05)
    eval_df['abs_error'] = eval_df['yoy_rating_diff'].abs()

    # Creating binned attributes - YoY diff
    yoy_diff_cats = [number_to_cat_bins(d, cats_thresholds=yoy_diff2cats_thresholds) for d in ratings_yoy_diffs]
    eval_df['yoy_diff_cat'] = yoy_diff_cats
    eval_df['yoy_diff_cat_name'] = [YOY_DIFFS_TO_CATS_NAMES[x] for x in yoy_diff_cats]

    # Rating
    ratings_bins = RATING_BINS[:]  # [60, 70, 75, 80, 85, 90, 100]
    ratings_bins_names = RATING_BINS_NAMES[:]  # ['<60', '60-70', '70-75', '76-80', '81-85', '85-90', '90+']
    last_y_ratings_cats = [number_to_cat_bins(r * 100, cats_thresholds=ratings_bins) for r in ratings_y1_back]
    last_y_ratings_cats_names = [ratings_bins_names[x] for x in last_y_ratings_cats]
    eval_df['last_year_rating_cat'] = last_y_ratings_cats
    eval_df['last_year_rating_cat_name'] = last_y_ratings_cats_names
    ratings_cats = [number_to_cat_bins(r * 100, cats_thresholds=ratings_bins) for r in rating_labels]
    ratings_cats_names = [ratings_bins_names[x] for x in ratings_cats]
    eval_df['rating_label_bin'] = eval_df['label_rating'].apply(lambda x: (x // 0.05) * 0.05)
    eval_df['rating_cat'] = ratings_cats
    eval_df['rating_cat_name'] = ratings_cats_names

    # Age
    age_bins = AGE_TO_CATS[:]  # [18, 21, 25, 29, 30, 35]
    age_bins_names = AGE_BINS_NAMES[:]
    age_cats = [number_to_cat_bins(a * AGE_NORM_VALUE, cats_thresholds=age_bins) for a in players_age]
    age_cats_name = [age_bins_names[x] for x in age_cats]
    eval_df['age_cat'] = age_cats
    eval_df['age_cat_name'] = age_cats_name

    return eval_df


def generate_bar_and_hist(eval_df: pd.DataFrame,
                          x: str,
                          y_variables: List[str],
                          model_name: str = '',
                          min_count: int = 0,
                          row_heights: tuple = (0.75, 0.25),
                          output_dir: str = os.path.join(ArtifactsPaths.ARTIFACTS_DIR, 'reports'),
                          text: Optional[Literal['y', 'count', 'y+count']] = None,
                          export_plots: bool = False,
                          show_plots: bool = False):
    """ Util function to generate aggregated bar plot + histogram of values below it """
    fig = make_subplots(rows=2, cols=1,
                        row_heights=list(row_heights),
                        shared_xaxes=True
                        )

    for y in y_variables:
        agg_df = eval_df[[x, y]].groupby(x).agg(['mean', 'count']).reset_index()
        agg_df.columns = [x, f'avg_{y}', 'count']
        agg_df = agg_df[agg_df['count'] > min_count]
        fig.add_trace(go.Bar(
            x=agg_df[x],
            y=agg_df[f'avg_{y}'],
            hovertext=agg_df['count'],
            text=agg_df[f'avg_{y}'].round(2) if 'y' in text else None,
            name=f"Average {y} by {x}",
        ),
            row=1, col=1)
        fig.add_trace(go.Bar(
            x=agg_df[x],
            y=agg_df['count'],
            text=agg_df['count'].round(2) if 'count' in text else None,
            textposition='outside',
            marker=dict(color='grey'),
            name=f"{x.capitalize()} Distribution"
        ), row=2, col=1)

    short_model_name = get_model_short_name(model_name)
    fig.update_layout(
        template="plotly_white",
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        title=f'Average {y} by {x}: {short_model_name}',
        barmode="group",
        barcornerradius=15,
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
    )
    fig.update_yaxes(title_text=f"{y.capitalize().replace('_', ' ')}", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text=f"{x.capitalize().replace('_', ' ')}", row=2, col=1)
    if text is not None:
        fig.update_traces(textposition='auto')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    if export_plots:
        short_model_name = get_model_short_name(model_name)
        fig.write_html(os.path.join(output_dir, f'{short_model_name}_average_{y}_by_{x}.html'))
    if show_plots:
        fig.show()


def top_bottom_predictions_evaluation(eval_df: pd.DataFrame,
                                      model_name: str,
                                      metrics: Dict,
                                      output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                                      show_plots: bool = False,
                                      export_plots: bool = False,
                                      prints: bool = True,
                                      prefix: str = 'top_'
                                      ) -> Dict:
    sorted_prob_df = eval_df.sort_values('prob', ascending=False)
    short_model_name = get_model_short_name(model_name)
    fig_values = []
    # Define the top/bottom values
    n_values = [10, 20, 50, 100, 250, 500, 750, 1000]
    for n in n_values:
        top_n_prob_df = pd.concat([sorted_prob_df.head(n), sorted_prob_df.tail(n)], axis=0)
        if prints:
            print(f'\nTop/Bottom n={n} classification report')
            print(classification_report(top_n_prob_df['label'], top_n_prob_df['pred'], zero_division=0))
        metrics = cls_metrics(top_n_prob_df['pred'].tolist(),
                              top_n_prob_df['label'].tolist(),
                              top_n_prob_df['prob'].tolist(),
                              prefix=f'{prefix}{n}_',
                              model_name=model_name,
                              metrics=metrics,
                              prints=prints
                              )
        fig_values.append({
            'n': n,
            'precision': metrics[f'top_{n}_class_1_precision'],
            'recall': metrics[f'top_{n}_class_1_recall'],
            'counts': top_n_prob_df['label'].value_counts(normalize=True).to_dict(),
            'cls_1_precision': round(metrics[f'top_{n}_class_1_precision'], 2),
            'cls_1_recall': round(metrics[f'top_{n}_class_1_recall'], 2),
            'cls_1_count': metrics[f'top_{n}_class_1_support'],
            'cls_0_precision': round(metrics[f'top_{n}_class_0_precision'], 2),
            'cls_0_recall': round(metrics[f'top_{n}_class_0_recall'], 2),
            'cls_0_count': metrics[f'top_{n}_class_0_support'],
        })

    fig_df = pd.DataFrame(fig_values)
    fig_df['count_1'] = fig_df['counts'].apply(lambda x: x[1])
    fig_df['count_0'] = fig_df['counts'].apply(lambda x: x[0])
    fig_df.to_csv(os.path.join(output_dir, f'{short_model_name}_top_bottom_n_evaluation.csv'))

    # Plot - top / bottom K precision & recall curves across different Ks
    top_p_fig = go.Figure()
    top_p_fig.add_trace(go.Line(
        x=fig_df['n'],
        y=fig_df['precision'],
        name='precision',
        mode='lines+markers',
        hovertemplate=
        '<b>Precisions</b><br>' +
        '%{text}',
        text=[f"class 0: {fig_values[n]['cls_0_precision']}<br>class 1: {fig_values[n]['cls_1_precision']}"
              for n in range(len(n_values))],
    ))
    top_p_fig.add_trace(go.Scatter(
        x=fig_df['n'],
        y=fig_df['recall'],
        name='recall',
        mode='lines+markers',
        hovertemplate=
        '<b>Recalls</b><br>' +
        '<b>%{text}</b>',
        text=[
            f"class 0: {fig_values[n]['cls_0_recall']}<br>class 1: {fig_values[n]['cls_1_recall']}"
            for n in range(len(n_values))],
    ))
    top_p_fig.update_layout(
        template="plotly_white",
        title=f'Top/bottom N players by probability: {prefix} {short_model_name}',
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        xaxis_title="Num of top players (top+bottom)",
        hovermode='x'
    )
    if export_plots:
        top_p_fig.write_html(os.path.join(output_dir, f'{short_model_name}_top_p_fig.html'))
    if show_plots:
        top_p_fig.show()
    return metrics


def evaluate_players_of_interest(preds: List[int],
                                 probs: List[float],
                                 labels: List[int],
                                 ratings_y1_back: List[float],
                                 rating_labels: List[float],
                                 ratings_yoy_diffs: List[float],
                                 players_names,
                                 output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                                 model_name: str = 'model',
                                 prints: bool = True,
                                 show_plots: bool = True,
                                 targeted_players: Optional[List[str]] = None,
                                 players_collection_name: str = 'Players of interest',
                                 players_text: Optional[Literal['all', 'selected', 'exact']] = 'selected',
                                 players_to_label: Optional[List[str]] = None
                                 ) -> dict:
    if targeted_players is None:
        targeted_players = PLAYERS_OF_INTEREST[:]

    vals = []
    players_covered = defaultdict(bool)
    for player_name in targeted_players:
        try:
            player_idx = players_names.index(player_name)
            players_covered[player_name] = True
        except ValueError:
            players_covered[player_name] = False
            continue
        vals.append(dict(player=player_name,
                         pred=preds[player_idx],
                         prob=probs[player_idx],
                         label=labels[player_idx],
                         yoy_diff=ratings_yoy_diffs[player_idx],
                         rating_label=rating_labels[player_idx],
                         ratings_y1_back=ratings_y1_back[player_idx]
                         )
                    )
    players_df = pd.DataFrame(vals).round(2)
    players_collection_name_proc = players_collection_name.lower().replace(' ', '_')
    players_df.to_csv(os.path.join(output_dir, f'{model_name}_{players_collection_name_proc}.csv'))
    if prints:
        print(f'\nEvaluation {players_collection_name}')
        print(f'\n{players_collection_name} label distribution:')
        print(players_df['label'].value_counts(), players_df['label'].value_counts(normalize=True))
        print(f'\n{players_collection_name} pred distribution:')
        print(players_df['pred'].value_counts(), players_df['pred'].value_counts(normalize=True))
        # Confusion matrix
        try:
            cm_norm = confusion_matrix(players_df['label'], players_df['pred'], normalize='true')
            cm_norm = pd.DataFrame(cm_norm, columns=['pred 0', 'pred 1'], index=['true 0', 'true 1']).round(2)
            print(f'\n{players_collection_name} confusion matrix:')
            print(cm_norm)
        except:
            print(f'Could not calculate confusion matrix '
                  f'for model {get_model_short_name(model_name)} > {players_collection_name}')
        print(f'Covering {len(players_covered)} players')
        print('- Players missing:', [p for p in players_covered if not players_covered[p]])
        # Classification report
        print(f'\n{players_collection_name} classification_report')
        print(classification_report(players_df['label'], players_df['pred']))

    cls_report = classification_report(players_df['label'], players_df['pred'], output_dict=True)
    players_metrics = {}
    for cls in ['0', '1']:
        if cls not in cls_report:
            # all players have the same label
            players_metrics[f'class_{cls}_recall'] = None
            players_metrics[f'class_{cls}_precision'] = None
            players_metrics[f'class_{cls}_f1'] = None
            players_metrics[f'class_{cls}_support'] = 0
            continue
        players_metrics[f'class_{cls}_recall'] = cls_report[cls]['recall']
        players_metrics[f'class_{cls}_precision'] = cls_report[cls]['precision']
        players_metrics[f'class_{cls}_f1'] = cls_report[cls]['f1-score']
        players_metrics[f'class_{cls}_support'] = cls_report[cls]['support']

    if show_plots:
        # Scatter of prob, color is label, y is yoy diff
        # Define which players to label
        if players_text == 'all':
            players_to_label = targeted_players[:]
        elif players_text == 'selected':
            players_to_label = players_df.loc[(players_df['yoy_diff'].abs() >= 0.3), 'player'].tolist()
        elif players_text is None:
            players_to_label = []
        else:
            # 'exact' -> players_to_label = players_to_label
            assert players_to_label is not None, \
                "When using players_text == 'exact', you must supply players_to_label: List[str]"

        def get_player_label(_name):
            if _name not in players_to_label:
                return ''
            if _name in PLAYERS_NAMES_CONVERTER:
                return PLAYERS_NAMES_CONVERTER[_name]
            else:
                return f"{_name.split(' ')[0].capitalize()}. {_name.split(' ')[-1]}"

        # Create a new column with text only for selected players
        players_df['text_label'] = players_df['player'].apply(lambda x: get_player_label(x))
        players_df['color'] = players_df[['pred', 'label']].apply(lambda x: 'green' if x[0] == x[1] else 'red', axis=1)

        # Fig by prob and yoy diff
        fig = go.Figure()
        min_rating, max_rating = players_df['ratings_y1_back'].min(), players_df['ratings_y1_back'].max()
        fig.add_trace(go.Scatter(
            x=players_df['prob'].round(2),
            y=players_df['yoy_diff'].round(2),
            marker=dict(color=players_df['color'],
                        size=(
                            players_df['ratings_y1_back'].apply(
                                lambda x: 7 + 30 * ((x - min_rating) / (max_rating - min_rating))
                            )))
            ,
            mode='markers+text',
            text=players_df['text_label'],
            textposition='top center',
            hovertext=players_df.apply(lambda x: f"{x['player']}"
                                                 f"<br>Probability={x['prob']};"
                                                 f"<br>rating={x['ratings_y1_back']} -> "
                                                 f"{round(x['ratings_y1_back'] + x['yoy_diff'], 2)}"
                                                 f"({'+' if x['yoy_diff'] > 0 else '-'}{int(100 * x['yoy_diff'])})",
                                       axis=1).tolist(),
        ))
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="grey")
        fig.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="grey")
        short_model_name = get_model_short_name(model_name)
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"{players_collection_name} {short_model_name}",
                subtitle=dict(
                    text=f"Marker size = player rating, color = correct prediction"
                         f"<br>Accuracy = {round(cls_report['accuracy'], 2)}, "
                         f"Precision = {round(players_metrics['class_1_precision'], 2)}, "
                         f"Recall = {round(players_metrics['class_1_recall'], 2)}",
                    font=dict(color="gray", size=13),
                ),
            ),
            xaxis_title="Probability",
            yaxis_title="YoY change in FIFA rating"
        )
        fig.write_html(os.path.join(output_dir,
                                    f'{short_model_name}_{players_collection_name.lower().replace(' ', '_')}_by_prob_and_yoy.html'))
        fig.show()

    return players_metrics


def yoy_distribution_by_correctness(eval_df: pd.DataFrame,
                                    model_name: str,
                                    x_col: str = 'yoy_diff',
                                    xaxis_categories_order: Optional[Dict] = None,
                                    output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                                    show_plots: bool = False,
                                    export_plots: bool = False
                                    ):
    data = []
    data_dict = {}
    for col in ['correct', 'error']:
        agg_df = eval_df.round(2).loc[eval_df[col] == 1, [x_col, col]].groupby(x_col).count().reset_index().round(2)
        agg_df.columns = [x_col, col]
        data_dict[col] = agg_df.set_index(x_col)[col].round(2).to_dict()
        agg_df[f"normed_{col}"] = agg_df[col] / agg_df[col].sum()
        data.append(go.Bar(x=agg_df[x_col], y=agg_df[col],
                           name=col.capitalize(),
                           text=agg_df[f"normed_{col}"].round(2)))
        agg_df[f"normed_{col}"] = agg_df[col] / agg_df[col].sum()

    # Calculate the lift
    correct_over_error = {}
    for _diff in data_dict['correct']:
        correct_over_error[_diff] = data_dict['correct'].get(_diff, 0) / max(data_dict['error'].get(_diff, 1), 1)
    correct_over_error = pd.DataFrame.from_dict(correct_over_error, orient='index').round(2).reset_index()
    correct_over_error.columns = [x_col, 'correct_lift']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for _plt in data:
        fig.add_trace(_plt, secondary_y=False)

    fig.update_layout(xaxis=xaxis_categories_order, )
    fig.add_hline(y=1, line_width=3, line_dash="dash", line_color="grey", secondary_y=True, name='Lift = 1')
    fig.add_trace(go.Scatter(x=correct_over_error[x_col],
                             y=correct_over_error['correct_lift'],
                             name='Correct lift',
                             marker=dict(color='green', size=10, symbol='cross'),
                             mode='markers'
                             ),
                  secondary_y=True)

    short_model_name = get_model_short_name(model_name)
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"Predictions Distribution by YoY FIFA Rating Change: {short_model_name}",
            subtitle=dict(
                text=f"<br>In-bar text is its share (%) out of all errors / correct predictions",
                font=dict(color="gray", size=13),
            ),
        ),
        xaxis=dict(title=x_col.title().replace('_', ' ')),
        yaxis=dict(title="Count"),
        barmode="group",
        barcornerradius=15,
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
    )
    fig.update_yaxes(title_text="Correct <b>lift</b>", secondary_y=True)
    fig.update_layout(xaxis=xaxis_categories_order)
    if export_plots:
        short_model_name = get_model_short_name(model_name)
        fig.write_html(os.path.join(output_dir, f'{short_model_name}_correct_lift_by_yoy_cat.html'))
    if show_plots:
        fig.show()


def yoy_evaluation(eval_df: pd.DataFrame,
                   metrics: Dict,
                   model_name: str,
                   output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                   show_plots: bool = False,
                   export_plots: bool = False,
                   prints: bool = True
                   ) -> Dict:
    # Analyze performance latest YoY change of rating - do results affected by the magnitude of YoY growth?
    # Transform the yoy diffs into categories of change (by magnitude)
    cm_ex = np.round(confusion_matrix(eval_df['yoy_diff_cat'], eval_df['pred']), 2)
    cm_norm_ex = np.round(confusion_matrix(eval_df['yoy_diff_cat'], eval_df['pred'], normalize='true'), 2)
    cm_norm_ex_df = pd.DataFrame(cm_norm_ex, index=YOY_DIFFS_TO_CATS_NAMES[:]).iloc[:, :2]
    cm_norm_ex_df.columns = [0, 1]
    cm_ex_df = pd.DataFrame(cm_ex, index=YOY_DIFFS_TO_CATS_NAMES[:]).iloc[:, :2]
    cm_ex_df.columns = [0, 1]
    if prints:
        print(f'\n{model_name} Confusion matrix extended\n{cm_ex_df.T}')
        print(f'\nNormed version:\n{cm_norm_ex_df.T}')

    # Plot error rate by yoy_diff
    # Yoy diff distribution for correct preds
    if show_plots or export_plots:
        yoy_distribution_by_correctness(eval_df,
                                        model_name,
                                        x_col='yoy_diff_cat_name',
                                        xaxis_categories_order={'categoryorder': 'array',
                                                                'categoryarray': YOY_DIFFS_TO_CATS_NAMES[:]},
                                        output_dir=output_dir,
                                        show_plots=show_plots,
                                        export_plots=export_plots
                                        )

    # Aggregate results by yoy diff, yoy diff cat, and when pred = 1
    yoy_df_agg_cat = eval_df[['yoy_diff_cat_name', 'correct']].groupby(['yoy_diff_cat_name']).agg(
        ['mean', 'count']).reset_index()
    yoy_df_agg_cat.columns = ['yoy_diff_cat_name', 'accuracy', 'count']

    # Positive pred cases -> precision
    yoy_df_agg = eval_df[['yoy_diff_cat_name', 'pred']].round(2).groupby(['yoy_diff_cat_name']).count().reset_index()
    yoy_df_agg.columns = ['yoy_diff_cat_name', 'count']
    yoy_df_agg_cat_pos_pred = eval_df.loc[eval_df['pred'] == 1, ['yoy_diff_cat_name', 'correct']].round(2).groupby(
        ['yoy_diff_cat_name']).agg(
        ['mean', 'count']).reset_index()
    yoy_df_agg_cat_pos_pred.columns = ['yoy_diff_cat_name', 'precision', 'count']

    # Positive cases -> recall
    yoy_df_agg_cat_pos_label = eval_df.loc[eval_df['label'] == 1, ['yoy_diff_cat_name', 'correct']].round(2).groupby(
        ['yoy_diff_cat_name']).agg(
        ['mean', 'count']).reset_index()
    yoy_df_agg_cat_pos_label.columns = ['yoy_diff_cat_name', 'recall', 'count']

    if prints:
        print('-' * 250)
        print('\nCorrelation of correct rate - YoY change in FIFA rating - prob')
        print(eval_df[['yoy_diff', 'correct', 'prob']].corr().round(3))

    # Box plot - prob per yoy cat
    agg_df = eval_df[['yoy_diff_cat_name', 'prob', 'correct']].groupby('yoy_diff_cat_name').agg(
        ['mean', 'count']).reset_index()
    agg_df.columns = ['yoy_diff_cat_name', 'avg_prob', 'count', 'correct_rate', '_']
    box_yoy_diff_prob = make_subplots(rows=2, cols=1,
                                      row_heights=[0.7, 0.3],
                                      shared_xaxes=True
                                      )
    box_yoy_diff_prob.add_trace(go.Box(
        x=eval_df['yoy_diff_cat_name'],
        y=eval_df['prob'],
        name='probability',
    ), row=1, col=1)
    box_yoy_diff_prob.add_trace(go.Bar(
        x=agg_df['yoy_diff_cat_name'],
        y=agg_df['count'],
        name='count',
    ), row=2, col=1)
    short_model_name = get_model_short_name(model_name)
    box_yoy_diff_prob.update_layout(
        template="plotly_white",
        title=dict(text=f'Prob Distribution by YoY Change in FIFA Rating: {short_model_name}',
                   font=dict(size=Font.TITLE_SIZE)),
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        xaxis_title="YoY change FIFA rating bin",
        xaxis={'categoryorder': 'array',
               'categoryarray': YOY_DIFFS_TO_CATS_NAMES[:]},
        barmode="group",
        barcornerradius=10,
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
    )
    box_yoy_diff_prob.update_traces(marker_color='grey', opacity=0.85, row=2, col=1)
    box_yoy_diff_prob.update_yaxes(title_text="probability", row=1, col=1)
    box_yoy_diff_prob.update_xaxes(title_text="YoY Change Bin", row=2, col=1)
    box_yoy_diff_prob.update_yaxes(title_text="Count", row=2, col=1)
    if export_plots:
        box_yoy_diff_prob.write_html(os.path.join(output_dir,
                                                  f'{get_model_short_name(model_name)}_prob_distribution_by_yoy.html'))
    if show_plots:
        box_yoy_diff_prob.show()

    metrics['cm_norm_extended'] = cm_norm_ex
    return metrics


def evaluation_by_player_attributes(eval_df: pd.DataFrame,
                                    model_name: str,
                                    output_dir: str = ArtifactsPaths.ARTIFACTS_DIR,
                                    show_plots: bool = False,
                                    export_plots: bool = False,
                                    ):
    print('Correlation between age and correct rate')
    print(eval_df[['player_age', 'correct', 'prob']].corr(method='spearman').round(3))

    print('\nCorrelation between rating and correct rate')
    print(eval_df[['last_year_rating', 'correct', 'prob']].corr(method='spearman').round(3))

    print('\nYoY std across age bins')
    print(eval_df[['age_cat_name', 'yoy_diff']].groupby('age_cat_name').std())
    print('\nYoY std across rating bins')
    print(eval_df[['rating_cat_name', 'yoy_diff']].groupby('rating_cat_name').std())

    print('\nYoY std across age bins')
    years_data_by_age = eval_df[['age_cat_name', 'num_years_data']].groupby('age_cat_name').agg(
        [np.mean, np.std]).reset_index()
    years_data_by_age.columns = ['age_cat_name', 'mean', 'std']
    print(years_data_by_age)

    # Aggregation of performance by age
    age_df = eval_df[['age_cat_name', 'correct', 'yoy_diff']].groupby('age_cat_name').agg(
        ['mean', 'count', 'std']).reset_index()
    age_df.columns = ['age_cat_name', 'accuracy', 'count', '_', 'yoy_diff_avg', '__', 'yoy_diff_std']
    pos_preds_by_age = eval_df.loc[eval_df['pred'] == 1, ['age_cat_name', 'correct']] \
        .groupby('age_cat_name').count().reset_index()
    pos_preds_by_age.columns = ['age_cat_name', 'count']

    # Performance by player age
    perform_by_age = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                                   specs=[[{"secondary_y": True}],  # top row has two y-axes
                                          [{}]]  # bottom row has one y-axis
                                   )
    marker_color = '#3c6141'  # green-grey
    main_bar_color = '#636EFA'
    bar_hist_color = 'grey'
    pos_hist_color = '#3c6141'  # green-grey
    marker_shape, marker_size = 'octagon-dot', 10
    perform_by_age.add_trace(go.Bar(
        x=age_df['age_cat_name'],
        y=age_df['accuracy'],
        text=age_df['accuracy'].round(2),
        marker=dict(color=main_bar_color),
        name='Accuracy',
    ), col=1, row=1, secondary_y=False)
    # Fill missing bins with None
    full_bins = AGE_BINS_NAMES[:]  # ensure this matches your category order exactly
    yoy_std_dict = dict(zip(age_df['age_cat_name'], age_df['yoy_diff_std']))
    yoy_std_vals = [yoy_std_dict.get(bin_name, None) for bin_name in full_bins]
    text_vals = [f"{val:.2f}" if val is not None else "" for val in yoy_std_vals]
    perform_by_age.add_trace(go.Scatter(
        x=full_bins,
        y=yoy_std_vals,
        text=text_vals,
        line=dict(color=marker_color, width=3, dash='dash'),
        marker=dict(color=marker_color, symbol=marker_shape, size=marker_size),
        mode='markers+lines',
        name='YoY rating change Std',
        connectgaps=False
    ), col=1, row=1, secondary_y=True)
    # Bottom chart - distribution
    perform_by_age.add_trace(go.Bar(
        x=age_df['age_cat_name'],
        y=age_df['count'],
        text=age_df['count'],
        name='Count',
        marker=dict(color=bar_hist_color),
    ), row=2, col=1)
    perform_by_age.add_trace(go.Bar(
        x=pos_preds_by_age['age_cat_name'],
        y=pos_preds_by_age['count'],
        text=pos_preds_by_age['count'].round(2),
        marker=dict(color=pos_hist_color),
        name='Pred = 1 count'
    ), row=2, col=1)
    short_model_name = get_model_short_name(model_name)
    perform_by_age.update_layout(
        template="plotly_white",
        title=dict(
            text=f'Performance by Player Age: {short_model_name}', font=dict(size=Font.TITLE_SIZE),
            subtitle=dict(
                text=f"Main figure: Model accuracy by players age bin (bars) and YoY rating change std (scatter)"
                     f"<br>Bottom figure: Age distribution of players and of positive predictions",
                font=dict(color="gray", size=13),
            )),
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        barmode="group",
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
        barcornerradius=10,
        xaxis={'categoryorder': 'array',
               'categoryarray': AGE_BINS_NAMES[:]},

    )
    # perform_by_age.update_traces(textposition='outside', row=1, col=1)
    perform_by_age.update_traces(opacity=0.85, row=2, col=1)
    perform_by_age.update_xaxes(title_text="Player age bin", row=1, col=1,
                                type='category', categoryorder='array',
                                categoryarray=AGE_BINS_NAMES[:]
                                )
    perform_by_age.update_yaxes(title_text="Accuracy", row=1, col=1, secondary_y=False)
    perform_by_age.update_yaxes(title_text="YoY rating change", row=1, col=1, secondary_y=True,
                                title_font=dict(color=marker_color), tickfont=dict(color=marker_color))
    perform_by_age.update_yaxes(title_text="Count", row=2, col=1)

    if export_plots:
        perform_by_age.write_html(os.path.join(output_dir, f'{short_model_name}_perform_by_age.html'))
    if show_plots:
        perform_by_age.show()

    # Performance by player rating
    ratings_df = eval_df[['last_year_rating_cat_name', 'correct']].groupby(
        'last_year_rating_cat_name').agg(['mean', 'count']).reset_index()
    ratings_df.columns = ['last_year_rating_cat_name', 'accuracy', 'count']
    # Focus only on positive predictions
    pos_preds_by_rating = (eval_df.loc[eval_df['pred'] == 1, ['last_year_rating_cat_name', 'pred']] \
                           .groupby('last_year_rating_cat_name')
                           .agg(['mean', 'count']).reset_index())
    pos_preds_by_rating.columns = ['last_year_rating_cat_name', 'avg_pred', 'count']

    # Calculate data coverage by rating
    years_data_by_rating = eval_df[['last_year_rating_cat_name', 'num_years_data']] \
        .groupby('last_year_rating_cat_name').agg(
        [np.mean, np.std]).reset_index()
    years_data_by_rating.columns = ['last_year_rating_cat_name', 'mean', 'std']
    # Convert coverage to (0-1)
    max_num_years_back = eval_df['num_years_data'].max()
    years_data_by_rating['mean'] /= max_num_years_back

    # Show performance by overall rating bins
    performance_by_rating = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
    performance_by_rating.add_trace(go.Bar(
        x=ratings_df['last_year_rating_cat_name'],
        y=ratings_df['accuracy'],
        name='Accuracy',
        text=ratings_df['accuracy'].round(2),
        marker=dict(color=main_bar_color),
        textposition='inside',
    ), col=1, row=1)
    # Fill missing bins with None
    full_bins = RATING_BINS_NAMES[:]  # ensure this matches your category order exactly
    data_horizon_dict = dict(zip(years_data_by_rating['last_year_rating_cat_name'], years_data_by_rating['mean']))
    data_horizon_vals = [data_horizon_dict.get(bin_name, None) for bin_name in full_bins]
    text_vals = [f"{val:.2f}" if val is not None else "" for val in data_horizon_vals]

    performance_by_rating.add_trace(go.Scatter(
        x=full_bins,
        y=data_horizon_vals,
        text=text_vals,
        name='Data horizon coverage (%)',
        marker=dict(color=marker_color, symbol=marker_shape, size=marker_size),
        line=dict(color=marker_color, width=3, dash='dash'),
        mode='markers+lines',
        connectgaps=False
    ), col=1, row=1)
    # Bottom chart - distribution
    # 1. Distribution of all predictions
    performance_by_rating.add_trace(go.Bar(
        x=ratings_df['last_year_rating_cat_name'],
        y=ratings_df['count'],
        text=age_df['count'],
        marker=dict(color=bar_hist_color),
        name='Count'
    ), row=2, col=1)
    # 2. Distribution of positive predictions
    performance_by_rating.add_trace(go.Bar(
        x=pos_preds_by_rating['last_year_rating_cat_name'],
        y=pos_preds_by_rating['count'],
        text=pos_preds_by_rating['count'].round(2),
        marker=dict(color=pos_hist_color),
        name='Pred = 1 count'
    ), row=2, col=1)
    short_model_name = get_model_short_name(model_name)
    performance_by_rating.update_layout(
        template="plotly_white",
        title=dict(text=f'Performance vs Player Rating Bin: {short_model_name}',
                   font=dict(size=Font.TITLE_SIZE),
                   subtitle=dict(
                       text=f"Main figure: Accuracy by FIFA rating bin (bars), and data horizon (1-{3} years, normed) coverage (scatter)"
                            f"<br>Bottom figure: FIFA ratings distribution of players and positive predictions",
                       font=dict(color="gray", size=13),
                   ),
                   ),
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        xaxis_title="Player rating bin",
        xaxis={'categoryorder': 'array', 'categoryarray': RATING_BINS_NAMES[:]},
        barmode="group",
        barcornerradius=10,
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.)
    )
    performance_by_rating.update_traces(opacity=0.85, row=2, col=1)
    performance_by_rating.update_xaxes(title_text="Player Rating Bin", row=2, col=1)
    performance_by_rating.update_yaxes(title_text="%", row=1, col=1)
    performance_by_rating.update_yaxes(title_text="Count", row=2, col=1)

    if export_plots:
        performance_by_rating.write_html(os.path.join(output_dir, f'{short_model_name}_performance_by_rating.html'))
    if show_plots:
        performance_by_rating.show()


def model_evaluation(
        labels: List[int],
        preds: List[int],
        probs: List[float],
        players_ratings: List[float] = None,
        rating_labels: List[float] = None,
        ratings_yoy_diffs: List[float] = None,
        players_names: List[str] = None,
        players_age: List[str] = None,
        num_years_data: List[float] = None,
        output_dir: str = os.path.join(ArtifactsPaths.ARTIFACTS_DIR, 'reports'),
        model_name: str = 'model',
        show_plots: bool = False,
        export_plots: bool = False,
        prints: bool = True,
):
    """
    model evaluation for StatPredict
    :param x_test: pandas dataframe of test set features
    :param labels: list of binary true values
    :param preds: list of binary predications
    :param probs: list of float predications probabilities
    :param players_ratings: list of last year ratings [0-1] (ordered same as labels)
    :param rating_labels: list of label year ratings [0-1] (ordered same as labels)
    :param players_names: list of player names
    :param num_years_data: each item holds the number of years (without padding) the player had
    :param output_dir: directory to store outputs
    :param players_age: list of player ages
    :param ratings_yoy_diffs: list of YoY change of label (= rating_labels - ratings_y1_back)
    :param model_name: name of evaluated model
    :param show_plots: bool = False - whether to show Plotly figures or not
    :param export_plots: bool = False, whether to save (selected) Plotly figures as html of not
    :param prints: bool = False, whether to use print or not
    :return:
    """
    metrics = {}
    if prints:
        print(f'\n{model_name} classification report')
        print(classification_report(labels, preds, zero_division=0))

    metrics = cls_metrics(preds, labels, probs,
                          prefix='',
                          model_name=model_name,
                          metrics=metrics,
                          conf_matrix=True,
                          prints=prints)

    # Build analysis dataframe
    eval_df = build_eval_df(labels,
                            preds,
                            probs,
                            players_ratings,
                            rating_labels,
                            ratings_yoy_diffs,
                            players_names,
                            players_age,
                            num_years_data,
                            yoy_diff2cats_thresholds=YOY_DIFFS_TO_CATS
                            )
    if export_plots:
        eval_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'))

    # Accuracy / precision by prob - percentiles
    generate_bar_and_hist(eval_df,
                          x='prob_decile',
                          y_variables=['correct'],
                          text='y+count',
                          model_name=model_name,
                          min_count=5,
                          export_plots=export_plots,
                          show_plots=show_plots,
                          output_dir=output_dir
                          )

    # Performance on top and bottom probs
    metrics = top_bottom_predictions_evaluation(eval_df,
                                                model_name,
                                                metrics,
                                                output_dir=output_dir,
                                                show_plots=False,
                                                export_plots=export_plots,
                                                prints=prints,
                                                prefix='top_'
                                                )

    # Players of interest analysis - shortlist, top growers, top decliners, young
    metrics['players_of_interest'] = evaluate_players_of_interest(preds,
                                                                  probs,
                                                                  labels,
                                                                  players_ratings,
                                                                  rating_labels,
                                                                  ratings_yoy_diffs,
                                                                  players_names,
                                                                  output_dir=output_dir,
                                                                  model_name=model_name,
                                                                  prints=prints,
                                                                  show_plots=show_plots,
                                                                  players_text='exact',
                                                                  players_to_label=SELECTED_PLAYERS_OF_INTEREST[:]
                                                                  )

    # EA-FC 24 future stars list
    metrics['ea_fc24_future_stars'] = evaluate_players_of_interest(preds,
                                                                   probs,
                                                                   labels,
                                                                   players_ratings,
                                                                   rating_labels,
                                                                   ratings_yoy_diffs,
                                                                   players_names,
                                                                   output_dir=output_dir,
                                                                   model_name=model_name,
                                                                   prints=prints,
                                                                   show_plots=show_plots,
                                                                   players_text='exact',
                                                                   players_collection_name='2024 Future Stars',
                                                                   targeted_players=FIFA24_FUTURE_STARS[:],
                                                                   players_to_label=FIFA24_FUTURE_STARS[:]
                                                                   )

    under_19 = eval_df.loc[(eval_df['player_age'] <= 19 / AGE_NORM_VALUE), 'player'].tolist()
    under_19 = [p for p in under_19 if p not in FIFA24_FUTURE_STARS]
    metrics['under_19_players'] = evaluate_players_of_interest(preds,
                                                               probs,
                                                               labels,
                                                               players_ratings,
                                                               rating_labels,
                                                               ratings_yoy_diffs,
                                                               players_names,
                                                               output_dir=output_dir,
                                                               model_name=model_name,
                                                               prints=prints,
                                                               show_plots=show_plots,
                                                               targeted_players=under_19,
                                                               players_collection_name='Under 19 players',
                                                               players_text='exact',
                                                               players_to_label=SELECTED_PLAYERS_U21[:]
                                                               )

    # Evaluate performance by YoY growth
    yoy_evaluation(eval_df,
                   metrics,
                   model_name,
                   output_dir=output_dir,
                   show_plots=show_plots,
                   export_plots=export_plots,
                   prints=prints)

    # Analyze model performance by the player absolute score (ranges of rantings)
    evaluation_by_player_attributes(eval_df,
                                    model_name,
                                    output_dir=output_dir,
                                    show_plots=show_plots,
                                    export_plots=export_plots)

    # Show calibration curve
    plot_calibration_curve(probs, labels,
                           model_name=model_name,
                           output_dir=output_dir,
                           show_plots=show_plots,
                           export_plots=export_plots)
    return metrics


def build_final_report(run_report: dict) -> dict:
    final_report = {k: v for k, v in run_report.items() if k in
                    ['auc',
                     'class_0_precision',
                     'class_0_recall',
                     'class_1_precision',
                     'class_1_recall',
                     'class_1_f1',
                     'top_1000_class_0_precision',
                     'top_1000_class_0_recall',
                     'top_1000_class_1_precision',
                     'top_1000_class_1_recall',
                     'top_1000_auc',
                     'top_750_class_0_precision',
                     'top_750_class_0_recall',
                     'top_750_class_1_precision',
                     'top_750_class_1_recall',
                     'top_750_auc',
                     'top_500_class_0_precision',
                     'top_500_class_0_recall',
                     'top_500_class_1_precision',
                     'top_500_class_1_recall',
                     'top_500_auc',
                     'top_250_class_0_precision',
                     'top_250_class_0_recall',
                     'top_250_class_1_precision',
                     'top_250_class_1_recall',
                     'top_250_auc',
                     'top_100_class_0_precision',
                     'top_100_class_0_recall',
                     'top_100_class_1_precision',
                     'top_100_class_1_recall',
                     'top_100_auc',
                     'top_50_class_0_precision',
                     'top_50_class_0_recall',
                     'top_50_class_1_precision',
                     'top_50_class_1_recall',
                     'top_20_class_0_precision',
                     'top_20_class_0_recall',
                     'top_20_class_1_precision',
                     'top_20_class_1_recall',
                     'top_10_class_0_precision',
                     'top_10_class_0_recall',
                     'top_10_class_1_precision',
                     'top_10_class_1_recall',
                     'cm_norm_extended',
                     'top_10_class_1_f1',
                     'top_10_auc',
                     'top_20_class_1_f1',
                     'top_20_auc',
                     'top_50_class_1_f1',
                     'top_50_auc',
                     'top_100_class_1_f1',
                     'top_100_auc',
                     'top_250_class_1_f1',
                     'top_250_auc',
                     ]
                    }

    for pop in ['players_of_interest', 'ea_fc24_future_stars', 'under_19_players']:
        final_report.update({f'{pop}_{k}': v for k, v in run_report[pop].items()})

    # Recall@Precision
    recall_at_precision_dict = run_report['Recall@Precision'][1]
    for thresh in recall_at_precision_dict:
        final_report[f'Recall@Precision{thresh}'] = recall_at_precision_dict[thresh]['recall']
        final_report[f'Precision@Precision{thresh}'] = recall_at_precision_dict[thresh]['precision']
    return final_report


def multi_model_top_bottom_comparison(models_results: Dict[str, Dict],
                                      output_dir: str = '',
                                      show_auc: bool = True,
                                      n_values: Tuple[int] = (10, 20, 50, 100, 250, 500, 750, 1000),
                                      title: Optional[str] = None):
    subplot_titles = ("Precision", "Recall", "ROC AUC") if  show_auc else ("Precision", "Recall")
    fig = make_subplots(rows=1,
                        cols=3 if show_auc else 2,
                        shared_yaxes=True,
                        horizontal_spacing=0.001,
                        subplot_titles=subplot_titles
                        )
    colors_lst = ['#636EFA', '#EF553B', '#756f71', '#FF6692', '#8C564B', '#7F7F7F', '#2CA02C', '#B82E2E', '#222A2A']
    colors_map = {m: colors_lst[i] for i, m in enumerate(list(models_results.keys()))}
    for model_name, model_metrics in models_results.items():
        model_class_1_df = [{'n': n,
                             'precision': model_metrics[f'top_{n}_class_1_precision'],
                             'recall': model_metrics[f'top_{n}_class_1_recall'],
                             'auc': model_metrics[f'top_{n}_auc'],
                             }
                            for n in n_values]
        model_class_1_df = pd.DataFrame(model_class_1_df).round(3)

        short_model_name = get_model_short_name(model_name)
        fig.add_trace(go.Line(
            x=model_class_1_df['n'],
            y=model_class_1_df['precision'],
            name=short_model_name,
            mode='lines+markers',
            hovertemplate=
            '<b>Precisions</b><br>' +
            '%{y}',
            marker=dict(color=colors_map[model_name])
        ), row=1, col=1)
        fig.add_trace(go.Line(
            x=model_class_1_df['n'],
            y=model_class_1_df['recall'],
            name=None,
            showlegend=False,
            mode='lines+markers',
            hovertemplate=
            '<b>Recalls</b><br>' +
            '%{y}',
            marker=dict(color=colors_map[model_name])
        ), row=1, col=2)
        if show_auc:
            fig.add_trace(go.Line(
                x=model_class_1_df['n'],
                y=model_class_1_df['auc'],
                name=None,
                showlegend=False,
                mode='lines+markers',
                hovertemplate=
                '<b>ROC-AUC</b><br>' +
                '%{y}',
                marker=dict(color=colors_map[model_name])
            ), row=1, col=3)

    title = title if title is not None else 'Models Comparison: Top/bottom N players by probability'
    fig.update_layout(
        template="plotly_white",
        title=title,
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        hovermode='x',
        width=1200, height=650
    )

    # Axes titles
    if show_auc:
        fig.update_xaxes(title_text="N players top+bottom", row=1, col=2)
    else:
        fig.add_annotation(dict(
            text="N players top+bottom",
            x=0.5, y=-0.1, showarrow=False, xref='paper', yref='paper',
            font=dict(size=16)
        ))
    # Remove default y-axis titles
    fig.update_yaxes(title_text='', row=1, col=1)
    fig.update_yaxes(title_text='', row=1, col=2)
    if show_auc:
        fig.update_yaxes(title_text='', row=1, col=3)

    # Move legend below the charts
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Update subplot layout to reduce spacing
    fig.update_layout(margin=dict(t=80, b=80, r=10, l=0))
    fig.write_html(os.path.join(output_dir, 'all_models_top_bottom_n_fig.html'))
    fig.show()


def multi_model_precision_recall_curve(test_preds_dfs: Dict[str, pd.DataFrame],
                                       title: Optional[str] = None,
                                       ):
    """ Generate precision-recall curves for multiple model on same figure """
    pr_curve_fix = go.Figure()
    for model_name in test_preds_dfs:
        short_model_name = get_model_short_name(model_name)
        press, recalls, thresholds = precision_recall_curve(test_preds_dfs[model_name]['labels'],
                                                            test_preds_dfs[model_name]['probs'])
        pr_curve_fix.add_trace(go.Line(x=press, y=recalls, name=short_model_name))

    title = title if title is not None else 'Precision recall curve comparison'
    pr_curve_fix.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=Font.TITLE_SIZE)),
        font_family=Font.FAMILY,
        font_size=Font.LABEL_SIZE,
        xaxis_title="Precision",
        yaxis_title="Recall",
        width=1200, height=800
    )
    pr_curve_fix.show()


def multi_model_roc_auc_curve(test_preds_dfs: Dict[str, pd.DataFrame],
                              title: Optional[str] = None,
                              output_dir: str = ''
                              ):
    """ Generate ROC-AUC curves for multiple model on same figure """
    roc_auc_fig = go.Figure()

    roc_auc_fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    for model_name in test_preds_dfs:
        fpr, tpr, thresholds = metrics.roc_curve(test_preds_dfs[model_name]['labels'],
                                                 test_preds_dfs[model_name]['probs'],
                                                 pos_label=1)
        auc_score = roc_auc_score(test_preds_dfs[model_name]['labels'], test_preds_dfs[model_name]['probs'])

        short_model_name = get_model_short_name(model_name)
        name = f"{short_model_name} (AUC={auc_score:.3f})"
        roc_auc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    title = title if title is not None else 'ROC curve comparison'
    roc_auc_fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=Font.TITLE_SIZE)),
        xaxis=dict(
            title=dict(
                text='False positive rate'
            ),
            constrain='domain'
        ),
        yaxis=dict(
            title=dict(
                text='True positive rate'
            ),
            scaleanchor='x',
            scaleratio=1
        ),
        width=1000, height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    roc_auc_fig.write_html(os.path.join(output_dir, 'multi_roc_auc_curve.html'))
    roc_auc_fig.show()
