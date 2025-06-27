import os.path
import pickle
from typing import Dict
import pandas as pd

from stat_predict.dataset.label import FifaClfLabel
from stat_predict.eval.evaluation import multi_model_precision_recall_curve, multi_model_roc_auc_curve, \
    multi_model_top_bottom_comparison, model_evaluation
from stat_predict.models.experiment import ExperimentArgs, Experiment
from stat_predict.models.dl import DLArgs, DLStatPredict, HistEmbedNetModelArgs
from stat_predict.models.baselines import LRArgs, CatboostArgs, XGBArgs, CatBoostStatPredict, LRStatPredict, \
    XGBStatPredict, TabNetStatPredict, TabNetArgs
from stat_predict.run_experiments import YEARS_BACK
from stat_predict.static.config import ArtifactsPaths

USE_EXISTING_MODELS = True
SHOW_PLOTS = True
EXPORT_FIGURES = True

model2class = {
    'XGBoost': XGBStatPredict,
    'CatBoost': CatBoostStatPredict,
    'LogisticRegression': LRStatPredict,
    'TabNet': TabNetStatPredict,
    'MLPNet': DLStatPredict,
    'HistEmbedNet': DLStatPredict
}

model2args = {
    'XGBoost': XGBArgs,
    'CatBoost': CatboostArgs,
    'LogisticRegression': LRArgs,
    'TabNet': TabNetArgs,
    'MLPNet': DLArgs,
    'HistEmbedNet': HistEmbedNetModelArgs,
}

final_selection = [
    'HistEmbedNet_1-3yback__lr-1e-05_SplitPhaseBy-Age5-Pos5_3y_nc8_knn-ove',
    'CatBoost_1-3yback__nfeatures=125_corr=pearson-1.0_SplitPhaseBy-Age5-Ove5_3y_nc24_knn-ove',
    'LogisticRegression_1-3yback__nfeatures=125_corr=pearson-0.99_SplitPhaseBy-Age5-Pos5_3y_nc8_knn-ove'
]

selected_models = [
    # HistEmbedNet
    {'model_name': 'HistEmbedNet',
     'num_years_back': 3,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },
    {'model_name': 'HistEmbedNet',
     'num_years_back': 4,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },

    # CatBoost
    {'model_name': 'CatBoost',
     'num_years_back': 3,
     'min_years_back': 1,
     'num_features_to_select': 125,
     'correlation_threshold': 1.0,
     'career_phase_model_path': 'Split-by-Age5-Ove5_3y_nc24_knn-ove.pickle'
     },
    {'model_name': 'CatBoost',
     'num_years_back': 4,
     'min_years_back': 1,
     'num_features_to_select': 250,
     'correlation_threshold': 1.0,
     'career_phase_model_path': 'Split-by-Age5-Ove5_3y_nc24_knn-ove.pickle'
     },

    # XGBoost
    {'model_name': 'XGBoost',
     'num_years_back': 3,
     'num_features_to_select': 250,
     'min_years_back': 1,
     'correlation_threshold': 0.99,
     'career_phase_model_path': 'Split-by-Age5-Ove5_3y_nc24_knn-ove.pickle'
     },
    {'model_name': 'XGBoost',
     'num_years_back': 4,
     'num_features_to_select': 250,
     'min_years_back': 1,
     'correlation_threshold': 0.99,
     'career_phase_model_path': 'Split-by-Age5-Ove5_3y_nc24_knn-ove.pickle'
     },

    # LogisticRegression
    {'model_name': 'LogisticRegression',
     'num_years_back': 3,
     'num_features_to_select': 125,
     'min_years_back': 1,
     'correlation_threshold': 0.99,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },
    {'model_name': 'LogisticRegression',
     'num_years_back': 4,
     'num_features_to_select': 125,
     'correlation_threshold': 0.99,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },

    # TabNet
    {'model_name': 'TabNet',
     'num_years_back': 3,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },
    {'model_name': 'TabNet',
     'num_years_back': 4,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },

    # MLP
    {'model_name': 'MLPNet',
     'num_years_back': 3,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle'
     },
    {'model_name': 'MLPNet',
     'num_years_back': 4,
     'min_years_back': 1,
     'career_phase_model_path': 'Split-by-Age5-Pos5_3y_nc8_knn-ove.pickle',
     },
]


def compare_models(results_dict: Dict[str, Dict], test_preds_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Iterate over experimented model (selected model from out of function scope)
    for model_name in results_dict:
        output_dir = os.path.join(ArtifactsPaths.ARTIFACTS_DIR, 'reports', model_name)
        eval_df = pd.read_csv(os.path.join(output_dir, 'test_predictions.csv'))
        _ = model_evaluation(
            labels=eval_df['label'].tolist(),
            preds=eval_df['pred'].tolist(),
            probs=eval_df['prob'].tolist(),
            players_ratings=eval_df['last_year_rating'].tolist(),
            rating_labels=eval_df['label_rating'].tolist(),
            ratings_yoy_diffs=eval_df['yoy_diff'].tolist(),
            players_names=eval_df['player'].tolist(),
            players_age=eval_df['player_age'].tolist(),
            num_years_data=eval_df['num_years_data'].tolist(),
            output_dir=output_dir,
            model_name=model_name,
            show_plots=SHOW_PLOTS,
            export_plots=SHOW_PLOTS,
            prints=SHOW_PLOTS
        )

    """ Comparison of different models - graphs and metrics. Figures are grouped by num years back. """
    for years_back in YEARS_BACK:
        relevant_results = {k: v for k, v in results_dict.items() if int(k.split('yback')[0][-1]) == years_back}
        if len(relevant_results) == 0:
            continue
        relevant_pred_dfs = {k: v for k, v in test_preds_dfs.items() if int(k.split('yback')[0][-1]) == years_back}
        multi_model_precision_recall_curve(relevant_pred_dfs,
                                           title=f'Precision-Recall Comparison: {years_back}-Years Models')
        multi_model_roc_auc_curve(relevant_pred_dfs,
                                  output_dir=ArtifactsPaths.ARTIFACTS_DIR,
                                  title=f'ROC-AUC Plot: {years_back}-Years Models Comparison')
        multi_model_top_bottom_comparison(relevant_results,
                                          output_dir=ArtifactsPaths.ARTIFACTS_DIR,
                                          show_auc=False,
                                          title=f'Top/Bottom K Comparison: {years_back}-Years Models')

    # Build comparison table
    unified_df = []
    for model_name, metrics in results_dict.items():
        model_metrics = {k: metrics[k] for k in ['auc',
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
                                                 ]}
        model_metrics['model'] = model_name
        unified_df.append(model_metrics)
    compare_df = pd.DataFrame(unified_df)
    compare_df.set_index('model')
    return compare_df


def evaluate_models(output_file_path: str = 'all_models_summary.pickle') -> (Dict[str, Dict], Dict[str, pd.DataFrame]):
    """ Function runs all experiments and return dict of all models results + dataframe of predictions """
    if USE_EXISTING_MODELS and os.path.exists(output_file_path):
        with open(output_file_path, 'rb') as f:
            ret = pickle.load(f)
        return ret['reports'], ret['test_preds_dfs']

    results = {}
    test_preds_dfs = {}
    for model_config in selected_models:
        model_name = model_config['model_name']
        num_years_back = model_config['num_years_back']
        min_years_back = model_config.get('min_years_back', num_years_back - 1)
        name = model_config.get('name', None)
        model_args = model2args[model_name](show_plots=SHOW_PLOTS,
                                            export_figures=EXPORT_FIGURES,
                                            model_class_name=model_name,
                                            **model_config)
        experiment = Experiment(
            FifaClfLabel(),
            model2class[model_name](model_args, name=name, num_years_back=num_years_back,
                                    min_years_back=min_years_back),
            ExperimentArgs(years_back=num_years_back, use_existing_models=USE_EXISTING_MODELS)
        )
        results[experiment.model.name], test_preds_dfs[experiment.model.name] = experiment.run()

    with open(output_file_path, 'wb') as f:
        pickle.dump({'reports': results, 'test_preds_dfs': test_preds_dfs}, f)

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv('experiment_summary.csv')

    return results, test_preds_dfs


if __name__ == '__main__':
    models_results, test_preds_dfs = evaluate_models()

    test_preds_dfs = {k: v for k, v in test_preds_dfs.items() if k in final_selection}
    models_results = {k: v for k, v in models_results.items() if k in final_selection}
    compare_df = compare_models(models_results, test_preds_dfs)
