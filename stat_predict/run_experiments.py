import pickle
from itertools import product
import pandas as pd
from tqdm import tqdm

from stat_predict.dataset.label import FifaClfLabel
from stat_predict.models.experiment import ExperimentArgs, Experiment
from stat_predict.models.baselines import LRArgs, CatboostArgs, XGBArgs, TabNetStatPredict, TabNetArgs

YEARS_BACK = [3, 4]

model2class = {
    # 'XGBStatPredict': XGBStatPredict,
    # 'CatBoostStatPredict': CatBoostStatPredict,
    # 'LRStatPredict': LRStatPredict,
    # 'KAN': KanNetworkStatPredict,
    # 'KAN_LARGE': KanNetworkStatPredict,
    # 'StatPredictDLModel': DLStatPredict,
    # 'StatPredictDLModelV2': DLStatPredict,
    # 'StatPredictDLModelV3': DLStatPredict,
    # 'StatPredictDLModelV4': DLStatPredict,
    # 'StatPredictDLModelV5': DLStatPredict,
    'TabNet': TabNetStatPredict,
    # 'MLPNet': DLStatPredict,
}

model2args = {
    'XGBStatPredict': XGBArgs,
    'CatBoostStatPredict': CatboostArgs,
    'LRStatPredict': LRArgs,
    'TabNet': TabNetArgs,
    # 'KAN': KanArgs,
    # 'KAN_LARGE': KanLargeArgs,
    # 'MLPNet': DLArgs,
    # 'StatPredictDLModel': StatPredictDLModelArgs,
    # 'StatPredictDLModelV2': StatPredictDLModelArgs,
    # 'StatPredictDLModelV3': StatPredictDLModelArgs,
    # 'StatPredictDLModelV4': StatPredictDLModelArgs,
    # 'StatPredictDLModelV5': StatPredictDLModelArgs,
}
# KanArgs(layers_width=[5, 3, 2]),
# KanArgs(layers_width=[5, 3, 2], k=3, grid=5),


# Experiment parameters
param_grid = {
    # 'filter_families': [['growth', 'diffs']], #  ['diffs'], ['growth'],
    # 'num_features_to_select': [75, 125, 150, 250],  # 500
    # 'selection_correlation_method': ['pearson'],  # , 'spearman'
    # 'correlation_threshold': [0.99, 1.], # 0.95,
    'career_phase_model_path': [
        # 4 years
        'SplitPhase-by-Age4-Ove5_series-raw_4y_nc16_knn_attrs=ove_.pickle',
        'SplitPhase-by-Age4-Ove5_series-raw_4y_nc8_knn_attrs=ove_.pickle',
        # 'SplitPhase-by-Ove3-Age4_series-raw_4y_nc8_knn_attrs=ove_.pickle',
        # 'SplitPhase-by-Age4-Ove5_series-raw_4y_nc8_knn_attrs=ove-pot_.pickle',
        # 'SplitPhase-by-Age4-PY5-Ove5_series-raw_4y_nc8_knn_attrs=ove-pot_.pickle',
        'SplitPhase-by-Age4-PY5-Ove5_series-raw_4y_nc8_knn_attrs=ove_.pickle',

        # 3 years
        # 'SplitPhase-by-Age4-PY5_series-diff_3y_nc8_knn_attrs=ove_diff.pickle',
        'SplitPhase-by-Age4-Ove5_series-raw_3y_nc16_knn_attrs=ove_.pickle',
        'SplitPhase-by-Age4-Ove5_series-raw_3y_nc24_knn_attrs=ove_.pickle',
        'SplitPhase-by-Age4-PY5_series-raw_3y_nc8_knn_attrs=ove_.pickle',
        # 'SplitPhase-by-Age4-PY5-Ove5_series-raw_3y_nc8_knn_attrs=ove_.pickle',
        # None
    ]
}
grid_search_params = [
    dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
]

if __name__ == '__main__':
    USE_EXISTING_MODELS = True
    SHOW_PLOTS = False
    EXPORT_FIGURES = False

    skipped_configs = []
    results = {}
    for num_years_back in YEARS_BACK:
        for model_name in model2class:
            model_class = model2class[model_name]

            for model_config in tqdm(grid_search_params, total=len(grid_search_params)):
                if num_years_back == 3 and model_config['career_phase_model_path'] is not None:
                    if '_4y_' in model_config['career_phase_model_path']:
                        continue

                model_args = model2args[model_name](show_plots=SHOW_PLOTS,
                                                    export_figures=EXPORT_FIGURES,
                                                    model_class_name=model_name,
                                                    **model_config)
                experiment = Experiment(
                    FifaClfLabel(),
                    model_class(model_args, num_years_back=num_years_back),
                    ExperimentArgs(years_back=num_years_back, use_existing_models=USE_EXISTING_MODELS)
                )
                # if len(os.listdir(experiment.model.reports_output_dir)) > 0:
                #     continue
                try:
                    results[experiment.model.name], test_pred_df = experiment.run()
                except:
                    skipped_configs.append((model_name, model_config))
                    continue

        print(f'Finished run {num_years_back} years back')
        print('Skipped configs due to errors:', skipped_configs)
        pd.DataFrame.from_dict(results, orient='index').to_csv(f'experiment_summary_yback={num_years_back}.csv')
        print()

    print('Skipped configs due to errors:', skipped_configs)
    with open('experiment_summary.pickle', 'wb') as f:
        pickle.dump(results, f)
    pd.DataFrame.from_dict(results, orient='index').to_csv('experiment_summary.csv')
    print()
