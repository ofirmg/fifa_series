import pickle
import pandas as pd

from stat_predict.eval.evaluation import build_final_report
from stat_predict.models.career_phase import SplitCareerPhaseModel
from stat_predict.static.config import LABEL_COLUMN
from stat_predict.static.utils import TS_COLS, COLUMNS

# Splits should be mutual exclusive
age_splits = {TS_COLS.AGE: ((15, 20), (21, 25), (26, 30), (31, 34), (35, 50))}
position_split = {
    'position:pitch_pos_position_y_row_0': ((0, 1), (2, 2), (3, 5), (6, 6), (7, 8))
    #  (0, 1),  # GK
    #  (2, 2),  # defenders
    #  (3, 5),  # midfielders
    #  (6, 6)  # attackers
    #  (7, 8) # forwards
}
rating_trinary_split = {TS_COLS.PLAYER_RATING: ((0., 0.64), (0.65, 0.75), (0.76, 0.99))}
rating_full_split = {TS_COLS.PLAYER_RATING: ((0., 0.64), (0.65, 0.69), (0.70, 0.74), (0.75, 0.79), (0.80, 0.99))}

# Clustering configs
base3_clustering = dict(n_clusters=8, num_years_back=3)
large3_clustering = dict(n_clusters=24, num_years_back=3)

# KNN configs
base_knn = dict(knn_attrs=[LABEL_COLUMN])

if __name__ == '__main__':
    USE_EXISTING = True

    stats_datasets = {}
    with open('artifacts/stat_predict_dataset_3yback.pickle', 'rb') as f:
        stats_datasets[3] = pickle.load(f)

    clustering_models = [
        SplitCareerPhaseModel(splits_config={**age_splits, **rating_full_split, **position_split},
                              clustering_attr=[LABEL_COLUMN],
                              **large3_clustering,
                              **base_knn),
        SplitCareerPhaseModel(splits_config={**age_splits, **rating_full_split, **position_split},
                              clustering_attr=[LABEL_COLUMN],
                              **base3_clustering,
                              **base_knn),
        # SplitCareerPhaseModel(splits_config={**age_splits, **rating_full_split},
        #                       clustering_attr=[LABEL_COLUMN],
        #                       **large3_clustering,
        #                       **base_knn),
        # SplitCareerPhaseModel(splits_config={**age_splits, **position_split},
        #                       clustering_attr=[LABEL_COLUMN],
        #                       **base3_clustering,
        #                       **base_knn),
    ]

    reports = {}
    for model in clustering_models:
        print('-' * 150 + '-' * 150 + f'\nExperimenting model {model.name}')
        knn_report, clustering_report = model.train(
            stats_datasets[model.model_args.num_years_back],
            use_existing=USE_EXISTING
        )
        clustering_final_models = build_final_report(clustering_report)
        knn_final_report = build_final_report(knn_report)
        reports[model.name] = {**clustering_final_models, **{f'knn{k}': v for k, v in knn_final_report.items()}}
        pd.DataFrame.from_dict(reports, orient='index').to_csv('experiment_summary_career_phase.csv')
