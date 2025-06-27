from typing import Tuple, List
import pandas as pd

from stat_predict.static.config import YOY_DIFF_TO_CAT, YOY_CATS, LABEL_COLUMN
from stat_predict.static.utils import COLUMNS


class FifaStatLabel(object):
    def label_encoding(self,
                       df: pd.DataFrame
                       ) -> pd.DataFrame:
        """
        :param df: players history data, for all years and players
        :return: df with encoded values
        """
        raise NotImplementedError


class FifaClfLabel(FifaStatLabel):
    def __init__(self,
                 label_bins: Tuple[float] = YOY_DIFF_TO_CAT,
                 clf_labels: Tuple[str] = YOY_CATS
                 ):
        """
        :param label_bins: values used as thresholds to convert the continuous raw label to label categories
        :param clf_labels: names of classification labels (length of categories names = len(dif_clf_ranges) + 1)
        """
        self.label_bins = label_bins
        self.clf_labels = clf_labels

    def label_encoding(self,
                       df: pd.DataFrame,
                       cats_thresholds: tuple[float] = YOY_DIFF_TO_CAT,
                       label_col: str = 'label',
                       raw_label_col: str = 'raw_label'
                       ) -> (pd.DataFrame, List[str]):
        """
        :param df: players history data, for all years and players
        :param cats_thresholds: thresholds to encode the raw label value to category (=label column)
        :param label_col: name of the label columns
        :param raw_label_col: name of the raw label column value
        :return: df with encoded values
        """
        # Sort by col1 and year to ensure correct order for diff calculation
        df = df.sort_values(by=[COLUMNS.PLAYER_ID, COLUMNS.YEAR])

        # Convert label to binary classification, keep raw_label_col with the original label rating
        df[raw_label_col] = df[label_col].apply(lambda x: x/100 if not pd.isna(x) else x ).round(2)
        df[label_col] = df[[LABEL_COLUMN, raw_label_col]].apply(lambda x: int(x[0] < x[1])
                                                                     if not pd.isna(x[1]) else x[1],
                                                                     axis=1)

        assert df[label_col].isna().sum() == df[raw_label_col].isna().sum()
        print('\nYears labels distribution')
        print(df[[COLUMNS.YEAR, label_col]].groupby(COLUMNS.YEAR).value_counts(normalize=True))
        return df, [raw_label_col, label_col, COLUMNS.YEAR, COLUMNS.PLAYER_ID]
