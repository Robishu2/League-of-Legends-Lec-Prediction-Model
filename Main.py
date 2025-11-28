import numpy as np
import pandas as pd

from DataGathering import request_data
from DataPrep import add_lags, data_prep

if __name__ == '__main__':
    lec_df = request_data()

    lec_df = data_prep(lec_df)

    home_cols = [x for x in lec_df.columns if 'home' in x.lower() and lec_df[x].dtype == float]
    away_cols = [x for x in lec_df.columns if 'away' in x.lower() and lec_df[x].dtype == float]

    lec_df = add_lags(lec_df, home_cols, away_cols, number_of_lags=3)

    y = lec_df['result_home']
    X = lec_df.drop(columns=['result_home']).copy()