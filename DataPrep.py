import pandas as pd

def add_lags(df, home_cols, away_cols, number_of_lags):
    df = df.copy()

    lag_dfs = []
    for i in range(1, number_of_lags + 1):
        for home_col in home_cols:
            lag_dfs.append(
                df.groupby('home_team', observed=False)[home_col]
                  .shift(-i)
                  .rename(f'{home_col}_lag{i}')
            )
        for away_col in away_cols:
            lag_dfs.append(
                df.groupby('away_team', observed=False)[away_col]
                  .shift(-i)
                  .rename(f'{away_col}_lag{i}')
            )

    df = pd.concat([df] + lag_dfs, axis=1)

    df.drop(columns=[c for c in home_cols + away_cols if c != 'result_home'], inplace=True, errors='ignore')

    drop_champs = ['champs_TOP_home', 'champs_JUNGLE_home', 'champs_MID_home', 'champs_ADC_home',
                   'champs_SUPPORT_home', 'champs_TOP_away', 'champs_JUNGLE_away', 'champs_MID_away',
                   'champs_ADC_away', 'champs_SUPPORT_away']

    df.drop(columns=drop_champs, inplace=True)

    return df

def data_prep(lec_df):
    lec_df['Date'] = pd.to_datetime(lec_df['Date'])
    lec_df = lec_df.sort_values('Date', ascending=False)
    lec_df.drop(columns=['Date'], inplace=True)

    lec_df = lec_df.fillna(0)

    col_name_percentage = [col for col in lec_df.columns if '%' in col]
    for col in col_name_percentage:
        lec_df[col] = lec_df[col].apply(lambda x: str(x).split('%')[0])

    lec_df.columns = [col.replace('%', '').replace('@', '').replace('+', '') for col in lec_df.columns]

    kda_cols = [x for x in lec_df if 'kda' in x.lower()]
    for col in kda_cols:
        temp = [x for x in lec_df[col] if x != 'Perfect KDA']
        temp = [float(x) for x in temp]
        lec_df[col] = lec_df[col].apply(lambda x: str(x).replace('Perfect KDA', str(max(temp))))

    for col in lec_df.columns:
        try:
            lec_df[col] = lec_df[col].astype(float)
        except:
            lec_df[col] = lec_df[col].astype('category')

    lec_df['result_home'] = (lec_df['result_home'] == 'W').astype(float)

    return lec_df