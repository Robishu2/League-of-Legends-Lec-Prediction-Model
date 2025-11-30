import pandas as pd

def add_lags(df, home_cols, away_cols, number_of_lags):

    df = df.copy().reset_index(drop=True)
    lag_cols = {}

    # ---- Get unique teams ----
    teams = set(df['home_team']).union(df['away_team'])

    for team in teams:

        # ---- Build long-format team history ----
        home_part = df.loc[df['home_team'] == team, home_cols].copy()
        home_part['original_index'] = home_part.index

        away_part = df.loc[df['away_team'] == team, away_cols].copy()
        away_part['original_index'] = away_part.index

        long_df = pd.concat([home_part, away_part], ignore_index=True)

        # Ensure chronological newest → oldest
        long_df = long_df.sort_values('original_index')

        # ---- Compute lag features ----
        # ---- Compute lag features efficiently ----
        lag_dict = {}

        for col in home_cols:
            if col not in long_df:
                continue

            series = long_df[col]

            for lag in range(1, number_of_lags + 1):
                lag_name = f"{col}_lag{lag}"
                lag_dict[lag_name] = series.shift(-lag).values

        # Add all lag columns in one concat (no fragmentation)
        lag_block = pd.DataFrame(lag_dict, index=long_df.index)
        long_df = pd.concat([long_df, lag_block], axis=1)


        # ---- Map lag features back to main df ----
        for col in home_cols:
            for lag in range(1, number_of_lags + 1):

                lag_name = f"{col}_lag{lag}"
                if lag_name not in long_df:
                    continue

                mapped = long_df.set_index("original_index")[lag_name]

                if lag_name not in lag_cols:
                    lag_cols[lag_name] = pd.Series([None] * len(df), dtype=float)

                lag_cols[lag_name].update(mapped)

    # ---- Attach all lag columns ----
    df = pd.concat([df, pd.DataFrame(lag_cols)], axis=1)

    # ---- Determine columns to keep ----
    keep_cols = [
        c for c in df.columns
        if (
            "lag" in c
            or "rolling" in c
            or c == "result_home"
            or "team" in c.lower()
            or "player" in c.lower()
        )
    ]

    df = df[keep_cols].copy()

    # ---- Drop champion columns ----
    drop_champs = [
        'champs_TOP_home','champs_JUNGLE_home','champs_MID_home','champs_ADC_home',
        'champs_SUPPORT_home','champs_TOP_away','champs_JUNGLE_away','champs_MID_away',
        'champs_ADC_away','champs_SUPPORT_away'
    ]

    df.drop(columns=[c for c in drop_champs if c in df.columns], inplace=True)

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

def rolling_window(df, home_cols, away_cols):
    """
    Compute rolling player stats for each role (TOP/JUNGLE/MID/ADC/SUPPORT)
    using the original logic:
    
    - Identify player stat columns by role tag
    - Merge home + away appearances for each role
    - Compute rolling window over last 10 games (newest → oldest)
    - Assign rolling stats back to correct match rows
    """

    player_tags = ['TOP','JUNGLE','MID','ADC','SUPPORT']

    # Work on fresh sequential index
    temp = df.copy().reset_index(drop=True)

    rolling_cols = {}

    for tag in player_tags:

        # ---- Identify stat columns for this role ----
        home_stat_cols = [c for c in home_cols if tag in c and 'Player' not in c]
        away_stat_cols = [c for c in away_cols if tag in c and 'Player' not in c]

        # Zip ensures you always match TOP_home to TOP_away etc.
        for home_col, away_col in zip(home_stat_cols, away_stat_cols):

            # ---- Build long-format player history ----
            # Home-side appearances
            home_games = temp[[f'Player_{tag}_home', home_col]].copy()
            home_games.columns = ['player', 'stat']
            home_games['original_index'] = home_games.index

            # Away-side appearances
            away_games = temp[[f'Player_{tag}_away', away_col]].copy()
            away_games.columns = ['player', 'stat']
            away_games['original_index'] = away_games.index

            # Concatenate all games of this role
            all_games = pd.concat([home_games, away_games], ignore_index=True)

            # ---- Compute rolling window ----
            # Newest → oldest order preserved by original_index
            all_games['rolling_avg'] = (
                all_games.groupby('player')['stat']
                    .rolling(window=10, min_periods=1)
                    .mean()
                    .shift(-1)  # because newest→oldest
                    .reset_index(level=0, drop=True)
            )

            # ---- Map back to home rows ----
            home_part = (
                all_games.iloc[:len(temp)]
                        .set_index('original_index')['rolling_avg']
            )
            rolling_cols[f'{home_col}_rolling10'] = home_part

            # ---- Map back to away rows ----
            away_part = (
                all_games.iloc[len(temp):]
                        .set_index('original_index')['rolling_avg']
            )
            rolling_cols[f'{away_col}_rolling10'] = away_part

    # ---- Add all rolling columns at once ----
    temp = pd.concat([temp, pd.DataFrame(rolling_cols)], axis=1)

    return temp