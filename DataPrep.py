import pandas as pd

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

    return lec_df, lec_df['result_home']

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

    df.drop(columns=home_cols, inplace=True)
    df.drop(columns=away_cols, inplace=True)

    return df

def rolling_window(df, home_cols, away_cols, window=5):
    player_tags = ['TOP','JUNGLE','MID','ADC','SUPPORT']

    temp = df.copy().reset_index(drop=True)
    rolling_cols = {}

    for tag in player_tags:

        # Identify stat columns for home/away for this role
        home_stat_cols = [c for c in home_cols if tag in c and 'Player' not in c]
        away_stat_cols = [c for c in away_cols if tag in c and 'Player' not in c]

        for home_col, away_col in zip(home_stat_cols, away_stat_cols):

            # Build stacked player-game data
            home_games = pd.DataFrame({
                "player": temp[f"Player_{tag}_home"],
                "stat": temp[home_col],
                "row": temp.index
            })

            away_games = pd.DataFrame({
                "player": temp[f"Player_{tag}_away"],
                "stat": temp[away_col],
                "row": temp.index
            })

            all_games = pd.concat([home_games, away_games], ignore_index=True)

            # ---- REVERSE → ROLL → REVERSE BACK ----

            # 1. reverse to oldest→newest
            rev = all_games.iloc[::-1].copy()

            # 2. rolling mean on previous window matches only
            rev["rolling_avg"] = (
                rev.groupby("player")["stat"]
                   .rolling(window=window, min_periods=window)
                   .mean()
                   .shift(1)  # ensure only past matches used
                   .reset_index(level=0, drop=True)
            )

            # 3. reverse back to original order
            all_games["rolling_avg"] = rev["rolling_avg"].iloc[::-1].values

            # Map back to home and away rows
            home_part = all_games.iloc[:len(temp)].set_index("row")["rolling_avg"]
            away_part = all_games.iloc[len(temp):].set_index("row")["rolling_avg"]

            rolling_cols[f"{home_col}_rolling{window}"] = home_part
            rolling_cols[f"{away_col}_rolling{window}"] = away_part

    # Add all rolling features at once
    temp = pd.concat([temp, pd.DataFrame(rolling_cols)], axis=1)

    return temp