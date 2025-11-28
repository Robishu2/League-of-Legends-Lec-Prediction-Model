import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def lec_games(respons):
    # parse HTML response
    soup = BeautifulSoup(respons.text, 'html.parser')

    # select the first table with class 'table_list'
    table = soup.select_one('table.table_list')

    # read HTML table into list of DataFrames
    df_list = pd.read_html(str(table))

    # take the first DataFrame
    df = df_list[0]

    # extract home scores
    score_home = [int(x.split('-')[0]) for x in df['Score']]

    # extract away scores
    score_away = [int(x.split('-')[1]) for x in df['Score']]

    # add home scores column
    df['score_home'] = score_home

    # add away scores column
    df['score_away'] = score_away

    # drop original Score column
    df.drop(columns=['Score'], inplace=True)

    # initialize results list
    result_home = []

    # iterate over rows
    for idx, value in df.iterrows():
        # home team wins
        if value['score_home'] > value['score_away']:
            result_home.append('W')
        # home team loses
        elif value['score_home'] < value['score_away']:
            result_home.append('L')

    # add result column
    df['result_home'] = pd.Series(result_home)

    # rename empty column names
    df = df.rename(columns={'Unnamed: 1': 'home_team', 'Unnamed: 3': 'away_team', 'Unnamed: 4': 'type_of_game'})
    
    ls = table.find_all('a', href=True, title=True)
    games_df = []

    for item in ls:
        game_url = item['href']
        game_teams = item['title']
        games_df.append((game_url, game_teams))

    games_df = pd.DataFrame(games_df, columns=['game_url', 'game_teams']) 
    
    return df, games_df

def get_game_info(game_df):
    game_info = pd.DataFrame()

    for idx, val in games_df.iterrows():
        url_game = val['game_url']
        url = f"https://gol.gg{url_game.split('..')[1]}"

        if 'page-game' in url:
            url = url.replace('page-game', 'page-fullstats')
            teams = val['game_teams'].split(' stats')[0]
        elif 'page-summary' in url:
            url = url.replace('page-summary', 'page-fullstats')
            teams = val['game_teams'].split(' summary')[0]

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        tables = soup.find_all('table')[0]
        df_list = pd.read_html(str(tables))

        imgs = tables.find_all('img')
        champs = [img['alt'] for img in imgs if img.has_attr('alt')]

        temp = pd.DataFrame(df_list[0])
        temp.columns = temp.iloc[1,:]
        temp.drop(1, inplace=True)
        temp = temp.fillna(0)
        temp = temp.set_index(temp.columns[0])

        temp.loc['champs'] = champs
        temp.columns = [f"{x}_home" if idx <= 4 else f"{x}_away" for idx, x in enumerate(temp.columns)]

        flat = temp.T.stack().to_frame().T
        flat.columns = [f'{col}_{idx}' for idx, col in flat.columns]

        flat['Game'] = teams

        game_info = pd.concat([game_info, flat], axis=0)
        
    return game_info

def get_lec_links():
    driver = webdriver.Chrome()
    driver.get('https://gol.gg/tournament/list/')
    wait = WebDriverWait(driver, 10)

    seasons = list(range(3, 16))
    lec_links = []

    for season in seasons:
        button = wait.until(EC.element_to_be_clickable((By.XPATH, f"//a[contains(text(), 'S{season}')]")))
        driver.execute_script('arguments[0].click();', button)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        test = soup.find_all('a')

        for x in test:
            text = x.get_text().lower()
            href = x.get('href', '')
            if 'lec' in text and 'tournament-stats' in href:
                # Extract the tournament name from the href
                name_part = href.split('tournament-stats/')[-1].strip('/')
                # Encode spaces and special characters
                name_encoded = quote(name_part)
                # Reconstruct correct URL
                full_url = f"https://gol.gg/tournament/tournament-matchlist/{name_encoded}/"
                lec_links.append(full_url)

    driver.quit()

    return lec_links

urls = get_lec_links()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

lec_df = pd.DataFrame()

for url in urls:
    respons = requests.get(url)

    df, games_df = lec_games(respons)
    game_info = get_game_info(games_df)
    df_games = df.merge(game_info, on='Game')

    lec_df = pd.concat([lec_df, df_games], axis=0)

lec_df.to_csv('lec_df.csv')