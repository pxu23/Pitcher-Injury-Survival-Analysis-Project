"""
Descripiton:
    This script is a webscraper for injury data from the Pro Sports Transaction website
Written by:
    G-R-H
"""
"""
    Acknowledgement: G-R-H, we adapt this script to scrape the baseball injury data from Pros Sports Transactions
    for my own work. 
"""

import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def replace_all(text, dic):
    '''
    This function will replace characters in text given a dictionary of characters to seach for and replace
    '''
    rc = re.compile('|'.join(map(re.escape, dic)))

    def translate(match):
        return dic[match.group(0)]

    return rc.sub(translate, text)


# Dictionary of characters to remove from text
char_replace = {' • ': ''}

# Create list of records read from webscrapper
list_of_rows = []

# scrape the Prosports Transactions baseball injury data for all seasons
for i in tqdm(range(0, 6225, 25)):
    url = 'https://prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate=2022-01-01&EndDate=2023-12-31&DLChkBx=yes&InjuriesChkBx=yes&submit=Search&start={}'.format(i)
    response = requests.get(url)
    html = response.content

    soup = BeautifulSoup(html, 'lxml')

    table = soup.find('table', attrs={'class': 'datatable center'})

    for row in table.findAll('tr', attrs={'align': 'left'}):
        list_of_cells = []

        for cell in row.findAll('td'):
            text = replace_all(cell.text, char_replace)
            text = text.strip()
            list_of_cells.append(text)
        list_of_rows.append(list_of_cells)

# Store data in a dataframe for manipulation
injuries_df = pd.DataFrame(list_of_rows, columns=[
    'Date', 'Team', 'Acquired', 'Relinquished', 'Notes'])

acq = injuries_df['Acquired']
rel = injuries_df['Relinquished']

# Remove instances where value is like "Name 1/ Name 2"
injuries_df['Acquired'] = np.where(
    acq.str.contains('/'), acq.str.split('/ ').str[1], acq)
injuries_df['Relinquished'] = np.where(
    rel.str.contains('/'), rel.str.split('/ ').str[1], rel)

# Remove instances where value is like "(some text)"
injuries_df['Acquired'] = injuries_df.Acquired.str.replace(
    r"[\(\[].*?[\)\]]", "")
injuries_df['Relinquished'] = injuries_df.Relinquished.str.replace(
    r"[\(\[].*?[\)\]]", "")

# Write contents to csv file
injuries_df.to_csv('Injury_Data/prosports_transactions_injuries_2022_2023.csv', index=False)
