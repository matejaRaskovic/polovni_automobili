import requests
from bs4 import BeautifulSoup
import re
from automobil import Automobil
import pprint
import csv
import pandas as pd
import time

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


URLS_all = []
with open('all_audi_a4_urls.txt', 'r') as file:
    for line in file:
        URLS_all.append(line)

URLS = URLS_all
data = []
i=1
for url in URLS:
    print(f"[{i}/{len(URLS)}]")
    i+=1
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    auto = Automobil()
    auto.readFromSoup(soup)
    data.append(vars(auto))
    del(auto)


    keys = data[0].keys()
    with open('audi_a4.csv', 'w', newline='', encoding="utf-8")  as file:
        dict_writer = csv.DictWriter(file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# df = pd.read_csv('audi_a4.csv')
