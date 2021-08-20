import requests
from bs4 import BeautifulSoup
from automobil import Automobil
import csv
from multiprocessing import Pool, Manager
import sys
sys.setrecursionlimit(25000)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

def fetch_a_sample(url):
    global data
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    auto = Automobil()
    auto.readFromSoup(soup)
    print(1)
    return vars(auto)


def get_data_to_csv():
    URLS_all = []
    with open('all_audi_a4_urls_1.txt', 'r') as file:
        for line in file:
            URLS_all.append(line)

    URLS = URLS_all

    p = Pool(32)
    data = p.map(fetch_a_sample, tuple(URLS))

    keys = data[0].keys()
    with open('all_cars.csv', 'w', newline='', encoding="utf-8") as file:
        dict_writer = csv.DictWriter(file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


if __name__ == '__main__':
    get_data_to_csv()
