import requests
from bs4 import BeautifulSoup
from automobil import Automobil
import csv
import argparse

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--urls_txt', required=True,
                        help='path to the txt containing urls which should be visited')
    parser.add_argument('--output_csv', required=True,
                        help='path to the output csv')

    args = parser.parse_args()

    URLS_all = []
    with open(args.urls_txt, 'r') as file:
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
        if vars(auto)['marka'] is None:
            continue
        data.append(vars(auto))
        del(auto)


        keys = data[0].keys()
        with open(args.output_csv, 'w', newline='', encoding="utf-8") as file:
            dict_writer = csv.DictWriter(file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)


if __name__ == '__main__':
    main()
