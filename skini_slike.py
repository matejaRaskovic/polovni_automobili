from bs4 import BeautifulSoup
import requests
import json
import time
import os

from multiprocessing import Pool


def createHtmlFile(url, html_file="webpage.html"):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    with open(html_file, "w", encoding='utf-8') as file:
        file.write(soup.prettify())


def download_images_for_url(car_ad_url_and_cnt):
    car_ad_url = car_ad_url_and_cnt[0]
    cnt = car_ad_url_and_cnt[1]
    car_id = car_ad_url.split('/')[-2]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(car_ad_url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    results = soup.find_all('script', type='application/ld+json')
    for d in results:
        if 'caption' in d.string:
            data = json.loads(d.string)

    os.makedirs('slike', exist_ok=True)

    pictures_urls = [dic['caption'] for dic in data]

    t1 = time.time()
    for url in pictures_urls:
        response = requests.get(url)

        os.makedirs(os.path.join('slike', car_id), exist_ok=True)
        file = open(os.path.join('slike', car_id, os.path.basename(url)), "wb")
        file.write(response.content)
        file.close()

    t2 = time.time()
    print('Finished car ad ' + str(cnt) + '. Time spent: ' + str(t2 - t1))
    print('')


def get_images_from_urls(txt_file="all_cars.txt"):
    with open(txt_file, 'r') as urls_file:
        urls = []
        i = 1
        for url in urls_file:
            urls.append([url, i])
            i += 1

        p = Pool(32)
        p.map(download_images_for_url, tuple(urls))


if __name__ == '__main__':
    get_images_from_urls()

