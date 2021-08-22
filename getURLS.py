import requests
from bs4 import BeautifulSoup
import json
import re

# URL pretrage koja se scrape-uje
# SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B0%5D=a4&year_from=2015&year_to=2017&chassis%5B0%5D=277&city_distance=0&showOldNew=all&without_price=1"
# SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B%5D=a4&price_to=&year_from=&year_to=&showOldNew=all&submit_1=&without_price=1"
# SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B0%5D=a4&city_distance=0&showOldNew=all&without_price=1"

# SEARCH_URL_GENERAL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=_BRAND_TO_REPLACE_&chassis%5B0%5D=277&chassis%5B1%5D=2631&chassis%5B2%5D=278&chassis%5B3%5D=2632&city_distance=0&showOldNew=all&without_price=1&color%5B0%5D=2573&color%5B1%5D=253&color%5B2%5D=254&color%5B3%5D=2574&color%5B4%5D=255&color%5B5%5D=59&color%5B6%5D=2575&color%5B7%5D=3328&color%5B8%5D=256&color%5B9%5D=2578&color%5B10%5D=57&color%5B11%5D=258&color%5B12%5D=259&color%5B13%5D=260&color%5B14%5D=2577&color%5B15%5D=2576&color%5B16%5D=261&color%5B17%5D=262&color%5B18%5D=263&interior_material%5B0%5D=3830&interior_material%5B1%5D=3831&interior_material%5B2%5D=3832&interior_material%5B3%5D=3833&interior_material%5B4%5D=3834&interior_color%5B0%5D=3836&interior_color%5B1%5D=3837&interior_color%5B2%5D=3838&interior_color%5B3%5D=3839&interior_color%5B4%5D=3840"
SEARCH_URL_GENERAL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=_BRAND_TO_REPLACE_&city_distance=0&showOldNew=all&without_price=1"
brands = ['volkswagen', 'bmw', 'peugeot', 'fiat', 'renault', 'mercedes-benz']

def createHtmlFile(url, html_file="webpage.html"):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    with open(html_file, "w", encoding='utf-8') as file:
        file.write(soup.prettify())
        
def get_URLS_from_HTML(html_file="webpage.html"):
    with open(html_file,'r', encoding='utf-8') as page:
        soup = BeautifulSoup(page, 'html.parser')
    
    results = soup.find_all('script', type='application/ld+json')
    urls = []
    for result in results:
        data = json.loads(result.string)[0]
        try:
            url = data['url']
            if "https://www.polovniautomobili.com/auto-oglasi" in url:
                urls.append(url.strip())
        except KeyError:
            pass
    return urls

def get_cars_from_page_url(url):
    """
    url - Adresa stranice pretrage na kojoj je prikazano do 25 modela
    
    Ova funkcija vraca listu svih URL adresa pojedinacnih vozila (oglasa)
    koji se nalaze na datoj stranici pretrage
    """

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find_all('script', type='application/ld+json')
    urls = []
    for result in results:
        data = json.loads(result.string)[0]
        try:
            url = data['url']
            if "https://www.polovniautomobili.com/auto-oglasi" in url:
                urls.append(url.strip())
        except KeyError:
            pass
    return urls

    
def get_num_pages_from_url(url):
    """
    Ova funkcija vraca ukupan broj stranica pretrage

    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(text=re.compile("Prikazano od"))
    results = str(results).strip()
    total = int(results.split()[-1])
    print("TOTAL: ", total)
    if total % 25 == 0:
        num_pages = int(total/25)
    else:
        num_pages = (total//25) + 1
    
    return num_pages
    
    
def get_list_of_all_pages(SEARCH_URL):
    """
    Ova funkcija vraca listu URL adresa svih stranica pretrage

    """

    match = re.search("page=[0-9]+", SEARCH_URL)
    if not match:
        raise Exception("Couldn't find 'page=?' in URL of search results page")
    num_pages = get_num_pages_from_url(SEARCH_URL)
    
    pages_url = []
    for num in range(1, num_pages+1):
        url = SEARCH_URL[0:match.start()+5] + str(num) + SEARCH_URL[match.end():]
        pages_url.append(url)

    return pages_url


if __name__ == "__main__":

    all_cars = []
    for brand in brands:
        SEARCH_URL = SEARCH_URL_GENERAL.replace('_BRAND_TO_REPLACE_', brand)
        all_pages = get_list_of_all_pages(SEARCH_URL)
        if len(all_pages) > 124:
            all_pages = all_pages[:124]  # this will keep just 3.1k cars

        for page_url in all_pages:
            page_cars = get_cars_from_page_url(page_url)
            all_cars += page_cars
    
    # Name of txt file to store URLs of individual cars
    URLS_FILE_NAME = "producer_classification_cars.txt"
    with open(URLS_FILE_NAME, 'w') as file:
        for car in all_cars:
            file.write(car + "\n")
