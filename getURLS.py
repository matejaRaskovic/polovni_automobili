import requests
from bs4 import BeautifulSoup
import json
import re

# URL pretrage koja se scrape-uje
# SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B0%5D=a4&year_from=2015&year_to=2017&chassis%5B0%5D=277&city_distance=0&showOldNew=all&without_price=1"
# SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B%5D=a4&price_to=&year_from=&year_to=&showOldNew=all&submit_1=&without_price=1"
SEARCH_URL = "https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1&sort=basic&brand=audi&model%5B0%5D=a4&city_distance=0&showOldNew=all&without_price=1"

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

def get_num_pages_from_html(html_file="webpage.html"):
    """
    Ova funkcija vraca ukupan broj stranica pretrage
    (STARA VERZIJA F-CIJE)

    """
    with open('webpage.html','r', encoding='utf-8') as page:
        soup = BeautifulSoup(page, 'html.parser')

    results = soup.find(text=re.compile("Prikazano od"))
    results = str(results).strip()
    total = int(results.split()[-1])
    print("TOTAL: ", total)
    if total%25==0:
        num_pages = total/25
    else:
        num_pages = (total//25) + 1
    
    return num_pages

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
    if total%25==0:
        num_pages = total/25
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
    
    all_pages = get_list_of_all_pages(SEARCH_URL)
    all_cars = []
    for page_url in all_pages:
        page_cars = get_cars_from_page_url(page_url)
        all_cars += page_cars

    createHtmlFile(all_cars[3])
    
    # Name of txt file to store URLs of individual cars
    URLS_FILE_NAME = "all_audi_a4_urls.txt"
    with open(URLS_FILE_NAME, 'w') as file:
        for car in all_cars:
            file.write(car + "\n")
