from bs4 import BeautifulSoup
import requests
import json
import time
import os

# html_file = 'webpage.html'

# with open(html_file,'r', encoding='utf-8') as page:
#     soup = BeautifulSoup(page, 'html.parser')

def createHtmlFile(url, html_file="webpage.html"):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    with open(html_file, "w", encoding='utf-8') as file:
        file.write(soup.prettify())

def get_images_from_urls(txt_file="all_audi_a4_urls.txt"):
    with open(txt_file, 'r') as urls_file:
        i = 1
        urls = []
        # id_set = set()
        for url in urls_file:
            urls.append(url)
            # car_id = url.split('/')[-2] + '-' + url.split('/')[-1]
            # id_set.add(car_id)

        print(len(urls))
        # print(id_set)
        exit(1)

        for url in urls:
            # print(url)

            createHtmlFile(url)

            car_id = url.split('/')[-2]
            html_file = 'webpage.html'

            with open(html_file, 'r', encoding='utf-8') as page:
                soup = BeautifulSoup(page, 'html.parser')

            results = soup.find_all('script', type='application/ld+json')
            # urls = []
            for d in results:
                if 'caption' in d.string:
                    data = json.loads(d.string)
            # print(results)
            # data = json.loads(results[1].string)
            # print(data)

            os.makedirs('slike', exist_ok=True)

            # print(data)

            pictures_urls = [dic['caption'] for dic in data]
            print("Downloading " + str(len(pictures_urls)) + " images for car " + str(i) + "/" + str(len(urls)) + "...")
            i += 1
            # print(*pictures_urls,sep='\n')

            t1 = time.time()
            for url in pictures_urls:
                response = requests.get(url)

                os.makedirs(os.path.join('slike', car_id), exist_ok=True)
                file = open(os.path.join('slike', car_id, os.path.basename(url)), "wb")
                file.write(response.content)
                file.close()

            t2 = time.time()
            print('Finished. Time spent: ' + str(t2-t1))
            print('')
    
    
    # for result in results:
    #     # print(result, '\n\n*************\n\n\n')
    #     data = json.loads(result.string)[0]
    #     try:
    #         url = data['url']
    #         if "gcdn" in url:
    #             urls.append(url.strip())
    #     except KeyError:
    #         print("ERROR!\n\n\n")
    #     except:
    #         print("Error 2\n\n")
    # return urls

linkovi = get_images_from_urls()

