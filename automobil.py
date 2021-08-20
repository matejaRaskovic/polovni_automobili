import re

class Automobil():
    
    attrs = {
        "marka": "Marka", 
        "model": "Model",
        "godiste": "Godište",
        "kilometraza": "Kilometraža",
        "gorivo": "Gorivo",
        "karoserija": "Karoserija",
        "kubikaza": "Kubikaža",
        "snaga": "Snaga motora",
        "pogon": "Pogon",
        "menjac": "Menjač",
        "br_vrata" : "Broj vrata",
        "boja": "Boja",
        "br_oglasa": "Broj oglasa:",
        "materijal_enterijera": "Materijal enterijera",
        "boja_enterijera": "Boja enterijera",
        "broj_sedista": "Broj sedišta",
        "registrovan_do": "Registrovan do"
        }
    
    def __init__(self):
        pass
    
    def readFromSoup(self, soup):
        for attr_name, attr_text in self.attrs.items():
            try:
                value = soup.find(text=attr_text).findNext('div').contents[0]
                setattr(self, attr_name, value)
            except AttributeError:
                setattr(self, attr_name, None)
        
        # get and set price        
        try:
            span = soup.find('span', {'class' : "priceClassified discountedPriceColor"})
            price = span.get_text()
            setattr(self, "cena", price)
        except AttributeError:
            try:
                span = soup.find('span', {'class' : "priceClassified regularPriceColor"})
                price = span.get_text()
                setattr(self, "cena", price)
            except AttributeError:
                setattr(self, "cena", None)
                
            
            
        # get and set mobile phone number
        try:
            for a in soup.find_all('a', href=True):
                if "tel:" in a['href']:
                    setattr(self, "mobilni", a['href'])
                    break
        except:
            setattr(self, "mobilni", None)
            
            
        
            


