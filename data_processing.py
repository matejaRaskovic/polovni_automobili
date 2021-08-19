import pandas as pd


"data_processing.py samo sredjuje preuzete podatke (cenu, telefon, itd..)"

df = pd.read_csv("opel_astra_J.csv")
df.dropna(inplace=True)

mob = []

for num in df['mobilni']:
    num = num.split(':')[1]
    num = num.strip()
    rep = [('/',''), ('-',''), (' ',''), ('+381', '0'), ('00381', '0')]
    for old, new in rep:
        num = num.replace(old, new)
    if num.startswith('381'):
        num = num.replace('381', '0')
    
    mob.append(num)

df['mobilni'] = mob

def cena_u_int(cena):
    cena = cena.replace(' ', '').replace('€','').replace('.','')    
    try:
        cena = int(cena)
    except:    
        cena = 0
    return cena
    
df['cena'] = df['cena'].map(cena_u_int)
df['br_oglasa'] = df['br_oglasa'].astype(int)

df['kubikaza'] = df['kubikaza'].map(lambda x: int(x.split()[0]))


# filt = (df['gorivo']=='Dizel') & (df['cena'] > 3000)

# df = df[filt]
# print(df['godiste'].mean())

# cene = list(df['cena'])
# import matplotlib.pyplot as plt
# plt.hist(cene, 9, alpha=0.5, edgecolor='black')
# plt.grid()
# plt.xlim(4000, 10000)