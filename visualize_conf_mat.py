import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# array = [[312, 67, 76],
#          [93, 68, 299],
#          [61, 54, 364]]

array = [[436, 21],
         [115, 358]]

lbl_names = ['Limuzina', 'Hečbek']
df_cm = pd.DataFrame(array, lbl_names, lbl_names)

sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

plt.show()