import pandas as pd
import numpy as np
import os


csv_path = 'all_cars.csv'
imgs_fldr = 'slike'
filter_without_images = True

def main():
    df = pd.read_csv(csv_path, header=0)
    valid_ads = []
    if filter_without_images:
        for ad_id in df['br_oglasa'].astype(str):
            ad_imgs_fldr = os.path.join(imgs_fldr, ad_id)
            if os.path.exists(ad_imgs_fldr) and len(os.listdir(ad_imgs_fldr)) > 0:
                #print(ad_id)
                valid_ads.append(ad_id)

    df = df[df['br_oglasa'].astype(str).isin(valid_ads)]

    msk = np.random.rand(len(df)) < 0.9
    train_valid = df[msk]
    print(len(df))
    print(len(train_valid))
    test = df[~msk]

    msk = np.random.rand(len(train_valid)) < 0.9
    train = train_valid[msk]
    valid = train_valid[~msk]

    test.to_csv(csv_path.replace('.csv', '_test.csv'), index=False)
    train.to_csv(csv_path.replace('.csv', '_train.csv'), index=False)
    valid.to_csv(csv_path.replace('.csv', '_valid.csv'), index=False)


if __name__ == '__main__':
    main()
