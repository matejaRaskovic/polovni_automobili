import pandas as pd
import numpy as np


csv_path = 'all_cars.csv'


def main():
    df = pd.read_csv(csv_path, header=0)

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