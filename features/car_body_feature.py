import torch.nn as nn
import torch
import numpy as np

class CarBodyFeature():
    d = {'Limuzina': 0, 'Karavan': 1, 'Džip/SUV': 2, 'Hečbek': 3}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['karoserija'].isin(['Limuzina', 'Karavan', 'Džip/SUV', 'Hečbek'])

    def name(self):
        return 'karoserija'

    def pos(self):
        return 1

    def calculateLoss(self, vector, target, device):
        vector = vector.to(device)
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:len(self.d)]

        dbg = False
        if dbg and np.random.random(1) < 0.1:
            print(vec)
            print(target)

        return lossFun(vec, target)

    def nameToClassId(self, name):
        return self.d[name]


    def idToClassName(self, id):
        for key in self.d:
            if self.d[key] == id:
                return key