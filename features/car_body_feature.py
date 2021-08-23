import torch.nn as nn
import torch
import numpy as np

class CarBodyFeature():
    num_classes = 3
    d = {'Limuzina': 0, 'Karavan': 1, 'Džip/SUV': 1, 'Monovolumen (MiniVan)': 1, 'Hečbek': 2}
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['karoserija'].isin(['Limuzina', 'Karavan', 'Džip/SUV', 'Hečbek', 'Monovolumen (MiniVan)'])

    def name(self):
        return 'karoserija'

    def pos(self):
        return 1

    def calculateLoss(self, vector, target, weight, device):
        vector = vector.to(device)
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:self.num_classes]

        dbg = True
        if dbg and np.random.random(1) < 0.025:
            print('\nCar body')
            print(vec)
            print(target)

        return lossFun(vec, target)

    def nameToClassId(self, name):
        return self.d[name]


    def idToClassName(self, id):
        for key in self.d:
            if self.d[key] == id:
                return key

    def calculateGradWeight(self, df):
        for sample in df[self.name()]:
            if sample in self.grad_weight:
                self.grad_weight[sample] += 1
            else:
                self.grad_weight[sample] = 1

        print(self.grad_weight)
        h = len(df.index) / len(self.grad_weight)
        for key in self.grad_weight:
            self.grad_weight[key] = h / self.grad_weight[key]

        print(self.grad_weight)

    def getWeightForSample(self, sample):
        return 1.