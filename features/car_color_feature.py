import torch.nn as nn
import torch
import numpy as np

class CarColorFeature():
    num_classes = 6
    d = {'Crna': 0, 'Bela': 1, 'Siva': 2, 'Plava': 3, 'Crvena': 4}
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['boja'].isin(['Crna', 'Bela', 'Siva', 'Plava', 'Crvena'])

    def name(self):
        return 'boja'

    def pos(self):
        return 2

    def calculateLoss(self, vector, target, weight, device):
        vector = vector.to(device)
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:self.num_classes]

        dbg = True
        if dbg and np.random.random(1) < 0.1:
            print(vec)
            print(target)

        return lossFun(vec, target)*weight

    def nameToClassId(self, name):
        if name in self.d:
            return self.d[name]
        else:
            return len(self.d)

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

        h = len(df.index) / len(self.grad_weight)
        for key in self.grad_weight:
            self.grad_weight[key] = h / self.grad_weight[key]

        print(self.grad_weight)

    def getWeightForSample(self, sample):
        self.grad_weight = {'Bela': 1.7183723797780517, 'Crvena': 4.389291338582677, 'Siva': 0.5013851412124483, 'Crna': 0.636783184829792, 'Plava': 1.5990820424555363}
        return self.grad_weight[sample]
