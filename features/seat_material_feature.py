import torch.nn as nn
import torch
import numpy as np

class SeatMaterialFeature():
    num_classes = 4
    # d = {'Štof': 0, 'Prirodna koža': 1, 'Kombinovana koža': 1, 'Velur': 2, 'Drugi': 3}
    d = {'Štof': 0, 'Koža': 1, 'Velur': 2, 'Drugi': 3}  # for testing with oversampling due to imbalance
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['materijal_enterijera'].isin(['Štof', 'Prirodna koža', 'Kombinovana koža', 'Velur', 'Drugi'])

    def name(self):
        return 'materijal_enterijera'

    def pos(self):
        return 3

    def calculateLoss(self, vector, target, weight, device):
        vector = vector.to(device)
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:self.num_classes]

        dbg = True
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