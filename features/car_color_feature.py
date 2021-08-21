import torch.nn as nn
import torch
import numpy as np

class CarColorFeature():
    num_classes = 6
    d = {'Crna': 0, 'Bela': 1, 'Siva': 2, 'Plava': 3, 'Crvena': 4}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['boja'].isin(['Crna', 'Bela', 'Siva', 'Plava', 'Crvena'])

    def name(self):
        return 'boja'

    def pos(self):
        return 2

    def calculateLoss(self, vector, target, device):
        vector = vector.to(device)
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:self.num_classes]

        dbg = False
        if dbg and np.random.random(1) < 0.1:
            print(vec)
            print(target)

        return lossFun(vec, target)

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
        pass
