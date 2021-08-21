import torch.nn as nn
import torch
import numpy as np

class InteriorColorFeature():
    num_classes = 5
    d = {'Crna': 0, 'Bež': 1, 'Smeđa': 2, 'Siva': 3, 'Druga': 4}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['boja_enterijera'].isin(['Crna', 'Bež', 'Smeđa', 'Siva', 'Druga'])

    def name(self):
        return 'boja_enterijera'

    def pos(self):
        return 4

    def calculateLoss(self, vector, target, device):
        vector = vector.to(device)
        # print(target)
        target, weight = target
        target = target.view((1)).type(torch.LongTensor).to(device)

        lossFun = nn.CrossEntropyLoss()
        vec = vector[:, 0:self.num_classes]

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

    def calculateGradWeight(self, df):
        pass

    def getWeightForSample(self, sample):
        return 1.
