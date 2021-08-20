import torch.nn as nn
import torch
import numpy as np

class SeatMaterialFeature():
    num_classes = 8
    d = {'Volkswagen': 0, 'Audi': 1, 'BMW': 2, 'Opel': 3, 'Peugeot': 4, 'Fiat': 5, 'Renault': 6, 'Mercedes Benz': 7}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['marka'].isin(['Volkswagen', 'Audi', 'BMW', 'Opel', 'Peugeot', 'Fiat', 'Renault', 'Mercedes Benz'])

    def name(self):
        return 'marka'

    def pos(self):
        return 0

    def calculateLoss(self, vector, target, device):
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