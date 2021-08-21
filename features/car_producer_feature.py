import torch.nn as nn
import torch
import numpy as np

class CarProducerFeature():
    num_classes = 6
    d = {'Volkswagen': 0, 'Audi': 1, 'BMW': 2, 'Opel': 3, 'Peugeot': 4, 'Fiat': 5}  # , 'Renault': 6, 'Mercedes Benz': 7}
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['marka'].isin(['Volkswagen', 'Audi', 'BMW', 'Opel', 'Peugeot', 'Fiat'])  # , 'Renault', 'Mercedes Benz'])

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

    def calculateGradWeight(self, df):
        for sample in df[self.name()]:
            if sample in self.grad_weight:
                self.grad_weight[sample] += 1
            else:
                self.grad_weight[sample] = 1

        h = len(df.index)/len(self.grad_weight)
        for key in self.grad_weight:
            self.grad_weight[key] = h/self.grad_weight[key]

        print(self.grad_weight)

    def nameToClassId(self, name):
        return self.d[name]


    def idToClassName(self, id):
        for key in self.d:
            if self.d[key] == id:
                return key
