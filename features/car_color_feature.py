import torch.nn as nn
import torch
import numpy as np

class CarColorFeature():
    # bela, bez i krem u svetlu
    # crna, bordo, teget, braon u tamnu
    # ostale u obojen

    # num_classes = 6
    num_classes = 3
    d = {'Crna': 0, 'Braon': 0, 'Teget': 0, 'Smeđa': 0,  # dark
         'Bež': 1, 'Bela': 1, 'Krem': 1,  # light
         'Bordo': 2, 'Crvena': 2, 'Kameleon': 2, 'Ljubičasta': 2, 'Narandžasta': 2,   # colorful
         'Plava': 2, 'Tirkiz': 2, 'Zelena': 2, 'Zlatna': 2, 'Žuta': 2}
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        # return df['boja'].isin(['Crna', 'Bela', 'Siva', 'Plava', 'Crvena'])
        return df['boja'].isin(['Crna', 'Braon', 'Teget', 'Smeđa', 'Bež', 'Bela', 'Krem', 'Bordo', 'Crvena', 'Kameleon',
                                'Ljubičasta', 'Narandžasta', 'Plava', 'Tirkiz', 'Zelena', 'Zlatna', 'Žuta'])

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
        if dbg and np.random.random(1) < 0.025:
            print('\nOutside color')
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
            if self.d[sample] in self.grad_weight:
                self.grad_weight[self.d[sample]] += 1
            else:
                self.grad_weight[self.d[sample]] = 1

        print(self.grad_weight)
        h = len(df.index) / len(self.grad_weight)
        for key in self.grad_weight:
            self.grad_weight[key] = h / self.grad_weight[key]

        print(self.grad_weight)

    def getWeightForSample(self, sample):
        # self.grad_weight = {'Bela': 1.7183723797780517, 'Crvena': 4.389291338582677, 'Siva': 0.5013851412124483, 'Crna': 0.636783184829792, 'Plava': 1.5990820424555363}
        self.grad_weight = {0: 0.7937894226103833, 2: 0.972073677956031, 1: 1.4054982817869417 - 0.2}
        return self.grad_weight[self.d[sample]]
        # return 1