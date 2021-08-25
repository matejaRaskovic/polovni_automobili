import torch.nn as nn
import torch
import numpy as np

class CarProducerFeature():
    num_classes = 4
    # d = {'Volkswagen': 0, 'Audi': 1, 'BMW': 2, 'Opel': 3, 'Peugeot': 4, 'Fiat': 5}  # , 'Renault': 6, 'Mercedes Benz': 7}
    d = {'Volkswagen': 0, 'Peugeot': 1, 'Fiat': 2, 'BMW': 3}
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['marka'].isin(['Volkswagen', 'Peugeot', 'Fiat', 'BMW'])  # , 'Opel', 'Peugeot', 'Fiat'])  # , 'Renault', 'Mercedes Benz'])i

    def name(self):
        return 'marka'

    def pos(self):
        return 0

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

    def calculateGradWeight(self, df):
        for sample in df[self.name()]:
            if sample in self.grad_weight:
                self.grad_weight[sample] += 1
            else:
                self.grad_weight[sample] = 1

        print(self.grad_weight)
        h = len(df.index)/len(self.grad_weight)
        for key in self.grad_weight:
            self.grad_weight[key] = h/self.grad_weight[key]

        print(self.grad_weight)

    def getWeightForSample(self, sample):
        # self.grad_weight = {'Volkswagen': 0.6157781367797522, 'Audi': 0.8940578577013292, 'BMW': 0.942315615986815, 'Opel': 1.1782586295723854, 'Peugeot': 1.243610657966286, 'Fiat': 1.839903459372486}
        # return self.grad_weight[sample]
        return 1

    def nameToClassId(self, name):
        return self.d[name]


    def idToClassName(self, id):
        for key in self.d:
            if self.d[key] == id:
                return key

    def getConfMat(self, vector, target):
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        tgt = target.cpu().detach().numpy()
        print(tgt)
        est = vector.cpu().detach().numpy()
        print(est)
        exit(1)