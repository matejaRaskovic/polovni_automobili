import torch.nn as nn
import torch
import numpy as np

class SeatMaterialFeature():
    num_classes = 2
    # d = {'Štof': 0, 'Prirodna koža': 1, 'Kombinovana koža': 1, 'Velur': 2, 'Drugi': 3}
    d = {'Štof': 0, 'Koža': 1}  # for testing with oversampling due to imbalance
    grad_weight = {}

    def __init__(self):
        pass

    def validDataMaskFromDF(self, df):
        return df['materijal_enterijera'].isin(['Štof', 'Prirodna koža', 'Kombinovana koža'])
        # leathers will be merged into one in dataset

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
        if dbg and np.random.random(1) < 0.04:
            print('\nSeat material')
            print(vec)
            print(target)

        return lossFun(vec, target) # *weight

    def nameToClassId(self, name):
        return self.d[name]


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
        self.grad_weight = {1: 0.9933211900425015, 0: 1.0067692307692309}
        return self.grad_weight[self.d[sample]]

    def getConfMat(self, vector, target):
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        tgt = target.cpu().detach().numpy().astype(int)
        est = vector.cpu().detach().numpy()
        est = np.argmax(est[0, :self.num_classes]).astype(int)
        conf_mat[tgt, est] = 1
        return conf_mat