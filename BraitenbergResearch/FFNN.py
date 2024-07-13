import numpy as np


# -----------------------
# Transfer functions
# -----------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)


# ----------------------------------------------
# Feedforward Artificial Neural Network (v2)
# Same architecture as before: Input layer, hidden layer, and output layer
# This time implemented more efficiently using dot products
# ----------------------------------------------
class ANNv2:

    def __init__(self, NIU, NHU, NOU):
        self.nI = NIU
        self.nH = NHU
        self.nO = NOU
        self.wIH = np.random.normal(0,5,size=(NIU,NHU)) #np.zeros((NIU,NHU))
        self.wHO = np.random.normal(0,5,size=(NHU,NOU)) #np.zeros((NHU,NOU))
        self.bH = np.random.normal(0,5,size=NHU) #np.zeros(NHU)
        self.bO = np.random.normal(0,5,size=NOU) #np.zeros(NOU)
        self.HiddenActivation = np.zeros(NHU)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

    def step(self, Input):
        self.Input = np.array(Input)
        self.HiddenActivation = sigmoid(np.dot(self.Input.T, self.wIH) + self.bH)
        self.OutputActivation = sigmoid(np.dot(self.HiddenActivation, self.wHO) + self.bO)
        return self.OutputActivation

    def output(self):
        return self.OutputActivation

    def update_parameters(self, genotype):
        self.wIH = 10 * genotype[:self.nI * self.nH].reshape((self.nI, self.nH))
        self.wHO = 10 * genotype[self.nI * self.nH:self.nI * self.nH + self.nH * self.nO].reshape((self.nH, self.nO))
        self.bH = 10 * genotype[self.nI * self.nH + self.nH * self.nO:self.nI * self.nH + self.nH * self.nO + self.nH]
        self.bO = 10 * genotype[self.nI * self.nH + self.nH * self.nO + self.nH:]
        self.HiddenActivation = 5*genotype

