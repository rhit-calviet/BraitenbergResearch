import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)


class ANNv3:
    def __init__(self, NIU, NHU, NOU, genotype):
        self.nI = NIU
        self.nH = NHU
        self.nO = NOU
        self.wIH = np.random.normal(0, 5, size=(NIU, NHU))
        self.wHO = np.random.normal(0, 5, size=(NHU, NOU))
        self.bH = np.random.normal(0, 5, size=NHU)
        self.bO = np.random.normal(0, 5, size=NOU)
        self.HiddenActivation = np.zeros(NHU)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)
        self.hidden_activation_functions = self.parse_activation_functions(genotype[:NHU])
        self.output_activation_functions = self.parse_activation_functions(genotype[NHU:NHU + NOU])

    def parse_activation_functions(self, genotype_slice):
        activation_functions = []
        for gene_value in genotype_slice:
            # Example logic: if value is less than 0, use sigmoid, else use relu
            if gene_value < 0:
                activation_functions.append(self.sigmoid)
            else:
                activation_functions.append(self.relu)
        return activation_functions




    def step(self, Input):
        self.Input = np.array(Input)

        # Calculate hidden layer activations
        hidden_layer_input = np.dot(self.Input.T, self.wIH) + self.bH
        self.HiddenActivation = np.array(
            [self.hidden_activation_functions[i](hidden_layer_input[i]) for i in range(self.nH)])

        # Calculate output layer activations
        output_layer_input = np.dot(self.HiddenActivation, self.wHO) + self.bO
        self.OutputActivation = np.array(
            [self.output_activation_functions[i](output_layer_input[i]) for i in range(self.nO)])

        return self.OutputActivation

    def output(self):
        return self.OutputActivation

    def update_parameters(self, genotype):
        self.wIH = 10 * genotype[:self.nI * self.nH].reshape((self.nI, self.nH))
        self.wHO = 10 * genotype[self.nI * self.nH:self.nI * self.nH + self.nH * self.nO].reshape((self.nH, self.nO))
        self.bH = 10 * genotype[self.nI * self.nH + self.nH * self.nO:self.nI * self.nH + self.nH * self.nO + self.nH]
        self.bO = 10 * genotype[self.nI * self.nH + self.nH * self.nO + self.nH:]