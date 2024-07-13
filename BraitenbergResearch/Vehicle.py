import numpy as np

from FFNN import ANNv2


class Vehicle:
    def __init__(self, NI, NH, NO):
        self.xPos = 0.0
        self.yPos = 0.0
        self.orientation = np.pi
        self.velocity = 0.0
        self.radius = 1.0
        self.leftSensor = 0.0
        self.rightSensor = 0.0
        self.rightHeatSensor = 0.0
        self.leftHeatSensor = 0.0
        self.rightMotor = 0.0
        self.leftMotor = 0.0
        self.angleOffset = np.pi / 2
        self.rs_xPos = self.radius * np.cos(self.orientation + self.angleOffset)
        self.rs_yPos = self.radius * np.sin(self.orientation + self.angleOffset)
        self.ls_xPos = self.radius * np.cos(self.orientation - self.angleOffset)
        self.ls_yPos = self.radius * np.sin(self.orientation - self.angleOffset)

        # Neural network initiation
        self.brain = ANNv2(NI, NH, NO)

    def sense(self, light):
        self.leftSensor = 1 - np.sqrt((self.ls_xPos - light.x) ** 2 + (self.ls_yPos - light.y) ** 2) / 10
        self.leftSensor = np.clip(self.leftSensor, 0, 1)
        self.rightSensor = 1 - np.sqrt((self.rs_xPos - light.x) ** 2 + (self.rs_yPos - light.y) ** 2) / 10
        self.rightSensor = np.clip(self.rightSensor, 0, 1)

    def think(self):
        # Prepare input vector for the neural network
        input_vector = [self.leftSensor, self.rightSensor]
        # Use the neural network to compute motor outputs
        motor_outputs = self.brain.step(input_vector)
        # print(motor_outputs)
        # Assign motor outputs to vehicle motors
        self.rightMotor = motor_outputs[1]
        self.leftMotor = motor_outputs[0]

    def move(self):
        self.rightMotor = np.clip(self.rightMotor, 0, 1)
        self.leftMotor = np.clip(self.leftMotor, 0, 1)
        self.orientation += ((self.leftMotor - self.rightMotor) / 10)  # + np.random.normal(0, 0.1)
        self.velocity = (self.leftMotor + self.rightMotor) / 10
        self.xPos += self.velocity * np.cos(self.orientation)
        self.yPos += self.velocity * np.sin(self.orientation)
        self.rs_xPos = self.xPos + self.radius * np.cos(self.orientation + self.angleOffset)
        self.rs_yPos = self.yPos + self.radius * np.sin(self.orientation + self.angleOffset)
        self.ls_xPos = self.xPos + self.radius * np.cos(self.orientation - self.angleOffset)
        self.ls_yPos = self.yPos + self.radius * np.sin(self.orientation - self.angleOffset)

    def update(self, light):
        self.sense(light)
        self.think()
        self.move()

    def distanceLight(self, light):
        return np.sqrt((self.xPos - light.x) ** 2 + (self.yPos - light.y) ** 2)

    def setController(self, genotype):
        self.brain.update_parameters(genotype)