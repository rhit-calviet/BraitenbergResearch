import numpy as np


class Heat:
    def __init__(self):
        angle = np.random.random() * 2 * np.pi
        self.x = 5.0 #* np.cos(angle)
        self.y = 8.0 #* np.sin(angle)