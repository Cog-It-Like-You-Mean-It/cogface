import numpy as np
class Profile:
    def __init__(self, name):
        self.name = name
        self.descriptors = []
        self.mean_descriptor = 0
        
    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)
        self.mean_descriptor = np.mean(self.descriptors, axis=0)