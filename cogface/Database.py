import numpy as np
import pickle
from facenet_models import FacenetModel
from Profile import Profile

class Database:
     
    # initialize database
    def __init__(self):
        self.database = {}
        self.model = FacenetModel()
        
    # add image descriptor to profile, or create profile if it doesn't exist    
    def add_image(self, name, img):
        boxes, probabilities, landmarks = self.model.detect(img)
        descriptor = self.model.compute_descriptors(img, boxes)
        if name in self.database:
            self.database[name].add_descriptor(descriptor)
        else:
            profile = Profile(name)
            profile.add_descriptor(descriptor)
            self.database[name] = profile
     
    def remove_profile(self, name):
        self.database.pop(name)
        
    def load_database(self):
        with open("./database.pkl", mode="rb") as db:
            self.database = pickle.load(db)
            
    def save_database(self):
        with open("./database.pkl", mode="wb") as db:
            pickle.dump(self.database, db)
            
    # compare input to mean descriptors in database         
    def find_match(self, descriptor):
        #to change
        cutoff = 0.3
        dists = []
        names = []
        for name, value in self.database.items():
            dists.append(cos_distance(value.mean_descriptor, descriptor))
            names.append(name)
        if np.min(dists) < cutoff:
            return names[np.argmin(dists)]
        return "Unknown??"

def cos_distance(descriptor_1, descriptor_2):
    return 1 - descriptor_1 @ descriptor_2 / (np.linalg.norm(descriptor_1) * np.linalg.norm(descriptor_2))