import numpy as np
import pickle
from facenet_models import FacenetModel
from Profile import Profile
from utils import cos_distance

class Database:
     
    # initialize database
    def __init__(self):
        self.database = {}
        self.model = FacenetModel()
        
    # add image descriptor to profile, or create profile if it doesn't exist    
    def add_image(self, name, img):
        if img.shape[-1] > 4000 or img.shape[-2] > 4000:
            img = img[::4, ::4]
        boxes, probabilities, landmarks = self.model.detect(img)
        idxs = []
        for prob in probabilities:
            if prob >= 0.98:
                idxs.append(np.where(probabilities == prob)[0])
        descriptor = self.model.compute_descriptors(img, boxes[idxs])
        if descriptor.shape[0] == 1:
            if name in self.database:
                self.database[name].add_descriptor(descriptor)
            else:
                profile = Profile(name)
                profile.add_descriptor(descriptor)
                self.database[name] = profile
            self.save_database()
     
    def remove_profile(self, name):
        self.database.pop(name)
        
    def load_database(self):
        with open("./database.pkl", mode="rb") as db:
            self.database = pickle.load(db)
            
    def save_database(self):
        with open("./database.pkl", mode="wb") as db:
            pickle.dump(self.database, db)
            
    def create_database(self):
        self.database = {}
        
    # compare input to mean descriptors in database         
    def find_match(self, descriptor):
        cutoff = 0.6
        dists = []
        names = []
        for name, value in self.database.items():
            dists.append(cos_distance(value.mean_descriptor, descriptor))
            names.append(name)
        if np.min(dists) < cutoff:
            self.database[names[np.argmin(dists)]].add_descriptor(descriptor)
            return names[np.argmin(dists)]
        return "Unknown??"