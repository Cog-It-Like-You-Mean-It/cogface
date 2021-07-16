import numpy as np
import pickle
from facenet_models import FacenetModel
import skimage.io as io
from camera import take_picture
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Database import Database

db = Database()

def upload_picture(file_path):
    image = io.imread(file_path)
    if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    return image

def detect_faces(image):
    model = FacenetModel()
    boxes, probabilities, landmarks = model.detect(image)
    descriptor = model.compute_descriptors(image, boxes)

    fig, ax = plt.subplots()
    ax.imshow(image)

    for des, box, prob, landmark in zip(descriptor, boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))
    
        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        #for i in range(len(landmark)):
            #ax.plot(landmark[i, 0], landmark[i, 1], "+", color="blue")
        
        # add name labels to each box
        plt.text(box[:2][0],box[:2][1], db.find_match(des),color="white")

def detect_from_camera():
    img = take_picture()
    detect_faces(img)

def detect_from_file(file_path):
    detect_faces(upload_picture(file_path))