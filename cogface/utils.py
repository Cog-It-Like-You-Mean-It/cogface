
def cos_distance(descriptor_1, descriptor_2):
    import numpy as np
    return 1 - descriptor_1 @ descriptor_2 / (np.linalg.norm(descriptor_1) * np.linalg.norm(descriptor_2))
