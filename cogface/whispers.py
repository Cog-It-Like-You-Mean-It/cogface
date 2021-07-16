import numpy as np


THRESHOLD_VALUE = 0.5  # TODO: replace


def create_dummy(n):
    return [np.random.rand(512) for i in range(n)]


def cos_distance(descriptor_1, descriptor_2):
    return 1 - descriptor_1 @ descriptor_2 / (np.linalg.norm(descriptor_1) * np.linalg.norm(descriptor_2))


def whispers(descriptors):
    # input: descriptor; shape:(n, 512),  vectors, output:
    threshold = THRESHOLD_VALUE

    n = len(descriptors)
    adj_mat = np.zeros(n, n)

    for d in range(n):
        for d_2 in range(d+1, n):
            if cos_distance(np.shape(descriptors[d], [512]), np.shape(descriptors[d_2], [512])) <= threshold:
                adj_mat[d, d_2] = 1
                adj_mat[d_2, d] = 1

    label = np.arange(n)  # turn into groups

    count = 0
    while(True):
        neighbors = np.where(adj_mat[count, :] == 1)  # find 1s in this vector
        closest_neighbor = -1
        closest_dist = 10000000

        for ne in neighbors:
            new_dist = cos_distance(
                np.shape(descriptors[count], [512]), np.shape(descriptors[ne], [512]))
            if new_dist < closest_dist:
                closest_dist = new_dist
                closest_neighbor = ne

        label[count] = closest_neighbor

        count += 1
        if count == n:
            count = 0

        #####
        if finished:
            break
        if count > stopping_point:
            break
        ########
