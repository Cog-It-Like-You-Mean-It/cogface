import numpy as np
from node import Node


THRESHOLD_VALUE = 0.5  # TODO: replace


def create_dummy(n):
    return [np.random.rand(512) for i in range(n)]


def cos_distance(descriptor_1, descriptor_2):
    return 1 - descriptor_1 @ descriptor_2 / (np.linalg.norm(descriptor_1) * np.linalg.norm(descriptor_2))


def whispers(descriptors):
    # input: descriptor; shape:(n, 512),  vectors, output:
    threshold = THRESHOLD_VALUE

    n = len(descriptors)
    adj_mat = np.zeros([n, n])

    for d in range(n):
        for d_2 in range(d+1, n):
            descriptor1 = np.reshape(descriptors[d], [512])
            descriptor2 = np.reshape(descriptors[d_2], [512])
            dist = cos_distance(descriptor1, descriptor2)
            if dist <= threshold:
                weight = 1 / (dist**2)
                #adj_mat[d, d_2] = 1
                #adj_mat[d_2, d] = 1
                adj_mat[d, d_2] = weight
                adj_mat[d_2, d] = weight

    # label = np.arange(n)  # turn into groups

    index = 0
    stopping_count = 0
    stopping_point = 1000
    repeat_count = 0
    repeat_stop = 5

    graph = [Node(i, np.where(adj_mat[i, :] != 0)[
             0], np.reshape(descriptors[i], [512])) for i in range(n)]

    while(True):
        neighbors = np.where(adj_mat[index, :] != 0)[
            0]  # find 1s in this vector
        closest_neighbor = -1
        closest_dist = 10000000

        for ne in neighbors:
            #curr_descriptor = np.reshape(descriptors[index], [512])
            #ne_descriptor = np.reshape(descriptors[ne], [512])
            #new_dist = cos_distance(curr_descriptor, ne_descriptor)

            new_dist = adj_mat[index, ne]

            if new_dist > closest_dist:  # new_dist is the weight, so it should be greater
                closest_dist = new_dist
                closest_neighbor = ne

        if n == closest_neighbor:
            repeat_count += 1
            n = closest_neighbor

        index += 1
        stopping_count += 1
        if index == n:
            index = 0

        if repeat_count >= 5:
            break
        if stopping_count >= stopping_point:
            break

        return graph, adj_mat


data = create_dummy(10)
graph, adj_mat = whispers(data)
