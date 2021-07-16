import numpy as np
from node import Node


def create_dummy(n):
    return [np.random.rand(512) for i in range(n)]


def cos_distance(descriptor_1, descriptor_2):
    return 1 - descriptor_1 @ descriptor_2 / (np.linalg.norm(descriptor_1) * np.linalg.norm(descriptor_2))


def whispers(descriptors, threshold=0.5):
    # input: descriptor; shape:(n, 512),  vectors, output:

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
    #repeat_stop = 300

    graph = [Node(i, np.where(adj_mat[i, :] != 0)[
             0], np.reshape(descriptors[i], [512])) for i in range(n)]

    cluster_count = []

    while(True):
        neighbors = np.where(adj_mat[index, :] != 0)[
            0]  # find 1s in this vector
        closest_neighbor = -1
        closest_dist = -1

        for ne in neighbors:
            #curr_descriptor = np.reshape(descriptors[index], [512])
            #ne_descriptor = np.reshape(descriptors[ne], [512])
            #new_dist = cos_distance(curr_descriptor, ne_descriptor)

            new_dist = adj_mat[index, ne]

            if new_dist > closest_dist:  # new_dist is the weight, so it should be greater
                closest_dist = new_dist
                closest_neighbor = ne

        # if graph[index].label == graph[closest_neighbor].label:
            #repeat_count += 1

        if closest_neighbor != -1:
            graph[index].label = graph[closest_neighbor].label

        index += 1
        if index == n:
            index = 0

        '''if repeat_count >= 5:
            print("stopping due to repeat")
            break'''

        cluster_count.append(len({n.label for n in graph}))
        stopping_count += 1
        if stopping_count >= stopping_point:
            print("stopping due to stopping point")
            break

    return graph, adj_mat, cluster_count
